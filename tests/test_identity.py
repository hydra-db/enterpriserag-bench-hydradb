"""Content identity, tolerant journals and ownership-safe locks (release re-review of 0d7c6ba)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from erb_hydradb import corpus, hydrate, identity

FIXTURE_REPO = Path(__file__).resolve().parents[1] / "tests" / "data" / "erb_repo"


def _git_checkout(tmp_path):
    root = tmp_path / "erb"
    root.mkdir()
    (root / "generated_data").mkdir()
    (root / "generated_data" / "doc1.json").write_text('{"title": "T", "content": "hello"}')
    (root / "questions.jsonl").write_text('{"question_id": "q1"}\n')
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "init"]):
        subprocess.run(cmd, cwd=root, check=True, env=env, capture_output=True)
    return root


def test_checkout_identity_changes_with_document_content_not_only_index(tmp_path):
    root = _git_checkout(tmp_path)
    clean = identity.checkout_state(root)
    assert clean["head"] and clean["dirty"] == [] and clean["dirty_content_sha256"] is None
    (root / "generated_data" / "doc1.json").write_text('{"title": "T", "content": "CHANGED"}')
    changed = identity.checkout_state(root)
    assert changed["head"] == clean["head"]
    assert changed["dirty"] == ["generated_data/doc1.json"]
    assert changed["corpus_identity"] != clean["corpus_identity"]
    # same content again -> same identity (deterministic), and an untracked file counts
    assert identity.checkout_state(root)["corpus_identity"] == changed["corpus_identity"]
    (root / "generated_data" / "new.json").write_text("{}")
    assert identity.checkout_state(root)["corpus_identity"] != changed["corpus_identity"]
    # a change outside the corpus paths does not
    (root / "README").write_text("x")
    assert identity.checkout_state(root)["dirty"] == ["generated_data/doc1.json", "generated_data/new.json"]


def test_sqlite_identity_is_a_content_hash(tmp_path):
    db = tmp_path / "documents.sqlite"
    corpus.build_from_repo(FIXTURE_REPO, db, quiet=True)
    a = identity.content_sha256_sqlite(db)
    assert a["rows"] > 0
    store = hydrate.DocumentStore(db, None)
    assert store.identity["sqlite"]["content_sha256"] == a["content_sha256"]
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute("UPDATE documents SET content = content || ' changed' WHERE doc_id = (SELECT MIN(doc_id) FROM documents)")
    conn.commit()
    conn.close()
    b = identity.content_sha256_sqlite(db)
    assert b["rows"] == a["rows"] and b["content_sha256"] != a["content_sha256"]
    assert hydrate.DocumentStore(db, None).identity["sqlite"]["content_sha256"] == b["content_sha256"]


def test_read_journal_quarantines_only_a_torn_tail(tmp_path):
    j = tmp_path / "j.jsonl"
    j.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b": "tor')
    rows = identity.read_journal(j)
    assert rows == [{"a": 1}, {"a": 2}]
    assert j.read_text() == '{"a": 1}\n{"a": 2}\n'
    torn = list(tmp_path.glob("j.jsonl.torn-*"))
    assert len(torn) == 1 and torn[0].read_text().startswith('{"a": 3')
    identity.append_journal(j, {"a": 3})
    assert identity.read_journal(j) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert identity.read_journal(tmp_path / "missing.jsonl") == []
    j.write_text('{"a": 1}\n{"a": 2')
    with pytest.raises(ValueError, match="incomplete trailing record"):
        identity.read_journal(j, repair=False)


def test_lock_double_acquire_is_deterministic_and_release_is_owner_checked(tmp_path):
    """P2-4 probe: two acquirers of the same path, exactly one succeeds, and the loser's
    release must not remove the winner's lock. The interleaving is forced by making the
    second acquirer run inside the first one's publish step."""
    lock = tmp_path / "run.lock"
    outcomes = []
    real_link = os.link

    def racing_link(src, dst):
        real_link(src, dst)                     # first acquirer publishes...
        try:                                    # ...and a second acquirer runs before it returns
            outcomes.append(("second", identity.acquire_lock(lock, what="test")))
        except identity.Locked as exc:
            outcomes.append(("second", exc))

    with patch.object(os, "link", racing_link):
        first = identity.acquire_lock(lock, what="test")
    assert len(outcomes) == 1 and isinstance(outcomes[0][1], identity.Locked)
    assert json.loads(lock.read_text())["token"] == first
    identity.release_lock(lock, "not-the-owner")
    assert lock.exists()                        # loser's release is a no-op
    identity.release_lock(lock, first)
    assert not lock.exists()
    # no temp files left behind
    assert list(tmp_path.glob("run.lock.*")) == []


def test_stale_lock_from_a_dead_process_is_taken_over(tmp_path):
    lock = tmp_path / "run.lock"
    lock.write_text(json.dumps({"pid": 2**22 + 12345, "token": "x", "started_at": "2026-01-01T00:00:00Z"}))
    with patch.object(identity, "_pid_alive", return_value=False):
        tok = identity.acquire_lock(lock)
    assert json.loads(lock.read_text())["token"] == tok
    with pytest.raises(identity.Locked):
        identity.acquire_lock(lock)
    identity.release_lock(lock, tok)
