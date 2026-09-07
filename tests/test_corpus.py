"""Corpus build/read tests. Hermetic: five sample documents under tests/data, no network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from erb_hydradb import corpus, paths

DATA = Path(__file__).parent / "data"
ERB_REPO = DATA / "erb_repo"
SOURCES = ERB_REPO / "generated_data" / "sources"

SAMPLE_FILES = {
    "slack": SOURCES / "slack" / "1710501234-region-placement-costs.json",
    "gmail": SOURCES / "gmail" / "20250610-private-upgrade-audit-log-requirements.json",
    "jira": SOURCES / "jira" / "customer-support"
    / "SUP-18588-audit-export-filtering-questions-and-category-list.json",
    "confluence": SOURCES / "confluence" / "eng-infra" / "gpu-fleet-and-capacity"
    / "dedicated-mi300-canada-east-capacity-and-reservations.json",
    "hubspot": SOURCES / "hubspot" / "company-aurora-payments.json",
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db_path(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("corpus") / "documents.sqlite"
    n = corpus.build_from_repo(ERB_REPO, p, quiet=True)
    assert n == len(SAMPLE_FILES)
    return p


# -- canonical extraction ------------------------------------------------------
def test_extract_multi_field_matches_hand_computed_rule():
    d = _load(SAMPLE_FILES["jira"])
    assert d["title_field_name"] == "summary"
    assert d["content_field_names"] == [
        "description", "comments", "resolution", "customer_facing_message", "internal_notes", "next_steps"]
    title, content = corpus.extract_document_content(d)
    assert title == d["summary"]
    expected = "\n\n".join([
        "description:\n" + d["description"],
        "comments:\n" + "\n".join(d["comments"]),      # list -> newline-joined
        "resolution:\n" + d["resolution"],
        "customer_facing_message:\n" + d["customer_facing_message"],
        "internal_notes:\n" + d["internal_notes"],
        "next_steps:\n" + d["next_steps"],
    ])
    assert content == expected


def test_extract_single_field_is_the_raw_value():
    d = _load(SAMPLE_FILES["slack"])
    title, content = corpus.extract_document_content(d)
    assert title == d["channel"]
    assert content == d["messages"]          # no "messages:" header for a single field


def test_extract_rejects_bad_labels():
    with pytest.raises(corpus.DocumentFieldError):
        corpus.extract_document_content({"content_field_names": ["a"], "a": "x"})
    with pytest.raises(corpus.DocumentFieldError):
        corpus.extract_document_content({"title_field_name": "t", "t": "x", "content_field_names": []})
    with pytest.raises(corpus.DocumentFieldError):
        corpus.extract_document_content({"title_field_name": "t", "t": "x", "content_field_names": ["nope"]})


# -- building -----------------------------------------------------------------
def test_repo_rows_use_uuid_and_top_level_dir():
    rows = {r[0]: r for r in corpus.iter_repo_rows(ERB_REPO)}
    assert len(rows) == len(SAMPLE_FILES)
    for source_type, path in SAMPLE_FILES.items():
        raw = _load(path)
        doc_id, st, title, content, content_len = rows[raw["dataset_doc_uuid"]]
        assert doc_id.startswith("dsid_")
        assert st == source_type
        assert (title, content) == corpus.extract_document_content(raw)
        assert content_len == len(content)


def test_hf_and_repo_builders_write_identical_rows(tmp_path):
    hf_rows = [
        {"doc_id": r[0], "source_type": r[1], "title": r[2], "content": r[3]}
        for r in corpus.iter_repo_rows(ERB_REPO)
    ]
    repo_db = tmp_path / "repo.sqlite"
    hf_db = tmp_path / "hf.sqlite"
    corpus.build_from_repo(ERB_REPO, repo_db, quiet=True)
    corpus.build_from_hf(hf_db, dataset=hf_rows, quiet=True)
    assert list(corpus.iter_documents(repo_db)) == list(corpus.iter_documents(hf_db))


def test_hf_builder_treats_missing_content_as_empty(tmp_path):
    db = tmp_path / "x.sqlite"
    corpus.build_from_hf(db, dataset=[{"doc_id": "dsid_x", "source_type": "slack", "title": "t", "content": None}],
                         quiet=True)
    row = next(corpus.iter_documents(db))
    assert row["content"] == "" and row["content_len"] == 0


def test_build_is_idempotent_and_limit_applies(tmp_path):
    db = tmp_path / "d.sqlite"
    assert corpus.build_from_repo(ERB_REPO, db, limit=2, quiet=True) == 2
    assert corpus.build_from_repo(ERB_REPO, db, quiet=True) == len(SAMPLE_FILES)
    assert corpus.build_from_repo(ERB_REPO, db, quiet=True) == len(SAMPLE_FILES)


def test_source_type_filter_on_build(tmp_path):
    db = tmp_path / "s.sqlite"
    assert corpus.build_from_repo(ERB_REPO, db, source_types=["slack", "gmail"], quiet=True) == 2
    assert {r["source_type"] for r in corpus.iter_documents(db)} == {"slack", "gmail"}


# -- reading ------------------------------------------------------------------
def test_iter_documents_streams_in_doc_id_order(db_path):
    rows = list(corpus.iter_documents(db_path))
    ids = [r["doc_id"] for r in rows]
    assert ids == sorted(ids)
    assert set(rows[0]) == {"doc_id", "source_type", "title", "content", "content_len"}


def test_iter_documents_filters_limit_and_resume_cursor(db_path):
    assert [r["source_type"] for r in corpus.iter_documents(db_path, source_types=["jira"])] == ["jira"]
    assert len(list(corpus.iter_documents(db_path, limit=2))) == 2
    ids = [r["doc_id"] for r in corpus.iter_documents(db_path)]
    after = [r["doc_id"] for r in corpus.iter_documents(db_path, after_doc_id=ids[1])]
    assert after == ids[2:]
    assert corpus.count_documents(db_path, after_doc_id=ids[1]) == len(ids) - 2


def test_fetch_documents(db_path):
    ids = [r["doc_id"] for r in corpus.iter_documents(db_path, limit=2)]
    got = corpus.fetch_documents(db_path, ids + ["dsid_missing"])
    assert set(got) == set(ids)
    assert corpus.fetch_documents(db_path, []) == {}


def test_document_stats(db_path):
    stats = corpus.document_stats(db_path)
    assert stats["total"] == len(SAMPLE_FILES)
    by = {row["source_type"]: row for row in stats["by_source_type"]}
    assert set(by) == set(SAMPLE_FILES)
    for row in by.values():
        assert row["count"] == 1
        assert row["min_content_len"] == row["max_content_len"] == row["avg_content_len"] > 0


def test_missing_db_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(corpus.iter_documents(tmp_path / "nope.sqlite"))


# -- questions ----------------------------------------------------------------
def test_copy_questions_rejects_hash_mismatch(tmp_path):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "questions.jsonl").write_text('{"question_id": "qst_0001"}\n', encoding="utf-8")
    with pytest.raises(corpus.QuestionsMismatchError):
        corpus.copy_questions(fake_repo, tmp_path / "out" / "questions.jsonl")
    assert not (tmp_path / "out" / "questions.jsonl").exists()


def test_copy_questions_accepts_pinned_hash(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    body = '{"question_id": "qst_0001"}\n'
    (fake_repo / "questions.jsonl").write_text(body, encoding="utf-8")
    monkeypatch.setattr(paths, "QUESTIONS_SHA256", hashlib.sha256(body.encode()).hexdigest())
    dest = corpus.copy_questions(fake_repo, tmp_path / "out" / "questions.jsonl")
    assert dest.read_text(encoding="utf-8") == body
    assert corpus.load_questions(dest) == [{"question_id": "qst_0001"}]


def test_copy_questions_verifies_the_real_pinned_file(tmp_path):
    """tests/data/questions.jsonl is the official 500-question file at the pinned commit."""
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / "questions.jsonl").write_bytes((DATA / "questions.jsonl").read_bytes())
    dest = corpus.copy_questions(fake_repo, tmp_path / "out" / "questions.jsonl")
    assert corpus.verify_questions(dest) == paths.QUESTIONS_SHA256
    questions = corpus.load_questions(dest)
    assert len(questions) == 500
    assert {q["question_id"] for q in questions} == {f"qst_{i:04d}" for i in range(1, 501)}


def test_copy_questions_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpus.copy_questions(tmp_path, tmp_path / "q.jsonl")
