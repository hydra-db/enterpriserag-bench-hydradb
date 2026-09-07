"""Content identity, tolerant journals and ownership-safe locks, shared by every stage.

Identity rules (used by ingestion, generation and judging alike):

* A corpus is identified by its CONTENT, never by a label. For a
  ``documents.sqlite`` that is a streaming sha256 over every row's
  (doc_id, title, content) in doc_id order, computed at use time (never read
  back from metadata). For a benchmark checkout it is the git HEAD plus a
  sha256 over the contents of every file git reports as modified, added or
  deleted under ``generated_data/`` and ``questions.jsonl`` — git already
  tracks content, so a clean tree at a known HEAD is a known corpus, and any
  local edit changes the identity. A stage records the identity it saw at
  start; a stage that reads the corpus while running (judging) checks it again
  at the end and fails if it changed.
* Append-only journals (attempt history, status logs) are read tolerantly: an
  incomplete trailing record left by an interrupted append is quarantined to a
  ``.torn-<timestamp>`` file and the journal is truncated to the last complete
  record before anything is appended again. Corruption in the middle of a
  journal is an error, never skipped.
* Locks are created atomically WITH their owner record (temp file + hard link),
  so a lock file is never observed empty; release is owner-checked; a lock
  whose owner process is dead is removed and re-acquired with a printed note.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

# ------------------------------------------------------------ corpus ----

def content_sha256_sqlite(db_path: str | Path) -> dict:
    """Streaming sha256 over (doc_id, title, content) in doc_id order."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        h = hashlib.sha256()
        n = 0
        for doc_id, title, content in conn.execute(
                "SELECT doc_id, title, content FROM documents ORDER BY doc_id"):
            h.update((doc_id or "").encode("utf-8")); h.update(b"\x1f")
            h.update((title or "").encode("utf-8")); h.update(b"\x1f")
            h.update((content or "").encode("utf-8")); h.update(b"\n")
            n += 1
        return {"content_sha256": h.hexdigest(), "rows": n}
    finally:
        conn.close()


def checkout_state(root: str | Path,
                   paths_of_interest: tuple[str, ...] = ("generated_data", "questions.jsonl")) -> dict:
    """HEAD plus a content hash of everything git sees as changed under the
    corpus paths. Untracked files count as changes."""
    root = Path(root)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    head = git.stdout.strip() if git.returncode == 0 else ""
    if head:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths_of_interest],
                                cwd=root, capture_output=True, text=True).stdout
        dirty = sorted(line[3:] for line in status.splitlines() if line.strip())
    else:
        # not a git checkout (e.g. a test fixture): every corpus file is "dirty", i.e. hashed by content
        dirty = sorted(str(p.relative_to(root)) for poi in paths_of_interest
                       for p in ([root / poi] if (root / poi).is_file() else (root / poi).rglob("*")) if p.is_file())
    h = hashlib.sha256()
    for rel in dirty:
        h.update(rel.encode("utf-8")); h.update(b"\x1f")
        p = root / rel
        if p.is_file():
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(b"<deleted>")
        h.update(b"\n")
    dirty_sha = h.hexdigest() if dirty else None
    ident = hashlib.sha256(f"{head}|{dirty_sha}".encode()).hexdigest()
    return {"head": head, "dirty": dirty, "dirty_content_sha256": dirty_sha, "corpus_identity": ident}


# ----------------------------------------------------------- journals ----

def read_journal(path: str | Path, *, repair: bool = True) -> list[dict]:
    """Read a JSON-lines journal. A torn trailing record (interrupted append)
    is quarantined and truncated away when ``repair`` is True; a malformed
    record anywhere else raises ``ValueError``."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return []
    lines = data.split(b"\n")
    complete = data.endswith(b"\n")
    body, tail = (lines[:-1], b"") if complete else (lines[:-1], lines[-1])
    rows: list[dict] = []
    for i, line in enumerate(body):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(body) - 1 and not tail:
                tail = line   # last line has a newline but is not valid JSON: treat as torn
                body = body[:-1]
                break
            raise ValueError(f"{path}: malformed record at line {i + 1} ({exc.msg}); "
                             "not repairing mid-journal corruption") from exc
    if tail:
        if not repair:
            raise ValueError(f"{path}: incomplete trailing record")
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        quarantine = path.with_name(path.name + f".torn-{stamp}")
        with open(quarantine, "ab") as q:
            q.write(tail + b"\n")
        keep = b"\n".join(body) + (b"\n" if body else b"")
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(keep)
        os.replace(tmp, path)
        print(f"  journal: quarantined an incomplete trailing record of {path.name} to {quarantine.name}", flush=True)
    return rows


def append_journal(path: str | Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# -------------------------------------------------------------- locks ----

class Locked(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: str | Path, *, what: str = "run") -> str:
    """Create ``path`` atomically with its owner record. Returns a token that
    ``release_lock`` requires. A lock whose owner pid is dead is taken over."""
    path = Path(path)
    token = uuid.uuid4().hex
    owner = {"pid": os.getpid(), "token": token, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "host": os.uname().nodename if hasattr(os, "uname") else "", "what": what}
    tmp = path.with_name(path.name + f".{token}.tmp")
    tmp.write_text(json.dumps(owner), encoding="utf-8")
    try:
        for _ in range(2):
            try:
                os.link(tmp, path)     # atomic publish: the lock never exists without its record
                return token
            except FileExistsError:
                try:
                    cur = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raise Locked(f"{path} is held and its owner record is unreadable; not taking it over") from None
                if _pid_alive(int(cur.get("pid", -1))):
                    raise Locked(f"{path} is held by pid {cur.get('pid')} since {cur.get('started_at')}") from None
                print(f"  lock: owner pid {cur.get('pid')} is not running; clearing stale {path.name}", flush=True)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        raise Locked(f"{path}: could not acquire after clearing a stale lock")
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def release_lock(path: str | Path, token: str) -> None:
    """Remove the lock only if we own it."""
    path = Path(path)
    try:
        cur = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if cur.get("token") == token and cur.get("pid") == os.getpid():
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
