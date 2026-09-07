"""Build and read the local corpus: ``documents.sqlite`` and ``questions.jsonl``.

The benchmark ships ~512k documents. Everything downstream (ingestion,
answer-context hydration) needs random access by ``doc_id`` and cheap filtering
by ``source_type``, so the corpus is streamed once into a SQLite file:

    documents(doc_id TEXT PRIMARY KEY, source_type TEXT, title TEXT,
              content TEXT, content_len INTEGER)

Two builders produce identical rows:

* ``build_from_hf``   — streams the Hugging Face dataset
  (``onyx-dot-app/EnterpriseRAG-Bench``, config ``documents``, split ``test``).
  Needs the optional ``corpus`` extra (``datasets``); the import is lazy.
* ``build_from_repo`` — walks ``generated_data/sources/`` in a checkout of the
  benchmark repository and applies the benchmark's own canonical extraction
  rule (``extract_document_content`` below, a line-for-line mirror of
  ``src/utils/document_content.py`` upstream). The Hugging Face rows were
  produced by that same rule (``parquet_format_export.py`` upstream):
  ``doc_id`` is the file's ``dataset_doc_uuid``, ``source_type`` is the
  top-level directory under ``sources/``.

Both are resumable: rows are inserted with ``INSERT OR IGNORE`` on the primary
key, so re-running after an interruption tops the table up.

``questions.jsonl`` is never rebuilt; ``copy_questions`` copies the file from
the checkout and verifies its sha256 against the pinned ``paths.QUESTIONS_SHA256``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from . import paths

# Bulk-insert tuning: commit every N rows so an interrupted build keeps its progress.
COMMIT_EVERY = 5_000
PROGRESS_EVERY = 10_000

SOURCES_SUBDIR = Path("generated_data") / "sources"
QUESTIONS_FILENAME = "questions.jsonl"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source_type  TEXT NOT NULL,
    title        TEXT,
    content      TEXT,
    content_len  INTEGER
);
"""
_INSERT = (
    "INSERT OR IGNORE INTO documents (doc_id, source_type, title, content, content_len) "
    "VALUES (?, ?, ?, ?, ?)"
)
# Corpus identity: which upstream source (and revision) the rows came from.
# A build refuses to add rows from a different identity, so two source
# versions can never be mixed silently.
_CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

Row = tuple[str, str, str, str, int]  # (doc_id, source_type, title, content, content_len)


# ---------------------------------------------------------------------------
# Canonical extraction (mirror of the benchmark's document_content.py)
# ---------------------------------------------------------------------------
class DocumentFieldError(ValueError):
    """Raised when a document's field labels are missing or invalid."""


def extract_document_content(doc_data: dict) -> tuple[str, str]:
    """Return ``(title, content)`` for one raw benchmark document.

    Exact mirror of the benchmark's rule: ``title_field_name`` names the title
    field; ``content_field_names`` lists the content fields. A single content
    field contributes its value as-is; several are rendered as
    ``"<name>:\\n<value>"`` blocks joined by a blank line, with list values
    joined by newlines. Every value passes through ``str()``.
    """
    if "title_field_name" not in doc_data:
        raise DocumentFieldError("Document missing 'title_field_name' field")
    title_field_name = doc_data["title_field_name"]
    if title_field_name not in doc_data:
        raise DocumentFieldError(f"title_field_name '{title_field_name}' not found in document")
    title = str(doc_data[title_field_name])

    if "content_field_names" not in doc_data:
        raise DocumentFieldError("Document missing 'content_field_names' field")
    content_field_names = doc_data["content_field_names"]
    if not isinstance(content_field_names, list):
        raise DocumentFieldError("'content_field_names' must be a list")
    if not content_field_names:
        raise DocumentFieldError("'content_field_names' is empty")
    for field_name in content_field_names:
        if field_name not in doc_data:
            raise DocumentFieldError(f"content_field_name '{field_name}' not found in document")

    if len(content_field_names) == 1:
        content = str(doc_data[content_field_names[0]])
    else:
        content_parts = []
        for field_name in content_field_names:
            value = doc_data[field_name]
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            content_parts.append(f"{field_name}:\n{value}")
        content = "\n\n".join(content_parts)
    return title, content


# ---------------------------------------------------------------------------
# Row producers
# ---------------------------------------------------------------------------
def _row(doc_id: str, source_type: str, title: str | None, content: str | None) -> Row:
    content = content or ""
    return (doc_id, source_type, title, content, len(content))


def sources_dir(erb_repo: str | Path) -> Path:
    return Path(erb_repo) / SOURCES_SUBDIR


def iter_repo_rows(erb_repo: str | Path, source_types: Iterable[str] | None = None) -> Iterator[Row]:
    """Yield corpus rows from a benchmark checkout (same walk as the upstream exporter).

    Files without ``dataset_doc_uuid``, unreadable files and files failing the
    extraction rule are skipped, exactly as the upstream exporter skips them.
    """
    root = sources_dir(erb_repo)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark sources not found at {root}")
    wanted = set(source_types) if source_types else None
    for source in sorted(os.listdir(root)):
        if wanted is not None and source not in wanted:
            continue
        source_path = root / source
        if not source_path.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(source_path):
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                full = Path(dirpath) / filename
                try:
                    with open(full, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:  # noqa: BLE001 - mirror upstream: unreadable -> skipped
                    continue
                if not isinstance(data, dict) or "dataset_doc_uuid" not in data:
                    continue
                try:
                    title, content = extract_document_content(data)
                except DocumentFieldError:
                    continue
                yield _row(str(data["dataset_doc_uuid"]), source, title, content)


def iter_hf_rows(dataset: Iterable[dict]) -> Iterator[Row]:
    """Yield corpus rows from Hugging Face dataset rows (``doc_id``/``source_type``/``title``/``content``)."""
    for r in dataset:
        yield _row(r.get("doc_id"), r.get("source_type"), r.get("title"), r.get("content"))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _open_for_build(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Fast bulk-load pragmas; safe because the file is a rebuildable derived artifact.
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_META)
    return conn


def corpus_identity(db_path: str | Path) -> dict:
    """The identity recorded by the builder: {"source": "repo"|"hf", "revision": ..., "built_at": ..., "rows": n}."""
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.OperationalError:
            return {"source": "unknown", "revision": None}
        ident = {k: v for k, v in rows}
        ident["rows"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        return ident
    finally:
        conn.close()


def _set_identity(conn: sqlite3.Connection, identity: dict) -> None:
    existing = {k: v for k, v in conn.execute("SELECT key, value FROM meta").fetchall()}
    for k in ("source", "revision"):
        if k in existing and existing[k] != str(identity.get(k)):
            raise ValueError(
                f"documents.sqlite was built from {existing.get('source')}@{existing.get('revision')}; "
                f"refusing to add rows from {identity.get('source')}@{identity.get('revision')}. "
                "Delete the file to rebuild it."
            )
    for k, v in identity.items():
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (k, str(v)))
    conn.commit()


def build_documents_db(db_path: str | Path, rows: Iterable[Row], *, limit: int | None = None,
                       quiet: bool = False, identity: dict | None = None) -> int:
    """Insert ``rows`` into ``documents.sqlite`` (idempotent within one identity).
    Returns the table's row count. ``identity`` names the source and revision;
    a file built from a different identity is refused."""
    db_path = Path(db_path)
    conn = _open_for_build(db_path)
    try:
        _set_identity(conn, {"source": "unknown", "revision": None, **(identity or {}),
                             "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        already = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if already and not quiet:
            print(f"[documents] {already:,} rows already present; duplicates are skipped", flush=True)

        start = time.monotonic()
        batch: list[Row] = []
        seen = 0
        for row in rows:
            batch.append(row)
            seen += 1
            if len(batch) >= COMMIT_EVERY:
                conn.executemany(_INSERT, batch)
                conn.commit()
                batch.clear()
            if not quiet and seen % PROGRESS_EVERY == 0:
                rate = seen / max(time.monotonic() - start, 1e-6)
                print(f"[documents] {seen:,} docs ({rate:,.0f}/s) ...", flush=True)
            if limit is not None and seen >= limit:
                break
        if batch:
            conn.executemany(_INSERT, batch)
            conn.commit()

        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);")
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    finally:
        conn.close()
    if not quiet:
        print(f"[documents] {total:,} rows in {db_path} ({seen:,} streamed this run, "
              f"{time.monotonic() - start:,.0f}s)", flush=True)
    return total


def build_from_repo(erb_repo: str | Path, db_path: str | Path, *, source_types: Iterable[str] | None = None,
                    limit: int | None = None, quiet: bool = False) -> int:
    """Build ``documents.sqlite`` from a benchmark checkout (identity = repo@<HEAD>)."""
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(erb_repo), capture_output=True, text=True).stdout.strip()
    return build_documents_db(db_path, iter_repo_rows(erb_repo, source_types), limit=limit, quiet=quiet,
                              identity={"source": "repo", "revision": head or "unknown",
                                        "source_types": ",".join(sorted(source_types)) if source_types else "all"})


def build_from_hf(db_path: str | Path, *, limit: int | None = None, quiet: bool = False,
                  dataset: Iterable[dict] | None = None, revision: str | None = None) -> int:
    """Build ``documents.sqlite`` by streaming the Hugging Face ``documents`` config.

    ``dataset`` may be supplied (any iterable of row dicts) to bypass the
    download; otherwise ``datasets`` is imported here and only here.
    """
    if dataset is None:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "the Hugging Face path needs the optional 'corpus' extra: pip install 'erb-hydradb[corpus]'"
            ) from exc
        if not quiet:
            print(f"[documents] streaming {paths.HF_DATASET} config '{paths.HF_DOCUMENTS_CONFIG}' "
                  f"split '{paths.HF_SPLIT}' (limit={limit or 'none'}) ...", flush=True)
        dataset = load_dataset(paths.HF_DATASET, paths.HF_DOCUMENTS_CONFIG, split=paths.HF_SPLIT,
                               streaming=True, revision=revision or paths.HF_REVISION)
    return build_documents_db(db_path, iter_hf_rows(dataset), limit=limit, quiet=quiet,
                              identity={"source": "hf", "revision": revision or paths.HF_REVISION or "main (unpinned)"})


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"corpus not found at {db_path}; build it first (erb-hydradb setup)")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _where(source_types: Iterable[str] | None, after_doc_id: str | None) -> tuple[str, list]:
    wheres: list[str] = []
    params: list = []
    if source_types:
        st = list(source_types)
        wheres.append(f"source_type IN ({','.join('?' for _ in st)})")
        params.extend(st)
    if after_doc_id:
        wheres.append("doc_id > ?")
        params.append(after_doc_id)
    return ("WHERE " + " AND ".join(wheres)) if wheres else "", params


def count_documents(db_path: str | Path, source_types: Iterable[str] | None = None,
                    after_doc_id: str | None = None) -> int:
    conn = connect_readonly(db_path)
    try:
        where, params = _where(source_types, after_doc_id)
        return conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]
    finally:
        conn.close()


def iter_documents(db_path: str | Path, source_types: Iterable[str] | None = None,
                   limit: int | None = None, after_doc_id: str | None = None) -> Iterator[dict]:
    """Stream document rows as dicts in ``doc_id`` order (the ingestion/checkpoint order)."""
    conn = connect_readonly(db_path)
    try:
        where, params = _where(source_types, after_doc_id)
        cur = conn.execute(
            f"SELECT doc_id, source_type, title, content, content_len FROM documents {where} "
            f"ORDER BY doc_id LIMIT ?",
            [*params, limit if limit is not None else -1],
        )
        for row in cur:
            yield dict(row)
    finally:
        conn.close()


def fetch_documents(db_path: str | Path, doc_ids: Iterable[str]) -> dict[str, dict]:
    """Resolve document ids to full rows (``{doc_id: row}``)."""
    ids = list(doc_ids)
    if not ids:
        return {}
    conn = connect_readonly(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT doc_id, source_type, title, content, content_len FROM documents "
            f"WHERE doc_id IN ({placeholders})", ids).fetchall()
        return {r["doc_id"]: dict(r) for r in rows}
    finally:
        conn.close()


def document_stats(db_path: str | Path) -> dict:
    """Total and per-source counts plus content-length statistics."""
    conn = connect_readonly(db_path)
    try:
        rows = conn.execute(
            "SELECT source_type, COUNT(*), CAST(AVG(content_len) AS INTEGER), "
            "MAX(content_len), MIN(content_len) FROM documents "
            "GROUP BY source_type ORDER BY COUNT(*) DESC, source_type").fetchall()
        total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()
    return {
        "total": total,
        "by_source_type": [
            {"source_type": s, "count": c, "avg_content_len": a, "max_content_len": mx, "min_content_len": mn}
            for (s, c, a, mx, mn) in rows
        ],
    }


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------
class QuestionsMismatchError(RuntimeError):
    """``questions.jsonl`` does not hash to the pinned value."""


def verify_questions(path: str | Path) -> str:
    """Return the file's sha256; raise ``QuestionsMismatchError`` if it is not the pinned one."""
    from .manifest import sha256_file

    digest = sha256_file(path)
    if digest != paths.QUESTIONS_SHA256:
        raise QuestionsMismatchError(
            f"{path}: sha256 {digest} != pinned {paths.QUESTIONS_SHA256} "
            f"(expected the file at benchmark commit {paths.ERB_COMMIT})")
    return digest


def copy_questions(erb_repo: str | Path, dest: str | Path) -> Path:
    """Copy ``questions.jsonl`` from the checkout to ``dest`` and verify its sha256."""
    src = Path(erb_repo) / QUESTIONS_FILENAME
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; is {erb_repo} a benchmark checkout?")
    verify_questions(src)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    verify_questions(dest)
    return dest


def load_questions(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
