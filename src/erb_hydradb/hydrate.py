"""Canonical document text for the answer model.

The generator gives the answer model the FULL canonical text of each retrieved
document, where "canonical" means exactly what the benchmark exports: the
document's title field plus its declared content fields, joined by the
benchmark's own rule (``src/utils/document_content.py`` in the ERB repo).
Nothing else from the raw corpus files is ever rendered. Benchmark-internal
annotations (``dataset_doc_uuid``, ``dataset_noise_document``, the field-name
labels) are forbidden in a context; ``assert_no_leak`` fails the run if one
appears.

Two document sources are supported and produce identical text:

* ``documents.sqlite`` built by ``erb-hydradb download`` (from the Hugging Face
  export or from a checkout), preferred;
* a checkout of the ERB repository (``generated_data/uuid_index.json`` +
  ``generated_data/sources/``), used as a fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_MARKERS = (
    "title_field_name",
    "content_field_names",
    "dataset_doc_uuid",
    "dataset_noise_document",
)

# Typed-item suffix the converter appends to a document id (_m0001 Slack
# message, _cNN comment, _eNN email); stripping it maps a chunk back to its
# ERB document id.
_ITEM_SUFFIX = re.compile(r"_[ecm]\d+$")


class ContextLeakError(RuntimeError):
    pass


def erb_doc_id(chunk_or_id: dict | str) -> str:
    cid = chunk_or_id.get("id", "") if isinstance(chunk_or_id, dict) else chunk_or_id
    return _ITEM_SUFFIX.sub("", str(cid or ""))


def extract_document_content(doc: dict) -> tuple[str, str]:
    """The benchmark's canonical (title, content) rule. One implementation,
    in ``corpus.py`` (mirrors the ERB repository's document_content.py)."""
    from .corpus import extract_document_content as _extract
    return _extract(doc)


@dataclass
class Document:
    doc_id: str
    source_type: str
    title: str
    content: str


class DocumentStore:
    """Look up canonical documents by ERB id from sqlite, else from a checkout."""

    def __init__(self, db_path: str | Path | None = None, erb_repo: str | Path | None = None):
        self._conn: sqlite3.Connection | None = None
        self._index: dict[str, str] | None = None
        self._sources: Path | None = None
        if db_path and Path(db_path).exists():
            self._conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        if erb_repo:
            base = Path(erb_repo) / "generated_data"
            idx = base / "uuid_index.json"
            if idx.exists():
                with open(idx, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
                self._sources = base / "sources"
        if self._conn is None and self._index is None:
            raise FileNotFoundError(
                "no document source: build data/documents.sqlite with `erb-hydradb download` "
                "or set ERB_REPO to an EnterpriseRAG-Bench checkout"
            )

    @property
    def identity(self) -> dict:
        """Where the documents come from, for manifests."""
        out: dict = {"backend": self.backend}
        if self._conn is not None:
            try:
                out["sqlite"] = {k: v for k, v in self._conn.execute("SELECT key, value FROM meta").fetchall()}
            except sqlite3.OperationalError:
                out["sqlite"] = {"source": "unknown"}
        if self._sources is not None:
            out["checkout"] = str(self._sources.parent.parent)
        return out

    @property
    def backend(self) -> str:
        if self._conn is not None and self._index is not None:
            return "sqlite+checkout"
        return "sqlite" if self._conn is not None else "checkout"

    def get(self, doc_id: str) -> Document | None:
        if self._conn is not None:
            row = self._conn.execute(
                "SELECT doc_id, source_type, title, content FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is not None:
                return Document(row["doc_id"], row["source_type"], row["title"], row["content"])
            if self._index is None:
                return None
        rel = (self._index or {}).get(doc_id)
        if not rel or self._sources is None:
            return None
        p = self._sources / rel
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        try:
            title, content = extract_document_content(raw)
        except (KeyError, TypeError, ValueError):
            return None
        return Document(doc_id, rel.split("/")[0], title, content)


def render_block(i: int, doc: Document) -> str:
    return f"[document {i} | source={doc.source_type} | title={doc.title} | id={doc.doc_id}]\n{doc.content}"


def assert_no_leak(context: str) -> None:
    for marker in FORBIDDEN_MARKERS:
        if marker in context:
            raise ContextLeakError(f"benchmark-internal marker {marker!r} leaked into context")


def build_context(doc_ids: list[str], chunks: list[dict], store: DocumentStore,
                  n_docs: int, max_chars: int) -> dict:
    """Full canonical text of the top ``n_docs`` documents, in rank order,
    under a total character budget. Whole documents are dropped from the tail
    of the ranking, never truncated. A document missing from the store falls
    back to its retrieved chunk text and is recorded as such."""
    by_doc_chunks: dict[str, list[str]] = {}
    for ch in chunks:
        by_doc_chunks.setdefault(erb_doc_id(ch), []).append(ch.get("chunk_content", ""))

    parts, used, dropped, fallback = [], [], [], []
    total = 0
    for i, did in enumerate(doc_ids[:n_docs], 1):
        doc = store.get(did)
        if doc is None:
            text = "\n...\n".join(by_doc_chunks.get(did, []))
            block = render_block(i, Document(did, "unknown (retrieved chunks only)", "", text))
            fallback.append(did)
        else:
            block = render_block(i, doc)
        if not block.strip():
            continue
        if total + len(block) > max_chars and parts:
            dropped.append(did)
            continue
        parts.append(block)
        used.append(did)
        total += len(block) + 9

    context = "\n\n=====\n\n".join(parts) or "(no context retrieved)"
    assert_no_leak(context)
    return {
        "context": context,
        "context_docs": used,
        "dropped_docs": dropped,
        "chunk_fallback_docs": fallback,
        "context_chars": len(context),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }


def distinct_docs(chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for ch in chunks:
        did = erb_doc_id(ch)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def env_erb_repo() -> str | None:
    return os.getenv("ERB_REPO")
