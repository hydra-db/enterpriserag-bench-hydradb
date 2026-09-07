"""Streaming, checkpointed, resumable ingestion of the corpus into one HydraDB collection.

Pipeline:

1. stream rows from ``documents.sqlite`` in ``doc_id`` order (``corpus.iter_documents``);
2. convert each row into typed ``app_knowledge`` items (``convert.convert``);
3. optionally provision the database (``POST /databases``, poll until
   ``ready_for_ingestion``);
4. accumulate items into batches and ``POST /context/ingest`` each batch,
   journaling every batch durably (see below) so a restart can pick up exactly
   where the previous process died;
5. unless ``wait=False``, poll ``/context/status`` for every batch's ids until
   they settle, keeping a trailing window of ``wait_window`` batches in flight
   so the server indexes while the client keeps sending. Each id's terminal
   status is appended to ``ingest_status.jsonl`` next to the state file;
6. with ``retry_failed=True``, re-send every item whose last attempt errored
   (see *Failed items* below) as new journal batches and poll those too.

Batch journal
-------------
The state file carries a ``journal``: one entry per batch, written atomically
(tmp file + ``os.replace``) together with the document cursor, so the cursor
can never get ahead of the record of what was sent. Each entry moves through
three states:

* ``sending`` — written *before* the HTTP request, with the batch's doc range
  ``{"after": <cursor before the batch>, "last": <last doc in the batch>}``
  and its item ids;
* ``sent``    — the server accepted the batch; ``statuses`` fills in with
  ``{id: terminal_status}`` as ids resolve;
* ``settled`` — every id has a terminal status. The entry is then compacted to
  ``counts`` (``settled`` / ``errored``); the per-id record lives in
  ``ingest_status.jsonl`` and, for errored ids, in ``failed_ids`` (below).
  This keeps the state file bounded by the number of *unresolved* batches
  (at most ``wait_window`` + 1 when ``wait=True``) plus the number of failed
  items.

Retry batches (``"retry": true``) carry ``doc_ids`` instead of a doc range;
they never move the document cursor and their items are not counted as sent
again (every retried id was already counted by its original batch).

On ``resume=True`` the runner first reconciles every entry that is not
``settled`` before converting anything new: batches stuck in ``sending`` are
re-sent, then every unresolved id is polled and recorded. Re-sending is safe
because ingestion is an upsert (``upsert=true``) keyed by deterministic item
ids (``convert`` derives every id from the document id), so a batch whose
request may or may not have reached the server can simply be sent again.
A ``sending`` batch is rebuilt by re-converting its doc range (or, for a retry
batch, its ``doc_ids``); the rebuilt ids must equal the journaled ids or the
resume is refused.

Readiness
---------
``ingest_manifest.json`` reports ``readiness`` (also mirrored as top-level
``ready`` / ``resolved`` flags) with ``items_sent``, ``items_settled``,
``items_errored``, ``items_pending``, ``unresolved_batches`` and two verdicts:

* ``resolved`` — every request has an answer: ``items_sent == items_settled +
  items_errored``, ``items_pending == 0`` and ``unresolved_batches == 0``.
  "All requests resolved" is *not* "corpus ready": an entirely failed batch is
  resolved.
* ``ready``    — the collection holds every item: ``items_sent ==
  items_settled``, ``items_errored == 0``, ``items_pending == 0`` and
  ``unresolved_batches == 0``. Only a ``ready`` run may be queried.

A run that ends not ready still writes its manifest and then raises
``IngestIncomplete`` so the CLI exits nonzero. The message says which
recovery applies: unresolved work (pending ids / unresolved batches) is
finished by ``--resume``; failed items are re-sent by ``--retry-failed``.
``accept_failures=True`` lets a *resolved* run with failures return normally
with ``ready: false`` and the failed ids listed under ``accepted_failures``
(the CLI's explicit override); it never covers unresolved work.

Failed items
------------
``failed_ids`` (``{id: {"batch", "status", "attempts"}}``) records every item
whose latest attempt ended in an errored status. It lives in the state file
and in the manifest and is never compacted away; an id leaves it only when a
later attempt settles. ``items_errored`` is derived from it, so a retried id
is counted exactly once whatever happened to it. ``retry_failed=True`` takes a
snapshot of ``failed_ids`` at the end of the run (after everything else has
been sent and polled), maps each item id back to its document (item ids are
``<doc_id>`` or ``<doc_id>_<e|c|m><n>``), re-converts those documents, sends
*only the failed items* as new retry batches and polls them; ids that settle
move out of ``failed_ids``, ids that fail again stay with ``attempts`` bumped.

Resume identity
---------------
A checkpoint is only resumed for the same target (``endpoint`` — the
normalised ``hydradb.base_url``, see ``normalize_endpoint`` — ``database``,
``collection``, ``infer``), the same source scope (``source_types``,
``limit``), the same corpus — ``source`` and ``revision`` from the corpus
``meta`` table (``corpus.corpus_identity``), row count, a sha256 over every
``doc_id`` in order and ``content_sha256``, a streaming sha256 over every
row's (doc_id, title, content) (``identity.content_sha256_sqlite``; about a
minute on the full corpus, computed at every run start and never read back
from metadata) — and the same converter (sha256 of ``convert.py``'s source,
``converter_sha256()``). A mismatch is refused with a message naming the
field, before the HTTP client is constructed. The doc-id hash guarantees the
journal's item ids are the ids this corpus produces; the content hash
guarantees the content behind those ids is the content that was sent (a label
such as ``revision`` cannot: a corpus rebuilt with the same ids and revision
but a changed body is a different corpus). Checkpoints written before the
endpoint / content identity existed are refused outright.

Status log
----------
``ingest_status.jsonl`` is the per-id attempt history: one
``{"batch", "id", "status", "attempt"}`` line per terminal status observed,
appended *before* the checkpoint that records it. It is read idempotently:
``read_status_log`` keeps the last row per ``(batch, id)``, so a crash between
the append and the checkpoint (which makes the resumed process observe the
same status again) cannot inflate it. The resume also adopts terminal
statuses already logged for its unresolved batches instead of re-polling
them, so that window normally leaves no duplicate at all. An id appears once
per attempt (its original batch, then one row per retry batch). A fresh
(non-resume) run starts a new log.

The log is read through ``identity.read_journal``: an incomplete trailing
record (the append itself was interrupted, before its checkpoint) is
quarantined to ``ingest_status.jsonl.torn-<timestamp>`` and the log is
truncated to its last complete record; the ids of that record are still
unresolved in the journal and are simply re-polled. A resume repairs the log
once, right after the identity check and before anything is appended to it.
Corruption anywhere else in the log raises ``ValueError`` and is never
skipped.

Memory stays bounded regardless of corpus size: one document is converted at a
time and at most ``batch_size`` items (plus the id lists of unresolved
batches and the failed-id record) are resident.

Files written next to ``state_path``:

* ``<state_path>``          checkpoint (cursor, counters, identity guard, journal, failed ids);
* ``ingest_status.jsonl``   one ``{"batch", "id", "status", "attempt"}`` line per resolved attempt;
* ``ingest_status.jsonl.torn-<ts>``  a torn trailing record quarantined by a resume (only if one was found);
* ``ingest_manifest.json``  the run summary (counts, readiness, failed ids, wall time, config);
* ``manifest.json`` / ``SHA256SUMS`` via ``manifest.write_manifest``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from collections import Counter, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from . import convert as _convert_module
from . import corpus, manifest
from . import identity as _identity
from .config import RunConfig
from .convert import convert
from .hydradb_client import HydraDBAdminClient

STATUS_LOG_NAME = "ingest_status.jsonl"
INGEST_MANIFEST_NAME = "ingest_manifest.json"
ERRORED_STATES = {"errored", "failed"}
READINESS_KEYS = ("items_sent", "items_settled", "items_errored", "items_pending", "unresolved_batches")
READINESS_FLAGS = ("resolved", "ready")
# item id -> doc id: convert() derives every item id from the doc id plus an optional typed suffix
_ITEM_SUFFIX = re.compile(r"_[ecm]\d+$")


@lru_cache(maxsize=1)
def converter_sha256() -> str:
    """sha256 of ``convert.py``'s source text: part of the resume identity (a changed converter changes items)."""
    return hashlib.sha256(inspect.getsource(_convert_module).encode("utf-8")).hexdigest()


class IngestIncomplete(RuntimeError):
    """The run finished but the collection is not ready to query.

    Raised by ``run_ingest`` *after* ``ingest_manifest.json`` has been written
    with ``ready: false``. ``readiness`` holds the counts and verdicts (see
    ``READINESS_KEYS`` and ``READINESS_FLAGS``), ``manifest`` the record that
    was written, ``failed_ids`` the failed-item record; each count and flag is
    also an attribute (``exc.items_pending``, ``exc.resolved`` ...).
    ``reason`` is ``"unresolved"`` (pending ids / unresolved batches: re-run
    with ``--resume``) or ``"failed"`` (every request resolved but some items
    errored: re-run with ``--retry-failed``).
    """

    def __init__(self, readiness: dict, manifest: dict) -> None:
        self.readiness = dict(readiness)
        self.manifest = manifest
        self.failed_ids = dict(manifest.get("failed_ids") or {})
        for key in READINESS_KEYS + READINESS_FLAGS:
            setattr(self, key, readiness[key])
        r = readiness
        counts = (f"{r['items_sent']:,} sent, {r['items_settled']:,} settled, {r['items_errored']:,} errored, "
                  f"{r['items_pending']:,} pending, {r['unresolved_batches']} unresolved batch(es)")
        if not r["resolved"]:
            self.reason = "unresolved"
            hint = (f"{r['items_pending']:,} item(s) pending / {r['unresolved_batches']} unresolved batch(es); "
                    "re-run with --resume to reconcile")
            if r["items_errored"]:
                hint += f", then --retry-failed for the {r['items_errored']:,} failed item(s)"
        else:
            self.reason = "failed"
            hint = (f"{r['items_errored']:,} item(s) failed; re-run with --resume --retry-failed "
                    "(or --accept-failures to record them and continue)")
        super().__init__(f"ingestion is not ready to query ({counts}): {hint}")


# ---------------------------------------------------------------------------
# checkpoint state
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _scope(source_types: list[str] | None) -> list[str] | None:
    return sorted(set(source_types)) if source_types else None


def normalize_endpoint(base_url: str) -> str:
    """The endpoint a checkpoint is bound to: ``scheme://host[:port]/path`` of ``base_url``,
    lowercased, without query, fragment or trailing slash. ``https://API.hydradb.com/``
    and ``https://api.hydradb.com`` are the same endpoint; a different host or path is not."""
    raw = base_url.strip()
    parts = urlsplit(raw if "://" in raw else f"//{raw}")
    scheme = parts.scheme or "https"
    return f"{scheme.lower()}://{parts.netloc.lower()}{parts.path.lower().rstrip('/')}"


def corpus_identity(db_path: str | Path) -> dict:
    """Identity of ``documents.sqlite`` for the resume guard.

    ``source`` / ``revision`` come from the builder's ``meta`` table
    (``corpus.corpus_identity``; ``"unknown"`` / ``None`` for a file built
    without one), ``documents`` is the row count, ``doc_ids_sha256`` a sha256
    over every ``doc_id`` in order and ``content_sha256`` a streaming sha256
    over every row's (doc_id, title, content) in that order
    (``identity.content_sha256_sqlite``). The doc-id hash guarantees the
    journal's item ids are the ids this corpus produces; the content hash
    guarantees the *content* behind those ids is the content that was sent.
    Streams the whole table (about a minute on the full corpus).
    """
    meta = corpus.corpus_identity(db_path)
    conn = corpus.connect_readonly(db_path)
    try:
        h = hashlib.sha256()
        total = 0
        for (doc_id,) in conn.execute("SELECT doc_id FROM documents ORDER BY doc_id"):
            h.update(doc_id.encode("utf-8"))
            h.update(b"\n")
            total += 1
    finally:
        conn.close()
    content = _identity.content_sha256_sqlite(db_path)
    if content["rows"] != total:
        raise RuntimeError(f"corpus {db_path} changed while its identity was being computed "
                           f"({total} vs {content['rows']} rows)")
    return {"source": meta.get("source", "unknown"), "revision": meta.get("revision"),
            "documents": total, "doc_ids_sha256": h.hexdigest(), "content_sha256": content["content_sha256"]}


def _new_state(cfg: RunConfig, infer: bool, source_types: list[str] | None, limit: int | None,
               db_path: Path, identity: dict) -> dict:
    return {
        "endpoint": normalize_endpoint(cfg.hydradb.base_url),
        "database": cfg.hydradb.database,
        "collection": cfg.hydradb.collection,
        "infer": infer,
        "source_types": _scope(source_types),
        "limit": limit,
        "corpus": {"db_path": str(db_path), **identity},
        "converter": {"sha256": converter_sha256()},
        "last_doc_id": None,
        "docs_processed": 0,
        "items_sent": 0,
        "batches_sent": 0,
        "items_by_kind": {},
        "statuses": {"settled": 0, "errored": 0, "pending": 0},
        "failed_ids": {},
        "retry": {"batches": 0, "items": 0, "recovered": 0},
        "journal": [],
        "updated_at": None,
    }


def load_state(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_state_matches(state: dict, cfg: RunConfig, infer: bool, source_types: list[str] | None,
                         limit: int | None, identity: dict) -> None:
    if "journal" not in state:
        raise RuntimeError("refusing to resume: checkpoint predates the batch journal format; "
                           "start a new run directory")
    if "failed_ids" not in state or "converter" not in state:
        raise RuntimeError("refusing to resume: checkpoint predates the failed-item record / converter identity; "
                           "start a new run directory")
    have_corpus = state.get("corpus") or {}
    if "endpoint" not in state or "content_sha256" not in have_corpus:
        raise RuntimeError("refusing to resume: checkpoint predates the endpoint / corpus-content identity; "
                           "start a new run directory")
    checks = [
        ("endpoint", normalize_endpoint(cfg.hydradb.base_url), state.get("endpoint")),
        ("database", cfg.hydradb.database, state.get("database")),
        ("collection", cfg.hydradb.collection, state.get("collection")),
        ("infer", infer, bool(state.get("infer"))),
        ("source_types", _scope(source_types), state.get("source_types")),
        ("limit", limit, state.get("limit")),
        ("corpus.source", identity["source"], have_corpus.get("source")),
        ("corpus.revision", identity["revision"], have_corpus.get("revision")),
        ("corpus.documents", identity["documents"], have_corpus.get("documents")),
        ("corpus.doc_ids_sha256", identity["doc_ids_sha256"], have_corpus.get("doc_ids_sha256")),
        ("corpus.content_sha256", identity["content_sha256"], have_corpus.get("content_sha256")),
        ("converter.sha256", converter_sha256(), (state.get("converter") or {}).get("sha256")),
    ]
    for field, want, have in checks:
        if want != have:
            raise RuntimeError(
                f"refusing to resume: checkpoint was written for {field}={have!r}, this run has {field}={want!r}")


# ---------------------------------------------------------------------------
# status log
# ---------------------------------------------------------------------------
def read_status_log(path: str | Path) -> list[dict]:
    """The attempt history: one row per ``(batch, id)``, the last one written winning.

    A crash between the log append and the checkpoint makes the resumed process
    observe (and may make it log) the same terminal status again; reading
    through this function makes that harmless. An id that was retried has one
    row per batch it was sent in.

    Reads through ``identity.read_journal``: an incomplete trailing record
    (an append interrupted mid-line) is quarantined to a ``.torn-<timestamp>``
    file and truncated away; a malformed record anywhere else raises
    ``ValueError`` (never skipped).
    """
    rows: dict[tuple[int, str], dict] = {}
    for row in _identity.read_journal(path):
        key = (row["batch"], row["id"])
        rows.pop(key, None)  # keep first-appearance order but the last content
        rows[key] = row
    return list(rows.values())


def _logged_terminal_statuses(path: Path, batches: set[int]) -> dict[int, dict[str, str]]:
    """``{batch: {id: status}}`` for the given batches, from the log (terminal statuses only)."""
    out: dict[int, dict[str, str]] = {}
    if not batches:
        return out
    for row in read_status_log(path):
        if row["batch"] in batches and row["status"] != "pending":
            out.setdefault(row["batch"], {})[row["id"]] = row["status"]
    return out


# ---------------------------------------------------------------------------
# journal accounting
# ---------------------------------------------------------------------------
def _readiness(sent: int, settled: int, errored: int, pending: int, unresolved: int) -> dict:
    resolved = sent == settled + errored and pending == 0 and unresolved == 0
    return {
        "items_sent": sent, "items_settled": settled, "items_errored": errored, "items_pending": pending,
        "unresolved_batches": unresolved,
        "resolved": resolved,
        "ready": resolved and errored == 0 and sent == settled,
    }


def _recount(state: dict) -> dict:
    """Recompute counters from the journal and ``failed_ids`` (never incremental, so re-polls cannot double count).

    ``items_sent`` counts original batches only (a retry re-sends ids already
    counted). ``items_errored`` is the number of failed ids that are not
    currently in flight in a retry batch; in-flight retried ids count as
    pending. ``items_settled`` is the remainder.
    """
    sent = batches = unresolved = pending = 0
    retry_batches = retry_items = retry_inflight = recovered = 0
    for e in state["journal"]:
        is_retry = bool(e.get("retry"))
        if e["state"] == "sending":  # the request may never have reached the server; not counted until re-sent
            unresolved += 1
            continue
        batches += 1
        if is_retry:
            retry_batches += 1
            retry_items += e["n_items"]
        else:
            sent += e["n_items"]
        if e["state"] == "settled":
            if is_retry:
                recovered += e["counts"]["settled"]
        else:  # "sent"
            unresolved += 1
            open_ids = len(e["ids"]) - len(e["statuses"])
            pending += open_ids
            if is_retry:
                retry_inflight += open_ids
                recovered += sum(1 for st in e["statuses"].values() if st not in ERRORED_STATES)
    errored = len(state["failed_ids"]) - retry_inflight
    settled = sent - pending - errored
    state["items_sent"] = sent
    state["batches_sent"] = batches
    state["statuses"] = {"settled": settled, "errored": errored, "pending": pending}
    state["retry"] = {"batches": retry_batches, "items": retry_items, "recovered": recovered}
    return _readiness(sent, settled, errored, pending, unresolved)


def _doc_id_of(item_id: str) -> str:
    return _ITEM_SUFFIX.sub("", item_id)


def _convert_selected(db_path: Path, doc_ids: list[str], wanted: list[str], database: str,
                      collection: str, what: str) -> list[dict]:
    """Re-convert ``doc_ids`` and return the items whose ids are ``wanted``, in that order."""
    rows = corpus.fetch_documents(db_path, doc_ids)
    produced: dict[str, dict] = {}
    for d in doc_ids:
        row = rows.get(d)
        if row is None:
            raise RuntimeError(f"refusing to {what}: document {d!r} is not in the corpus")
        produced.update((it["id"], it) for it in convert(row, database, collection))
    missing = [i for i in wanted if i not in produced]
    if missing:
        raise RuntimeError(
            f"refusing to {what}: re-converting {len(doc_ids)} document(s) did not produce "
            f"{len(missing)} expected item id(s) (first: {missing[0]!r}); the corpus or converter changed")
    return [produced[i] for i in wanted]


def _rebuild_batch(db_path: Path, source_types: list[str] | None, entry: dict, database: str,
                   collection: str) -> list[dict]:
    """Re-convert a ``sending`` journal entry (its doc range, or its ``doc_ids`` for a retry) into its items."""
    if entry.get("retry"):
        return _convert_selected(db_path, entry["doc_ids"], entry["ids"], database, collection,
                                 f"resume: rebuilding retry batch {entry['batch']}")
    items: list[dict] = []
    for doc in corpus.iter_documents(db_path, source_types, after_doc_id=entry["docs"]["after"]):
        if doc["doc_id"] > entry["docs"]["last"]:
            break
        items.extend(convert(doc, database, collection))
    ids = [it["id"] for it in items]
    if ids != entry["ids"]:
        raise RuntimeError(
            f"refusing to resume: re-converting batch {entry['batch']} (docs after {entry['docs']['after']!r} "
            f"through {entry['docs']['last']!r}) produced {len(ids)} ids that do not match the "
            f"{len(entry['ids'])} journaled ids; the corpus or converter changed")
    return items


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def run_ingest(cfg: RunConfig, db_path: str | Path, state_path: str | Path, *,
               infer: bool = False, resume: bool = False, dry_run: bool = False,
               limit: int | None = None, source_types: list[str] | None = None,
               batch_size: int = 80, batch_sleep: float = 1.0, provision: bool = True,
               wait: bool = True, api_key: str | None = None,
               wait_window: int = 4, wait_timeout_s: float = 1800.0,
               retry_failed: bool = False, accept_failures: bool = False) -> dict:
    """Ingest the corpus into ``cfg.hydradb.database`` / ``cfg.hydradb.collection``.

    ``dry_run`` converts and counts without constructing an HTTP client or
    touching the checkpoint. Returns the ingestion manifest dict (also written
    to ``ingest_manifest.json`` next to ``state_path``).

    ``retry_failed`` re-sends, at the end of the run, every item whose last
    attempt errored (from the resumed checkpoint and from this run) and polls
    the retry batches. ``accept_failures`` lets a run whose only defect is
    failed items return normally (``ready: false``, ``accepted_failures``
    listing them).

    Raises ``IngestIncomplete`` (after writing the manifest with
    ``ready: false``) when any sent item is still pending, any batch is
    unresolved, or (unless ``accept_failures``) any item failed.
    """
    db_path = Path(db_path)
    state_path = Path(state_path)
    run_dir = state_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    status_log = run_dir / STATUS_LOG_NAME
    database, collection = cfg.hydradb.database, cfg.hydradb.collection

    started_at = manifest.now_iso()
    started = time.monotonic()

    # -- resume point ------------------------------------------------------
    state: dict | None = None
    identity: dict = {}
    if not dry_run:
        print(f"corpus identity: hashing every row of {db_path} (content sha256; about a minute on the "
              "full corpus) ...", flush=True)
        identity = corpus_identity(db_path)
    if resume and not dry_run:
        state = load_state(state_path)
        if state:
            _check_state_matches(state, cfg, infer, source_types, limit, identity)
            # repair a torn trailing record left by an interrupted append before anything is appended again
            logged_rows = len(read_status_log(status_log))
            unresolved = [e for e in state["journal"] if e["state"] != "settled"]
            print(f"resuming after doc_id {state['last_doc_id']} "
                  f"({state['docs_processed']:,} docs / {state['items_sent']:,} items already sent, "
                  f"{len(unresolved)} unresolved batch(es) to reconcile, "
                  f"{len(state['failed_ids']):,} failed item(s) on record, "
                  f"{logged_rows:,} status row(s) in the log)", flush=True)
        else:
            print("no checkpoint found; starting from the beginning", flush=True)
    elif not dry_run:
        if state_path.exists():
            print(f"note: ignoring existing checkpoint {state_path} (pass resume=True to continue it)",
                  flush=True)
        if status_log.exists():
            print(f"note: starting a new status log; the previous {status_log} is discarded", flush=True)
            status_log.unlink()
    if state is None:
        state = _new_state(cfg, infer, source_types, limit, db_path, identity)
    start_after = state["last_doc_id"]

    remaining = corpus.count_documents(db_path, source_types, after_doc_id=start_after)
    print(f"corpus: {remaining:,} docs remaining" + (f" (limit {limit})" if limit else ""), flush=True)

    docs_read = 0
    items_produced = 0
    by_kind: Counter[str] = Counter(state.get("items_by_kind") or {})
    docs = corpus.iter_documents(db_path, source_types, limit=limit, after_doc_id=start_after)

    # -- dry run: convert only ---------------------------------------------
    if dry_run:
        would_send = 0
        pending_items = 0
        for doc in docs:
            items = convert(doc, database, collection)
            docs_read += 1
            items_produced += len(items)
            by_kind.update(it["kind"] for it in items)
            pending_items += len(items)
            if pending_items >= batch_size:  # same flush rule as the real run
                would_send += 1
                pending_items = 0
        if pending_items:
            would_send += 1
        print(f"[dry-run] {docs_read:,} docs -> {items_produced:,} items in {would_send} batches; "
              f"no API calls made", flush=True)
        readiness = {k: 0 for k in READINESS_KEYS}
        readiness.update(resolved=False, ready=False)  # nothing was sent: the collection is not ready because of this run
        return _finish(run_dir, cfg, state_path, status_log, dict(
            dry_run=True, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
            batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
            retry_failed=retry_failed, accept_failures=accept_failures,
            started_at=started_at, wall_time_s=round(time.monotonic() - started, 3),
            corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read),
            items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=0),
            batches=dict(sent=0, retried=0, would_send=would_send),
            statuses=dict(settled=0, errored=0, pending=0),
            readiness=readiness, ready=False, resolved=False,
            failed_ids={}, retry=dict(batches=0, items=0, recovered=0), accepted_failures=[],
        ))

    # -- real run ----------------------------------------------------------
    load_dotenv()
    api_key = api_key or os.getenv("HYDRADB_API_KEY")
    if not api_key:
        raise RuntimeError("HYDRADB_API_KEY is not set (put it in .env or pass api_key=)")

    journal: list[dict] = state["journal"]
    failed_ids: dict[str, dict] = state["failed_ids"]
    inflight: deque[dict] = deque()  # entries in state "sent" not yet polled in this run

    def checkpoint() -> None:
        state["items_by_kind"] = dict(sorted(by_kind.items()))
        state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _recount(state)
        _atomic_write_json(state_path, state)

    def record(entry: dict, item_id: str, status: str) -> int:
        """Record a terminal status in the entry and ``failed_ids``; return the attempt number."""
        previous = failed_ids.get(item_id)
        attempt = (previous["attempts"] + 1) if (previous and entry.get("retry")) else 1
        entry["statuses"][item_id] = status
        if status in ERRORED_STATES:
            failed_ids[item_id] = {"batch": entry["batch"], "status": status, "attempts": attempt}
        else:
            failed_ids.pop(item_id, None)
        return attempt

    def compact_if_complete(entry: dict) -> None:
        if len(entry["statuses"]) == len(entry["ids"]):
            errored = sum(1 for st in entry["statuses"].values() if st in ERRORED_STATES)
            entry["counts"] = {"settled": len(entry["ids"]) - errored, "errored": errored}
            entry["state"] = "settled"
            del entry["ids"], entry["statuses"]
        else:
            print(f"  batch {entry['batch']}: {len(entry['ids']) - len(entry['statuses'])} id(s) still pending",
                  flush=True)

    def settle(client: HydraDBAdminClient, entry: dict) -> None:
        """Poll the entry's unresolved ids once; log and record terminal statuses; compact when complete."""
        todo = [i for i in entry["ids"] if i not in entry["statuses"]]
        final = client.wait_processing(database, collection, todo, timeout_s=wait_timeout_s) if todo else {}
        # log first, checkpoint second: a crash in between is absorbed by read_status_log's
        # (batch, id) dedupe and by the resume adopting logged statuses before re-polling
        with open(status_log, "a", encoding="utf-8") as f:
            for i in todo:
                st = final.get(i, "pending")
                if st == "pending":
                    continue  # stays unresolved; logged once it reaches a terminal state
                attempt = record(entry, i, st)
                f.write(json.dumps({"batch": entry["batch"], "id": i, "status": st, "attempt": attempt}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        compact_if_complete(entry)
        checkpoint()

    def adopt_logged_statuses() -> None:
        """Apply terminal statuses the log already holds for unresolved batches (crash before their checkpoint)."""
        open_entries = [e for e in journal if e["state"] == "sent"]
        logged = _logged_terminal_statuses(status_log, {e["batch"] for e in open_entries})
        for entry in open_entries:
            rows = logged.get(entry["batch"]) or {}
            adopted = 0
            for i in entry["ids"]:
                if i not in entry["statuses"] and i in rows:
                    record(entry, i, rows[i])
                    adopted += 1
            if adopted:
                print(f"  batch {entry['batch']}: adopted {adopted} status(es) already in the log", flush=True)
                compact_if_complete(entry)
                checkpoint()

    def submit(entry: dict, items: list[dict]) -> None:
        """Journal ``entry`` as sending, POST ``items``, mark it sent; keep the wait window."""
        journal.append(entry)
        checkpoint()  # cursor and journal entry land in the same atomic write
        client.ingest_app_knowledge(database, collection, items, infer=infer)
        entry["state"] = "sent"
        inflight.append(entry)
        checkpoint()
        elapsed = time.monotonic() - started
        print(f"  docs={state['docs_processed']:,} items={state['items_sent']:,} "
              f"batches={state['batches_sent']:,} failed={len(failed_ids):,} ({elapsed:.0f}s)", flush=True)
        if wait:
            while len(inflight) > wait_window:
                settle(client, inflight.popleft())
        if batch_sleep > 0:
            time.sleep(batch_sleep)

    def reconcile(client: HydraDBAdminClient) -> None:
        """Finish every unresolved journal entry left by a previous process before ingesting anything new."""
        for entry in journal:
            if entry["state"] == "sending":
                print(f"  re-sending batch {entry['batch']} ({entry['n_items']} items) whose send was "
                      f"interrupted", flush=True)
                items = _rebuild_batch(db_path, source_types, entry, database, collection)
                client.ingest_app_knowledge(database, collection, items, infer=infer)
                entry["state"] = "sent"
                checkpoint()
        adopt_logged_statuses()
        for entry in journal:
            if entry["state"] == "sent":
                if wait:
                    print(f"  polling batch {entry['batch']} ({len(entry['ids']) - len(entry['statuses'])} "
                          f"unresolved id(s))", flush=True)
                    settle(client, entry)
                else:
                    print(f"  batch {entry['batch']} left unresolved (wait=False)", flush=True)

    def retry_failed_items(client: HydraDBAdminClient) -> None:
        """Re-send every failed id that is not already in flight, as retry batches of only those items."""
        in_flight = {i for e in journal if e["state"] != "settled" for i in e["ids"]}
        by_doc: dict[str, list[str]] = {}
        for i in sorted(failed_ids):
            if i not in in_flight:
                by_doc.setdefault(_doc_id_of(i), []).append(i)
        if not by_doc:
            print("retry: no failed items to retry", flush=True)
            return
        n_items = sum(len(v) for v in by_doc.values())
        print(f"retrying {n_items:,} failed item(s) from {len(by_doc):,} document(s) ...", flush=True)
        doc_ids = sorted(by_doc)
        batch_docs: list[str] = []
        batch_ids: list[str] = []

        def flush() -> None:
            items = _convert_selected(db_path, batch_docs, batch_ids, database, collection, "retry")
            submit({
                "batch": len(journal) + 1,
                "retry": True,
                "doc_ids": list(batch_docs),
                "n_items": len(items),
                "ids": [it["id"] for it in items],
                "state": "sending",
                "statuses": {},
            }, items)
            batch_docs.clear()
            batch_ids.clear()

        for d in doc_ids:
            batch_docs.append(d)
            batch_ids.extend(by_doc[d])
            if len(batch_ids) >= batch_size:
                flush()
        if batch_ids:
            flush()
        if wait:
            while inflight:
                settle(client, inflight.popleft())

    with HydraDBAdminClient(api_key=api_key, base_url=cfg.hydradb.base_url) as client:
        if provision:
            print(f"provisioning database {database!r} ...", flush=True)
            resp = client.create_database(database)
            if resp.get("already_exists"):
                print("  database already exists; checking readiness", flush=True)
            client.wait_ready(database)
            print("  database ready for ingestion", flush=True)
            existing = client.list_collections(database)
            if existing:
                print(f"  existing collections: {existing}", flush=True)

        if any(e["state"] != "settled" for e in journal):
            print("reconciling unresolved batches from the previous run ...", flush=True)
            reconcile(client)

        print(f"ingesting into {database!r}/{collection!r} (batch_size={batch_size}, infer={infer}, "
              f"wait={wait}) ...", flush=True)

        batch: list[dict] = []
        last_doc_id: str | None = start_after

        def send(batch_items: list[dict]) -> None:
            entry = {
                "batch": len(journal) + 1,
                "docs": {"after": state["last_doc_id"], "last": last_doc_id},
                "n_items": len(batch_items),
                "ids": [it["id"] for it in batch_items],
                "state": "sending",
                "statuses": {},
            }
            state["last_doc_id"] = last_doc_id
            submit(entry, batch_items)

        for doc in docs:
            items = convert(doc, database, collection)
            batch.extend(items)
            docs_read += 1
            items_produced += len(items)
            by_kind.update(it["kind"] for it in items)
            state["docs_processed"] += 1
            last_doc_id = doc["doc_id"]
            if len(batch) >= batch_size:
                send(batch)
                batch = []

        if batch:
            send(batch)
            batch = []
        elif docs_read:
            state["last_doc_id"] = last_doc_id
            checkpoint()  # trailing docs that produced no items still advance the cursor

        if wait:
            while inflight:
                settle(client, inflight.popleft())

        if retry_failed:
            retry_failed_items(client)

    readiness = _recount(state)
    accepted = sorted(failed_ids) if accept_failures and readiness["resolved"] else []
    elapsed = time.monotonic() - started
    print(f"done: {docs_read:,} docs read, {items_produced:,} items produced, "
          f"{state['items_sent']:,} items sent in total, {len(failed_ids):,} failed ({elapsed / 60:.1f} min); "
          f"resolved={readiness['resolved']} ready={readiness['ready']}", flush=True)
    record_ = _finish(run_dir, cfg, state_path, status_log, dict(
        dry_run=False, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
        batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
        retry_failed=retry_failed, accept_failures=accept_failures,
        started_at=started_at, wall_time_s=round(elapsed, 3),
        corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read,
                    documents_processed_total=state["docs_processed"], **identity),
        converter=dict(state["converter"]),
        items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=state["items_sent"]),
        batches=dict(sent=state["batches_sent"], retried=state["retry"]["batches"],
                     unresolved=readiness["unresolved_batches"]),
        statuses=dict(state["statuses"]),
        readiness=readiness, ready=readiness["ready"], resolved=readiness["resolved"],
        failed_ids={k: dict(v) for k, v in sorted(failed_ids.items())},
        retry=dict(state["retry"]),
        accepted_failures=accepted,
        last_doc_id=state["last_doc_id"],
    ))
    if not readiness["ready"] and not accepted:
        raise IngestIncomplete(readiness, record_)
    if accepted:
        print(f"WARNING: {len(accepted):,} failed item(s) accepted; the collection is NOT complete "
              f"(ready=false is recorded in the manifest)", flush=True)
    return record_


def _finish(run_dir: Path, cfg: RunConfig, state_path: Path, status_log: Path, summary: dict) -> dict:
    """Write ``ingest_manifest.json`` and append the ``ingest`` stage to the run manifest."""
    record = {
        "stage": "ingest",
        "finished_at": manifest.now_iso(),
        "hydradb": {"base_url": cfg.hydradb.base_url, "endpoint": normalize_endpoint(cfg.hydradb.base_url),
                    "database": cfg.hydradb.database, "collection": cfg.hydradb.collection},
        **summary,
        "config": cfg.to_dict(),
        "state_file": state_path.name,
        "status_log": status_log.name,
    }
    out = run_dir / INGEST_MANIFEST_NAME
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    manifest.write_manifest(run_dir, "ingest", cfg.to_dict(), [out, state_path, status_log],
                            extra={"ingest": {k: record[k] for k in ("dry_run", "infer", "items", "batches",
                                                                       "statuses", "readiness", "ready",
                                                                       "resolved", "retry", "accepted_failures",
                                                                       "wall_time_s")}})
    return record
