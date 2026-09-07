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
   status is appended to ``ingest_status.jsonl`` next to the state file.

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
  ``ingest_status.jsonl`` (exactly one line per id, keyed by batch). This keeps
  the state file bounded by the number of *unresolved* batches, which is at
  most ``wait_window`` + 1 when ``wait=True``.

On ``resume=True`` the runner first reconciles every entry that is not
``settled`` before converting anything new: batches stuck in ``sending`` are
re-sent, then every unresolved id is polled and recorded. Re-sending is safe
because ingestion is an upsert (``upsert=true``) keyed by deterministic item
ids (``convert`` derives every id from the document id), so a batch whose
request may or may not have reached the server can simply be sent again.
A ``sending`` batch is rebuilt by re-converting its doc range; the rebuilt ids
must equal the journaled ids or the resume is refused.

Readiness
---------
``ingest_manifest.json`` reports ``readiness`` (also mirrored as a top-level
``ready`` flag): ``items_sent``, ``items_settled``, ``items_errored``,
``items_pending``, ``unresolved_batches`` and ``ready``, which is true only
when ``items_sent == items_settled + items_errored``, ``items_pending == 0``
and ``unresolved_batches == 0``. A run that ends otherwise (status timeouts,
``wait=False``) still writes its manifest and then raises ``IngestIncomplete``
so the CLI exits nonzero; ``--resume`` finishes the reconciliation.

Resume identity
---------------
A checkpoint is only resumed for the same database / collection / ``infer``
flag, the same source scope (``source_types`` and ``limit``) and the same
corpus (row count plus a sha256 over every ``doc_id`` in order).

Memory stays bounded regardless of corpus size: one document is converted at a
time and at most ``batch_size`` items (plus the id lists of unresolved
batches) are resident.

Files written next to ``state_path``:

* ``<state_path>``          checkpoint (cursor, counters, identity guard, journal);
* ``ingest_status.jsonl``   one ``{"batch", "id", "status"}`` line per resolved id;
* ``ingest_manifest.json``  the run summary (counts, readiness, wall time, config);
* ``manifest.json`` / ``SHA256SUMS`` via ``manifest.write_manifest``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from . import corpus, manifest
from .config import RunConfig
from .convert import convert
from .hydradb_client import HydraDBAdminClient

STATUS_LOG_NAME = "ingest_status.jsonl"
INGEST_MANIFEST_NAME = "ingest_manifest.json"
ERRORED_STATES = {"errored", "failed"}
READINESS_KEYS = ("items_sent", "items_settled", "items_errored", "items_pending", "unresolved_batches")


class IngestIncomplete(RuntimeError):
    """The run finished but the collection is not ready to query.

    Raised by ``run_ingest`` *after* ``ingest_manifest.json`` has been written
    with ``ready: false``. ``readiness`` holds the counts (see ``READINESS_KEYS``
    plus ``ready``), ``manifest`` the record that was written; each count is
    also available as an attribute (``exc.items_pending`` ...).
    """

    def __init__(self, readiness: dict, manifest: dict) -> None:
        self.readiness = dict(readiness)
        self.manifest = manifest
        for key in READINESS_KEYS:
            setattr(self, key, readiness[key])
        super().__init__(
            "ingestion is not ready to query: "
            f"{readiness['items_sent']:,} sent, {readiness['items_settled']:,} settled, "
            f"{readiness['items_errored']:,} errored, {readiness['items_pending']:,} pending, "
            f"{readiness['unresolved_batches']} unresolved batch(es); re-run with --resume to reconcile")


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


def corpus_identity(db_path: str | Path) -> dict:
    """Weak identity of ``documents.sqlite``: row count + sha256 over every doc_id in order.

    ``corpus.py`` exposes no fingerprint of its own, so this is derived here.
    It is *weak*: two corpora with the same doc ids but different content hash
    the same. It is enough for what the resume needs, because every item id is
    derived from its doc id, so equal doc ids mean the journal's ids are the
    ids this corpus produces.
    """
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
    return {"documents": total, "doc_ids_sha256": h.hexdigest()}


def _new_state(cfg: RunConfig, infer: bool, source_types: list[str] | None, limit: int | None,
               db_path: Path, identity: dict) -> dict:
    return {
        "database": cfg.hydradb.database,
        "collection": cfg.hydradb.collection,
        "infer": infer,
        "source_types": _scope(source_types),
        "limit": limit,
        "corpus": {"db_path": str(db_path), **identity},
        "last_doc_id": None,
        "docs_processed": 0,
        "items_sent": 0,
        "batches_sent": 0,
        "items_by_kind": {},
        "statuses": {"settled": 0, "errored": 0, "pending": 0},
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
    want = (cfg.hydradb.database, cfg.hydradb.collection, infer)
    have = (state.get("database"), state.get("collection"), bool(state.get("infer")))
    if want != have:
        raise RuntimeError(
            f"refusing to resume: checkpoint was written for database/collection/infer={have!r}, "
            f"this run is {want!r}")
    want_scope = (_scope(source_types), limit)
    have_scope = (state.get("source_types"), state.get("limit"))
    if want_scope != have_scope:
        raise RuntimeError(
            f"refusing to resume: checkpoint was written for source_types/limit={have_scope!r}, "
            f"this run is {want_scope!r}")
    have_corpus = state.get("corpus") or {}
    for key in ("documents", "doc_ids_sha256"):
        if have_corpus.get(key) != identity[key]:
            raise RuntimeError(
                f"refusing to resume: checkpoint was written for a corpus with {key}={have_corpus.get(key)!r}, "
                f"this corpus has {identity[key]!r}")


# ---------------------------------------------------------------------------
# journal accounting
# ---------------------------------------------------------------------------
def _recount(state: dict) -> dict:
    """Recompute counters from the journal (never incremental, so re-polls cannot double count)."""
    sent = settled = errored = 0
    batches = unresolved = 0
    for e in state["journal"]:
        if e["state"] == "settled":
            sent += e["n_items"]
            batches += 1
            settled += e["counts"]["settled"]
            errored += e["counts"]["errored"]
        elif e["state"] == "sent":
            sent += e["n_items"]
            batches += 1
            unresolved += 1
            for st in e["statuses"].values():
                if st in ERRORED_STATES:
                    errored += 1
                else:
                    settled += 1
        else:  # "sending": the request may never have reached the server; not counted until re-sent
            unresolved += 1
    pending = sent - settled - errored
    state["items_sent"] = sent
    state["batches_sent"] = batches
    state["statuses"] = {"settled": settled, "errored": errored, "pending": pending}
    return {
        "items_sent": sent, "items_settled": settled, "items_errored": errored, "items_pending": pending,
        "unresolved_batches": unresolved,
        "ready": sent == settled + errored and pending == 0 and unresolved == 0,
    }


def _rebuild_batch(db_path: Path, source_types: list[str] | None, entry: dict, database: str,
                   collection: str) -> list[dict]:
    """Re-convert the doc range of a ``sending`` journal entry into its items."""
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
               wait_window: int = 4, wait_timeout_s: float = 1800.0) -> dict:
    """Ingest the corpus into ``cfg.hydradb.database`` / ``cfg.hydradb.collection``.

    ``dry_run`` converts and counts without constructing an HTTP client or
    touching the checkpoint. Returns the ingestion manifest dict (also written
    to ``ingest_manifest.json`` next to ``state_path``).

    Raises ``IngestIncomplete`` (after writing the manifest with
    ``ready: false``) when any sent item is still pending or any batch is
    unresolved at the end of the run; ``resume=True`` reconciles it.
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
    identity = corpus_identity(db_path) if not dry_run else {}
    if resume and not dry_run:
        state = load_state(state_path)
        if state:
            _check_state_matches(state, cfg, infer, source_types, limit, identity)
            unresolved = [e for e in state["journal"] if e["state"] != "settled"]
            print(f"resuming after doc_id {state['last_doc_id']} "
                  f"({state['docs_processed']:,} docs / {state['items_sent']:,} items already sent, "
                  f"{len(unresolved)} unresolved batch(es) to reconcile)", flush=True)
        else:
            print("no checkpoint found; starting from the beginning", flush=True)
    elif state_path.exists() and not dry_run:
        print(f"note: ignoring existing checkpoint {state_path} (pass resume=True to continue it)",
              flush=True)
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
        readiness["ready"] = False  # nothing was sent: the collection is not ready because of this run
        return _finish(run_dir, cfg, state_path, status_log, dict(
            dry_run=True, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
            batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
            started_at=started_at, wall_time_s=round(time.monotonic() - started, 3),
            corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read),
            items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=0),
            batches=dict(sent=0, would_send=would_send),
            statuses=dict(settled=0, errored=0, pending=0),
            readiness=readiness, ready=False,
        ))

    # -- real run ----------------------------------------------------------
    load_dotenv()
    api_key = api_key or os.getenv("HYDRADB_API_KEY")
    if not api_key:
        raise RuntimeError("HYDRADB_API_KEY is not set (put it in .env or pass api_key=)")

    journal: list[dict] = state["journal"]
    inflight: deque[dict] = deque()  # entries in state "sent" not yet polled in this run

    def checkpoint(last_doc_id: str | None) -> None:
        state["last_doc_id"] = last_doc_id
        state["items_by_kind"] = dict(sorted(by_kind.items()))
        state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _recount(state)
        _atomic_write_json(state_path, state)

    def settle(client: HydraDBAdminClient, entry: dict) -> None:
        """Poll the entry's unresolved ids once; record terminal statuses; compact when complete."""
        todo = [i for i in entry["ids"] if i not in entry["statuses"]]
        final = client.wait_processing(database, collection, todo, timeout_s=wait_timeout_s) if todo else {}
        with open(status_log, "a", encoding="utf-8") as f:
            for i in todo:
                st = final.get(i, "pending")
                if st == "pending":
                    continue  # stays unresolved; logged once it reaches a terminal state
                entry["statuses"][i] = st
                f.write(json.dumps({"batch": entry["batch"], "id": i, "status": st}) + "\n")
        if len(entry["statuses"]) == len(entry["ids"]):
            errored = sum(1 for st in entry["statuses"].values() if st in ERRORED_STATES)
            entry["counts"] = {"settled": len(entry["ids"]) - errored, "errored": errored}
            entry["state"] = "settled"
            del entry["ids"], entry["statuses"]
        else:
            print(f"  batch {entry['batch']}: {len(entry['ids']) - len(entry['statuses'])} id(s) still pending",
                  flush=True)
        checkpoint(state["last_doc_id"])

    def reconcile(client: HydraDBAdminClient) -> None:
        """Finish every unresolved journal entry left by a previous process before ingesting anything new."""
        for entry in journal:
            if entry["state"] == "sending":
                print(f"  re-sending batch {entry['batch']} ({entry['n_items']} items) whose send was "
                      f"interrupted", flush=True)
                items = _rebuild_batch(db_path, source_types, entry, database, collection)
                client.ingest_app_knowledge(database, collection, items, infer=infer)
                entry["state"] = "sent"
                checkpoint(state["last_doc_id"])
        for entry in journal:
            if entry["state"] == "sent":
                if wait:
                    print(f"  polling batch {entry['batch']} ({len(entry['ids']) - len(entry['statuses'])} "
                          f"unresolved id(s))", flush=True)
                    settle(client, entry)
                else:
                    print(f"  batch {entry['batch']} left unresolved (wait=False)", flush=True)

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
            journal.append(entry)
            checkpoint(last_doc_id)  # cursor and journal entry land in the same atomic write
            client.ingest_app_knowledge(database, collection, batch_items, infer=infer)
            entry["state"] = "sent"
            inflight.append(entry)
            checkpoint(last_doc_id)
            elapsed = time.monotonic() - started
            print(f"  docs={state['docs_processed']:,} items={state['items_sent']:,} "
                  f"batches={state['batches_sent']:,} ({elapsed:.0f}s)", flush=True)
            if wait:
                while len(inflight) > wait_window:
                    settle(client, inflight.popleft())
            if batch_sleep > 0:
                time.sleep(batch_sleep)

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
            checkpoint(last_doc_id)  # trailing docs that produced no items still advance the cursor

        if wait:
            while inflight:
                settle(client, inflight.popleft())

    readiness = _recount(state)
    elapsed = time.monotonic() - started
    print(f"done: {docs_read:,} docs read, {items_produced:,} items produced, "
          f"{state['items_sent']:,} items sent in total ({elapsed / 60:.1f} min); "
          f"ready={readiness['ready']}", flush=True)
    record = _finish(run_dir, cfg, state_path, status_log, dict(
        dry_run=False, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
        batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
        started_at=started_at, wall_time_s=round(elapsed, 3),
        corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read,
                    documents_processed_total=state["docs_processed"], **identity),
        items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=state["items_sent"]),
        batches=dict(sent=state["batches_sent"], unresolved=readiness["unresolved_batches"]),
        statuses=dict(state["statuses"]),
        readiness=readiness, ready=readiness["ready"],
        last_doc_id=state["last_doc_id"],
    ))
    if not readiness["ready"]:
        raise IngestIncomplete(readiness, record)
    return record


def _finish(run_dir: Path, cfg: RunConfig, state_path: Path, status_log: Path, summary: dict) -> dict:
    """Write ``ingest_manifest.json`` and append the ``ingest`` stage to the run manifest."""
    record = {
        "stage": "ingest",
        "finished_at": manifest.now_iso(),
        "hydradb": {"base_url": cfg.hydradb.base_url, "database": cfg.hydradb.database,
                    "collection": cfg.hydradb.collection},
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
                                                                       "wall_time_s")}})
    return record
