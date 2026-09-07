"""Streaming, checkpointed, resumable ingestion of the corpus into one HydraDB collection.

Pipeline:

1. stream rows from ``documents.sqlite`` in ``doc_id`` order (``corpus.iter_documents``);
2. convert each row into typed ``app_knowledge`` items (``convert.convert``);
3. optionally provision the database (``POST /databases``, poll until
   ``ready_for_ingestion``);
4. accumulate items into batches, ``POST /context/ingest`` each batch, and
   checkpoint progress (atomically) after every batch so a restart can resume
   from the last sent batch;
5. unless ``wait=False``, poll ``/context/status`` for every batch's ids until
   they settle, keeping a trailing window of ``wait_window`` batches in flight
   so the server indexes while the client keeps sending. Each id's final
   status is appended to ``ingest_status.jsonl`` next to the state file.

Memory stays bounded regardless of corpus size: one document is converted at a
time and at most ``batch_size`` items (plus the id lists of in-flight batches)
are resident.

Files written next to ``state_path``:

* ``<state_path>``          checkpoint (last doc id, counters, config guard);
* ``ingest_status.jsonl``   one ``{"batch", "id", "status"}`` line per settled id;
* ``ingest_manifest.json``  the run summary (counts, wall time, config);
* ``manifest.json`` / ``SHA256SUMS`` via ``manifest.write_manifest``.
"""

from __future__ import annotations

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


def _new_state(cfg: RunConfig, infer: bool) -> dict:
    return {
        "database": cfg.hydradb.database,
        "collection": cfg.hydradb.collection,
        "infer": infer,
        "last_doc_id": None,
        "docs_processed": 0,
        "items_sent": 0,
        "batches_sent": 0,
        "items_by_kind": {},
        "statuses": {"settled": 0, "errored": 0, "pending": 0},
        "updated_at": None,
    }


def load_state(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_state_matches(state: dict, cfg: RunConfig, infer: bool) -> None:
    want = (cfg.hydradb.database, cfg.hydradb.collection, infer)
    have = (state.get("database"), state.get("collection"), bool(state.get("infer")))
    if want != have:
        raise RuntimeError(
            f"refusing to resume: checkpoint was written for database/collection/infer={have!r}, "
            f"this run is {want!r}")


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
    if resume and not dry_run:
        state = load_state(state_path)
        if state:
            _check_state_matches(state, cfg, infer)
            print(f"resuming after doc_id {state['last_doc_id']} "
                  f"({state['docs_processed']:,} docs / {state['items_sent']:,} items already sent)",
                  flush=True)
        else:
            print("no checkpoint found; starting from the beginning", flush=True)
    elif state_path.exists() and not dry_run:
        print(f"note: ignoring existing checkpoint {state_path} (pass resume=True to continue it)",
              flush=True)
    if state is None:
        state = _new_state(cfg, infer)
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
        return _finish(run_dir, cfg, state_path, status_log, dict(
            dry_run=True, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
            batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
            started_at=started_at, wall_time_s=round(time.monotonic() - started, 3),
            corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read),
            items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=0),
            batches=dict(sent=0, would_send=would_send),
            statuses=dict(settled=0, errored=0, pending=0),
        ))

    # -- real run ----------------------------------------------------------
    load_dotenv()
    api_key = api_key or os.getenv("HYDRADB_API_KEY")
    if not api_key:
        raise RuntimeError("HYDRADB_API_KEY is not set (put it in .env or pass api_key=)")

    inflight: deque[tuple[int, list[str]]] = deque()

    def checkpoint(last_doc_id: str | None) -> None:
        state["last_doc_id"] = last_doc_id
        state["items_by_kind"] = dict(sorted(by_kind.items()))
        state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _atomic_write_json(state_path, state)

    def settle(client: HydraDBAdminClient, batch_no: int, ids: list[str]) -> None:
        final = client.wait_processing(database, collection, ids, timeout_s=wait_timeout_s)
        with open(status_log, "a", encoding="utf-8") as f:
            for i in ids:
                st = final.get(i, "pending")
                if st in ERRORED_STATES:
                    state["statuses"]["errored"] += 1
                elif st == "pending":
                    state["statuses"]["pending"] += 1
                else:
                    state["statuses"]["settled"] += 1
                f.write(json.dumps({"batch": batch_no, "id": i, "status": st}) + "\n")
        checkpoint(state["last_doc_id"])

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

        print(f"ingesting into {database!r}/{collection!r} (batch_size={batch_size}, infer={infer}, "
              f"wait={wait}) ...", flush=True)

        batch: list[dict] = []
        last_doc_id: str | None = start_after

        def send(batch_items: list[dict]) -> None:
            client.ingest_app_knowledge(database, collection, batch_items, infer=infer)
            state["items_sent"] += len(batch_items)
            state["batches_sent"] += 1
            inflight.append((state["batches_sent"], [it["id"] for it in batch_items]))
            checkpoint(last_doc_id)
            elapsed = time.monotonic() - started
            print(f"  docs={state['docs_processed']:,} items={state['items_sent']:,} "
                  f"batches={state['batches_sent']:,} ({elapsed:.0f}s)", flush=True)
            if wait:
                while len(inflight) > wait_window:
                    settle(client, *inflight.popleft())
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
                settle(client, *inflight.popleft())

    elapsed = time.monotonic() - started
    print(f"done: {docs_read:,} docs read, {items_produced:,} items produced, "
          f"{state['items_sent']:,} items sent in total ({elapsed / 60:.1f} min)", flush=True)
    return _finish(run_dir, cfg, state_path, status_log, dict(
        dry_run=False, infer=infer, wait=wait, resume=resume, limit=limit, source_types=source_types,
        batch_size=batch_size, batch_sleep=batch_sleep, provision=provision,
        started_at=started_at, wall_time_s=round(elapsed, 3),
        corpus=dict(db_path=str(db_path), documents_remaining_at_start=remaining, documents_read=docs_read,
                    documents_processed_total=state["docs_processed"]),
        items=dict(produced=items_produced, by_kind=dict(sorted(by_kind.items())), sent=state["items_sent"]),
        batches=dict(sent=state["batches_sent"]),
        statuses=dict(state["statuses"]),
        last_doc_id=state["last_doc_id"],
    ))


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
                                                                       "statuses", "wall_time_s")}})
    return record
