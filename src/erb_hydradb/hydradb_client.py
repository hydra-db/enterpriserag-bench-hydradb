"""HydraDB HTTP clients used by the benchmark harness.

Two thin clients over the public HydraDB API, kept deliberately small so a
reader can see exactly which calls the benchmark result depends on:

* ``HydraDBAdminClient`` (sync) — database provisioning, typed app-knowledge
  ingestion via ``POST /context/ingest``, and indexing-status polling. Used by
  ``erb-hydradb ingest``.
* ``HydraDBQueryClient`` (async) — ``POST /query``. Used by ``erb-hydradb generate``.

Both send ``Authorization: Bearer <api key>``. The admin client also sends
``API-Version: 2`` (the v2 tenancy surface: databases / collections).
Every request body the harness sends is built in this file and nowhere else.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

API_VERSION = "2"
DEFAULT_BASE_URL = "https://api.hydradb.com"

_RETRY_IN_RE = re.compile(r"retry in (\d+) second")


def _retry_seconds_from(body: dict) -> float | None:
    """Best-effort parse of the 429 retry hint from error/detail messages."""
    texts = []
    err = body.get("error") or {}
    texts.append(err.get("message") or "")
    detail = body.get("detail") or {}
    texts.append(detail.get("message") or err.get("detail") or "")
    for text in texts:
        m = _RETRY_IN_RE.search(str(text))
        if m:
            return float(m.group(1))
    return None


class HydraDBAdminClient:
    """Sync client: databases, ingestion, indexing status."""

    # Vector indexing is complete once a source reports completed/success.
    # Graph-backed app knowledge then runs an async graph-enrichment sweep that
    # surfaces as ``graph_creation`` on the source-level status; the vectors are
    # already searchable at that point, so those states count as settled.
    TERMINAL_STATES = {"completed", "success", "errored", "failed"}
    SETTLED_STATES = TERMINAL_STATES | {"graph_creation", "graph_completed"}

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, max_429_retries: int = 10) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_429_retries = max_429_retries
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "API-Version": API_VERSION},
            timeout=httpx.Timeout(120.0, connect=30.0),
        )

    # -- transport --------------------------------------------------------

    def _post_resilient(self, url: str, *, data=None, json_body=None) -> httpx.Response:
        """POST with retry on 429, honouring the server's retry hint."""
        for attempt in range(1, self.max_429_retries + 1):
            resp = self._http.post(url, data=data, json=json_body)
            if resp.status_code == 429:
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    body = {}
                seconds = _retry_seconds_from(body) or 30.0
                print(f"  rate-limited on {url} (attempt {attempt}): sleeping {seconds:.0f}s", flush=True)
                time.sleep(seconds)
                continue
            return resp
        raise RuntimeError(f"gave up after {self.max_429_retries} retries on 429 for POST {url}")

    # -- databases ----------------------------------------------------------

    def create_database(self, database: str) -> dict:
        """POST /databases. A repeat call on an existing database is treated as success."""
        resp = self._http.post("/databases", json={"database": database})
        if resp.status_code in (200, 201, 202):
            return resp.json()
        body = resp.text[:300]
        if resp.status_code == 409 or "exists" in body.lower():
            return {"already_exists": True, "database": database}
        resp.raise_for_status()
        return resp.json()

    def database_status(self, database: str) -> dict:
        resp = self._http.get("/databases/status", params={"database": database})
        resp.raise_for_status()
        return resp.json().get("data", resp.json())

    def wait_ready(self, database: str, timeout_s: float = 900.0, poll_interval_s: float = 20.0) -> dict:
        """Poll /databases/status until ready_for_ingestion is true."""
        deadline = time.monotonic() + timeout_s
        last: dict = {}
        while time.monotonic() < deadline:
            last = self.database_status(database)
            ready = last.get("ready_for_ingestion") or (last.get("infra") or {}).get("ready_for_ingestion") or False
            if ready:
                return last
            print(f"  provisioning... {last!r}", flush=True)
            time.sleep(poll_interval_s)
        raise TimeoutError(f"database {database!r} not ready for ingestion after {timeout_s}s: {last!r}")

    def list_collections(self, database: str) -> list[str]:
        resp = self._http.get("/databases/collections", params={"database": database})
        resp.raise_for_status()
        data = resp.json().get("data", resp.json())
        return data.get("collections") or data.get("sub_tenant_ids") or []

    # -- ingestion ----------------------------------------------------------

    def ingest_app_knowledge(self, database: str, collection: str, items: list[dict],
                             upsert: bool = True, infer: bool = False) -> dict:
        """POST /context/ingest (type=knowledge) with a typed ``app_knowledge`` array.

        ``infer=True`` opts the batch into HydraDB's ingest-time inference step.
        """
        data = {
            "type": "knowledge",
            "tenant_id": database,
            "sub_tenant_id": collection,
            "infer": str(infer).lower(),
            "upsert": str(upsert).lower(),
            "app_knowledge": json.dumps(items),
        }
        resp = self._post_resilient("/context/ingest", data=data)
        if resp.status_code >= 400:
            raise RuntimeError(f"ingest failed {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def ingest_status(self, database: str, collection: str, ids: list[str]) -> dict:
        resp = self._http.get("/context/status",
                              params={"tenant_id": database, "sub_tenant_id": collection, "ids": ids})
        resp.raise_for_status()
        return resp.json()

    def wait_processing(self, database: str, collection: str, source_ids: list[str],
                        timeout_s: float = 1800.0, poll_interval_s: float = 15.0,
                        poll_batch_size: int = 50) -> dict[str, str]:
        """Poll /context/status until every id has reached a settled state.

        Returns ``{id: final_status}`` for every id (``"pending"`` for ids that
        never settled before the timeout). Never raises on errored objects; the
        caller records them in the ingestion manifest.
        """
        deadline = time.monotonic() + timeout_s
        pending = set(source_ids)
        final: dict[str, str] = {}
        while pending and time.monotonic() < deadline:
            for start in range(0, len(source_ids), poll_batch_size):
                ids = [i for i in source_ids[start:start + poll_batch_size] if i in pending]
                if not ids:
                    continue
                body = self.ingest_status(database, collection, ids)
                for s in body.get("data", {}).get("statuses", []):
                    st = s.get("indexing_status")
                    if st in self.SETTLED_STATES:
                        pending.discard(s["id"])
                        final[s["id"]] = st
            if pending:
                print(f"  processing... {len(pending)}/{len(source_ids)} pending", flush=True)
                time.sleep(poll_interval_s)
        for i in pending:
            final[i] = "pending"
        return final

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "HydraDBAdminClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class HydraDBQueryClient:
    """Async client: ``POST /query``."""

    def __init__(self, api_key: str, database: str, base_url: str = DEFAULT_BASE_URL,
                 timeout_s: float = 60.0) -> None:
        self.database = database
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )

    async def query(self, collection: str, query: str, *, max_results: int = 100,
                    mode: str = "thinking", alpha: float = 0.5, query_by: str = "hybrid",
                    graph_context: bool = False, query_apps: bool = False) -> list[dict]:
        """Return the chunk list from ``POST /query`` (``data.chunks``).

        ``query_apps`` is sent only when True so the default wire body is the
        server default. Every parameter here is recorded in the run config.
        """
        body = {
            "tenant_id": self.database,
            "sub_tenant_id": collection,
            "query": query,
            "type": "knowledge",
            "max_results": max_results,
            "mode": mode,
            "alpha": alpha,
            "graph_context": graph_context,
            "query_by": query_by,
        }
        if query_apps:
            body["query_apps"] = True
        resp = await self._http.post("/query", json=body)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("chunks", [])

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "HydraDBQueryClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
