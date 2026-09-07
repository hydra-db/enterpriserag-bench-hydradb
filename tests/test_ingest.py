"""Ingestion runner tests. No network: the admin client is replaced by an in-memory fake."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from erb_hydradb import corpus, ingest
from erb_hydradb.config import RunConfig
from erb_hydradb.convert import convert

ERB_REPO = Path(__file__).parent / "data" / "erb_repo"


class NoNetwork:
    def __init__(self, *a, **kw):
        raise AssertionError("HTTP client must not be constructed in a dry run")


class FakeAdminClient:
    """Records every call the runner makes; never touches the network."""

    instances: list["FakeAdminClient"] = []

    def __init__(self, api_key: str, base_url: str = "", **kw):
        assert api_key
        self.base_url = base_url
        self.ingested: list[list[dict]] = []
        self.infer_flags: list[bool] = []
        self.waited: list[list[str]] = []
        self.provisioned = False
        FakeAdminClient.instances.append(self)

    def create_database(self, database):
        self.provisioned = True
        return {"already_exists": True, "database": database}

    def wait_ready(self, database):
        return {"ready_for_ingestion": True}

    def list_collections(self, database):
        return []

    def ingest_app_knowledge(self, database, collection, items, upsert=True, infer=False):
        assert all(it["tenant_id"] == database and it["sub_tenant_id"] == collection for it in items)
        self.ingested.append(list(items))
        self.infer_flags.append(infer)
        return {"ok": True}

    def wait_processing(self, database, collection, source_ids, **kw):
        self.waited.append(list(source_ids))
        # first id of every batch "fails", the rest complete
        return {i: ("errored" if n == 0 else "completed") for n, i in enumerate(source_ids)}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


@pytest.fixture
def cfg() -> RunConfig:
    c = RunConfig(name="test")
    c.hydradb.database, c.hydradb.collection = "erb_test_db", "entire"
    return c


@pytest.fixture
def db_path(tmp_path) -> Path:
    p = tmp_path / "documents.sqlite"
    corpus.build_from_repo(ERB_REPO, p, quiet=True)
    return p


@pytest.fixture
def expected_items(db_path, cfg) -> list[dict]:
    out = []
    for doc in corpus.iter_documents(db_path):
        out.extend(convert(doc, cfg.hydradb.database, cfg.hydradb.collection))
    return out


@pytest.fixture
def fake_client(monkeypatch):
    FakeAdminClient.instances.clear()
    monkeypatch.setattr(ingest, "HydraDBAdminClient", FakeAdminClient)
    return FakeAdminClient


def test_dry_run_counts_match_convert_and_makes_no_http_calls(tmp_path, cfg, db_path, expected_items,
                                                              monkeypatch):
    monkeypatch.setattr(ingest, "HydraDBAdminClient", NoNetwork)
    monkeypatch.delenv("HYDRADB_API_KEY", raising=False)
    run_dir = tmp_path / "run"
    state = run_dir / "ingest_state.json"
    rec = ingest.run_ingest(cfg, db_path, state, dry_run=True, batch_size=10)

    assert rec["dry_run"] is True
    assert rec["items"]["produced"] == len(expected_items)
    kinds = {}
    for it in expected_items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    assert rec["items"]["by_kind"] == kinds
    assert rec["corpus"]["documents_read"] == 5
    assert rec["items"]["sent"] == 0 and rec["batches"]["sent"] == 0
    assert rec["batches"]["would_send"] >= 1

    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["items"] == rec["items"]
    assert on_disk["config"] == cfg.to_dict()
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert [s["stage"] for s in stages] == ["ingest"]
    assert "ingest_manifest.json" in stages[0]["files"]
    assert (run_dir / "SHA256SUMS").exists()
    assert not state.exists()  # a dry run never checkpoints


def test_full_run_batches_checkpoints_and_waits(tmp_path, cfg, db_path, expected_items, fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "ingest_state.json"
    rec = ingest.run_ingest(cfg, db_path, state_path, infer=True, batch_size=10, batch_sleep=0,
                            api_key="test-key", wait_window=1)

    [client] = fake_client.instances
    assert client.provisioned
    sent = [it for batch in client.ingested for it in batch]
    assert sent == expected_items
    assert all(client.infer_flags)
    assert len(client.ingested) >= 2
    assert all(len(b) >= 10 for b in client.ingested[:-1])

    # every id was waited on and its final status recorded (the --no-wait bug fix)
    waited = [i for ids in client.waited for i in ids]
    assert sorted(waited) == sorted(it["id"] for it in expected_items)
    log = [json.loads(line) for line in (run_dir / "ingest_status.jsonl").read_text().splitlines()]
    assert len(log) == len(expected_items)
    assert rec["statuses"] == {"settled": len(expected_items) - len(client.ingested),
                               "errored": len(client.ingested), "pending": 0}

    state = json.loads(state_path.read_text())
    assert state["last_doc_id"] == max(d["doc_id"] for d in corpus.iter_documents(db_path))
    assert state["docs_processed"] == 5
    assert state["items_sent"] == len(expected_items)
    assert state["batches_sent"] == len(client.ingested)
    assert state["infer"] is True
    assert not state_path.with_name(state_path.name + ".tmp").exists()

    assert rec["items"]["sent"] == len(expected_items) and rec["infer"] is True
    assert rec["batches"]["sent"] == len(client.ingested)
    assert rec["wall_time_s"] >= 0
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert stages[-1]["ingest"]["statuses"] == rec["statuses"]


def test_no_wait_skips_status_polling(tmp_path, cfg, db_path, fake_client):
    rec = ingest.run_ingest(cfg, db_path, tmp_path / "run" / "s.json", batch_size=10, batch_sleep=0,
                            api_key="k", wait=False, provision=False)
    [client] = fake_client.instances
    assert client.waited == [] and not client.provisioned
    assert rec["statuses"] == {"settled": 0, "errored": 0, "pending": 0}
    assert not (tmp_path / "run" / "ingest_status.jsonl").exists()


def test_resume_continues_after_checkpoint(tmp_path, cfg, db_path, expected_items, fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    ingest.run_ingest(cfg, db_path, state_path, limit=2, batch_size=1000, batch_sleep=0, api_key="k",
                      provision=False)
    first = fake_client.instances[-1]
    first_ids = {it["id"] for b in first.ingested for it in b}
    assert json.loads(state_path.read_text())["docs_processed"] == 2

    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=1000, batch_sleep=0,
                            api_key="k", provision=False)
    second = fake_client.instances[-1]
    second_ids = {it["id"] for b in second.ingested for it in b}
    assert first_ids.isdisjoint(second_ids)
    assert first_ids | second_ids == {it["id"] for it in expected_items}
    assert rec["corpus"]["documents_read"] == 3
    assert rec["corpus"]["documents_processed_total"] == 5
    assert rec["items"]["sent"] == len(expected_items)

    # a third resume finds nothing left and sends nothing
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=1000, batch_sleep=0,
                            api_key="k", provision=False)
    assert fake_client.instances[-1].ingested == []
    assert rec["corpus"]["documents_read"] == 0


def test_resume_refuses_a_checkpoint_from_another_target(tmp_path, cfg, db_path, fake_client):
    state_path = tmp_path / "s.json"
    ingest.run_ingest(cfg, db_path, state_path, limit=1, batch_sleep=0, api_key="k", provision=False)
    other = RunConfig(name="other")
    other.hydradb.database, other.hydradb.collection = cfg.hydradb.database, "another"
    with pytest.raises(RuntimeError, match="refusing to resume"):
        ingest.run_ingest(other, db_path, state_path, resume=True, api_key="k", provision=False)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, infer=True, api_key="k", provision=False)


def test_missing_api_key_is_a_clear_error(tmp_path, cfg, db_path, fake_client, monkeypatch):
    monkeypatch.delenv("HYDRADB_API_KEY", raising=False)
    monkeypatch.setattr(ingest, "load_dotenv", lambda *a, **kw: False)  # ignore any developer .env
    with pytest.raises(RuntimeError, match="HYDRADB_API_KEY"):
        ingest.run_ingest(cfg, db_path, tmp_path / "s.json", provision=False)
