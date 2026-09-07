"""Ingestion runner tests. No network: the admin client is replaced by an in-memory fake.

The interruption tests model a process dying at a specific point of the
send -> journal -> poll cycle by raising inside the fake client, then resuming
with a healthy fake and checking that the journal converges to the accounting
of an uninterrupted run (reviewer finding P1-2).
"""

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
    """Records every call the runner makes; never touches the network.

    Class-level knobs (reset by the ``fake_client`` fixture) inject a crash:

    * ``fail_send_on_call``: the N-th ``ingest_app_knowledge`` call is recorded
      in ``aborted`` and then raises (the request "never reached the server");
    * ``fail_wait_on_call``: the N-th ``wait_processing`` call raises before
      returning anything (the process dies mid-poll);
    * ``pending_ids``: ids reported as never settling (status-poll timeout).
    """

    instances: list["FakeAdminClient"] = []
    fail_send_on_call: int | None = None
    fail_wait_on_call: int | None = None
    pending_ids: set[str] = set()

    def __init__(self, api_key: str, base_url: str = "", **kw):
        assert api_key
        self.base_url = base_url
        self.ingested: list[list[dict]] = []
        self.aborted: list[list[dict]] = []
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
        if FakeAdminClient.fail_send_on_call == len(self.ingested) + len(self.aborted) + 1:
            self.aborted.append(list(items))
            raise ConnectionError("simulated crash: request never reached the server")
        self.ingested.append(list(items))
        self.infer_flags.append(infer)
        return {"ok": True}

    def wait_processing(self, database, collection, source_ids, **kw):
        if FakeAdminClient.fail_wait_on_call == len(self.waited) + 1:
            raise ConnectionError("simulated crash while polling status")
        self.waited.append(list(source_ids))
        # the first id of every poll "fails", the rest complete, unless told to stay pending
        return {i: ("pending" if i in FakeAdminClient.pending_ids else "errored" if n == 0 else "completed")
                for n, i in enumerate(source_ids)}

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
    FakeAdminClient.fail_send_on_call = None
    FakeAdminClient.fail_wait_on_call = None
    FakeAdminClient.pending_ids = set()
    monkeypatch.setattr(ingest, "HydraDBAdminClient", FakeAdminClient)
    return FakeAdminClient


def _sent_ids(clients) -> list[str]:
    return [it["id"] for c in clients for batch in c.ingested for it in batch]


def _waited_ids(clients) -> list[str]:
    return [i for c in clients for ids in c.waited for i in ids]


def _status_log(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "ingest_status.jsonl").read_text().splitlines()]


def _assert_converged(rec: dict, run_dir: Path, state_path: Path, db_path: Path, clients, expected_items,
                      expected_batches):
    """The accounting a run must show once every item is sent and resolved, however it got there."""
    n = len(expected_items)
    assert rec["ready"] is True
    assert rec["readiness"] == {"items_sent": n, "items_settled": n - expected_batches,
                                "items_errored": expected_batches, "items_pending": 0,
                                "unresolved_batches": 0, "ready": True}
    assert rec["statuses"] == {"settled": n - expected_batches, "errored": expected_batches, "pending": 0}
    assert rec["items"]["sent"] == n and rec["batches"]["sent"] == expected_batches
    # no document skipped, no item sent twice
    assert sorted(_sent_ids(clients)) == sorted(it["id"] for it in expected_items)
    # every id polled exactly once, and logged exactly once with a terminal status
    assert sorted(_waited_ids(clients)) == sorted(it["id"] for it in expected_items)
    log = _status_log(run_dir)
    assert len(log) == n and len({r["id"] for r in log}) == n
    assert all(r["status"] in ("completed", "errored") for r in log)
    state = json.loads(state_path.read_text())
    assert state["items_sent"] == n and state["batches_sent"] == expected_batches
    assert [e["state"] for e in state["journal"]] == ["settled"] * expected_batches
    assert all("ids" not in e for e in state["journal"])  # settled entries are compacted
    assert sum(e["counts"]["settled"] + e["counts"]["errored"] for e in state["journal"]) == n
    assert state["last_doc_id"] == max(d["doc_id"] for d in corpus.iter_documents(db_path))
    assert not state_path.with_name(state_path.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
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
    assert rec["ready"] is False and rec["readiness"]["items_sent"] == 0  # nothing sent: not ready

    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["items"] == rec["items"]
    assert on_disk["config"] == cfg.to_dict()
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert [s["stage"] for s in stages] == ["ingest"]
    assert "ingest_manifest.json" in stages[0]["files"]
    assert (run_dir / "SHA256SUMS").exists()
    assert not state.exists()  # a dry run never checkpoints and writes no journal
    assert not (run_dir / "ingest_status.jsonl").exists()


# ---------------------------------------------------------------------------
# uninterrupted runs
# ---------------------------------------------------------------------------
def test_full_run_batches_journals_waits_and_is_ready(tmp_path, cfg, db_path, expected_items, fake_client):
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

    _assert_converged(rec, run_dir, state_path, db_path, [client], expected_items, len(client.ingested))
    state = json.loads(state_path.read_text())
    assert state["docs_processed"] == 5
    assert state["infer"] is True
    assert state["source_types"] is None and state["limit"] is None
    assert state["corpus"]["documents"] == 5 and len(state["corpus"]["doc_ids_sha256"]) == 64
    # the journal records each batch's doc range; ranges tile the corpus in cursor order
    ranges = [(e["docs"]["after"], e["docs"]["last"]) for e in state["journal"]]
    assert ranges[0][0] is None
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))
    assert ranges[-1][1] == state["last_doc_id"]

    assert rec["infer"] is True and rec["wall_time_s"] >= 0
    assert rec["corpus"]["doc_ids_sha256"] == state["corpus"]["doc_ids_sha256"]
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert stages[-1]["ingest"]["statuses"] == rec["statuses"]
    assert stages[-1]["ingest"]["readiness"] == rec["readiness"] and stages[-1]["ingest"]["ready"] is True


def test_no_wait_is_not_ready_until_resumed(tmp_path, cfg, db_path, expected_items, fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        ingest.run_ingest(cfg, db_path, state_path, batch_size=10, batch_sleep=0, api_key="k", wait=False,
                          provision=False)
    [client] = fake_client.instances
    assert client.waited == [] and not client.provisioned
    assert not (run_dir / "ingest_status.jsonl").exists()
    exc = exc_info.value
    n, batches = len(expected_items), len(client.ingested)
    assert exc.readiness == {"items_sent": n, "items_settled": 0, "items_errored": 0, "items_pending": n,
                             "unresolved_batches": batches, "ready": False}
    assert (exc.items_sent, exc.items_pending, exc.unresolved_batches) == (n, n, batches)
    assert "pending" in str(exc) and "--resume" in str(exc)
    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["ready"] is False and on_disk["readiness"] == exc.readiness == exc.manifest["readiness"]
    assert on_disk["statuses"] == {"settled": 0, "errored": 0, "pending": n}

    # --resume with wait=True polls the journaled ids without re-sending anything
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=10, batch_sleep=0, api_key="k",
                            provision=False)
    resumed = fake_client.instances[-1]
    assert resumed.ingested == []
    _assert_converged(rec, run_dir, state_path, db_path, [client, resumed], expected_items, batches)


# ---------------------------------------------------------------------------
# interruptions (reviewer finding P1-2)
# ---------------------------------------------------------------------------
def test_reviewer_fixture_crash_before_any_status_poll_is_reconciled_on_resume(tmp_path, cfg, db_path,
                                                                                expected_items, fake_client):
    """The external reviewer's interruption fixture: one batch sent, the process dies on the first poll.

    Before the journal, the resumed run reported 34 sent / 0 settled / 0 pending
    / 0 status polls and a complete-looking manifest.
    """
    run_dir = tmp_path / "ingest-resume"
    state_path = run_dir / "ingest_state.json"
    fake_client.fail_wait_on_call = 1
    with pytest.raises(ConnectionError):
        ingest.run_ingest(cfg, db_path, state_path, batch_size=10000, batch_sleep=0, provision=False, api_key="mock")
    assert not (run_dir / "ingest_manifest.json").exists()  # a crashed run leaves no manifest behind
    state = json.loads(state_path.read_text())
    assert state["items_sent"] == len(expected_items) and state["statuses"]["pending"] == len(expected_items)
    [entry] = state["journal"]
    assert entry["state"] == "sent" and entry["ids"] == [it["id"] for it in expected_items]

    fake_client.fail_wait_on_call = None
    rec = ingest.run_ingest(cfg, db_path, state_path, batch_size=10000, batch_sleep=0, provision=False,
                            api_key="mock", resume=True)
    first, resumed = fake_client.instances
    assert resumed.ingested == []  # nothing re-sent: the batch was already accepted
    assert resumed.waited == [[it["id"] for it in expected_items]]  # ... but every id was polled
    assert rec["corpus"]["documents_read"] == 0
    _assert_converged(rec, run_dir, state_path, db_path, [first, resumed], expected_items, 1)


def test_crash_mid_poll_on_batch_2_resumes_to_the_same_accounting(tmp_path, cfg, db_path, expected_items,
                                                                  fake_client):
    # reference: the uninterrupted accounting
    ref_dir = tmp_path / "ref"
    ref = ingest.run_ingest(cfg, db_path, ref_dir / "s.json", batch_size=10, batch_sleep=0, api_key="k",
                            provision=False, wait_window=1)
    ref_client = fake_client.instances[-1]
    batches = len(ref_client.ingested)
    assert batches >= 3
    fake_client.instances.clear()

    # crash while polling batch 2: batch 1 settled, batches 2 and 3 sent but unresolved, the rest unsent
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    fake_client.fail_wait_on_call = 2
    with pytest.raises(ConnectionError):
        ingest.run_ingest(cfg, db_path, state_path, batch_size=10, batch_sleep=0, api_key="k", provision=False,
                          wait_window=1)
    crashed = fake_client.instances[-1]
    assert len(crashed.ingested) == 3 and len(crashed.waited) == 1
    state = json.loads(state_path.read_text())
    assert [e["state"] for e in state["journal"]] == ["settled", "sent", "sent"]
    assert state["statuses"]["pending"] == len(crashed.ingested[1]) + len(crashed.ingested[2])

    fake_client.fail_wait_on_call = None
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=10, batch_sleep=0, api_key="k",
                            provision=False, wait_window=1)
    resumed = fake_client.instances[-1]
    # the resumed run first re-polled exactly the two unresolved batches, then sent only what was left
    assert resumed.waited[:2] == [[it["id"] for it in crashed.ingested[1]], [it["id"] for it in crashed.ingested[2]]]
    assert len(resumed.ingested) == batches - 3
    assert rec["corpus"]["documents_read"] < 5

    _assert_converged(rec, run_dir, state_path, db_path, [crashed, resumed], expected_items, batches)
    assert rec["readiness"] == ref["readiness"] and rec["statuses"] == ref["statuses"]
    assert sorted(_status_log(run_dir), key=lambda r: r["id"]) == sorted(_status_log(ref_dir), key=lambda r: r["id"])


def test_crash_after_journal_entry_before_send_resends_that_batch(tmp_path, cfg, db_path, expected_items,
                                                                  fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    fake_client.fail_send_on_call = 2
    with pytest.raises(ConnectionError):
        ingest.run_ingest(cfg, db_path, state_path, batch_size=10, batch_sleep=0, api_key="k", provision=False,
                          wait_window=1)
    crashed = fake_client.instances[-1]
    assert len(crashed.ingested) == 1 and len(crashed.aborted) == 1
    lost = crashed.aborted[0]
    state = json.loads(state_path.read_text())
    assert [e["state"] for e in state["journal"]] == ["sent", "sending"]
    assert state["journal"][1]["ids"] == [it["id"] for it in lost]
    assert state["items_sent"] == len(crashed.ingested[0])  # a "sending" batch is not counted as sent
    assert state["last_doc_id"] == state["journal"][1]["docs"]["last"]  # ... even though the cursor moved past it

    fake_client.fail_send_on_call = None
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=10, batch_sleep=0, api_key="k",
                            provision=False, wait_window=1)
    resumed = fake_client.instances[-1]
    # the stuck batch was re-sent first, item for item, before anything new
    assert resumed.ingested[0] == lost
    all_batches = [b for c in (crashed, resumed) for b in c.ingested + c.aborted]
    assert sum(1 for b in all_batches if b == lost) == 2  # seen twice: once aborted, once re-sent
    assert resumed.aborted == []
    batches = len(crashed.ingested) + len(resumed.ingested)
    _assert_converged(rec, run_dir, state_path, db_path, [crashed, resumed], expected_items, batches)


def test_run_ending_with_pending_ids_raises_incomplete_and_resume_finishes_it(tmp_path, cfg, db_path,
                                                                             expected_items, fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    stuck = {it["id"] for it in expected_items[3:6]}
    fake_client.pending_ids = stuck
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        ingest.run_ingest(cfg, db_path, state_path, batch_size=10, batch_sleep=0, api_key="k", provision=False)
    first = fake_client.instances[-1]
    batches = len(first.ingested)
    exc = exc_info.value
    n = len(expected_items)
    assert exc.readiness == {"items_sent": n, "items_settled": n - batches - 3, "items_errored": batches,
                             "items_pending": 3, "unresolved_batches": 1, "ready": False}
    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["ready"] is False and on_disk["readiness"] == exc.readiness
    assert on_disk["batches"] == {"sent": batches, "unresolved": 1}
    assert {r["id"] for r in _status_log(run_dir)}.isdisjoint(stuck)  # pending ids are not logged yet
    state = json.loads(state_path.read_text())
    [open_entry] = [e for e in state["journal"] if e["state"] == "sent"]
    assert set(open_entry["ids"]) - set(open_entry["statuses"]) == stuck

    fake_client.pending_ids = set()
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, batch_size=10, batch_sleep=0, api_key="k",
                            provision=False)
    resumed = fake_client.instances[-1]
    assert resumed.ingested == []
    assert [set(ids) for ids in resumed.waited] == [stuck]  # only the pending ids are re-polled
    # the re-poll's first id "errors" in the fake, so one more errored than the reference accounting
    assert rec["ready"] is True
    assert rec["readiness"]["items_pending"] == 0 and rec["readiness"]["unresolved_batches"] == 0
    assert rec["readiness"]["items_settled"] + rec["readiness"]["items_errored"] == n == rec["readiness"]["items_sent"]
    assert sorted(_waited_ids([first, resumed])) == sorted([it["id"] for it in expected_items] + sorted(stuck))
    log = _status_log(run_dir)
    assert len(log) == n and len({r["id"] for r in log}) == n


# ---------------------------------------------------------------------------
# resume identity
# ---------------------------------------------------------------------------
def test_resume_continues_after_checkpoint(tmp_path, cfg, db_path, expected_items, fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    ingest.run_ingest(cfg, db_path, state_path, limit=2, batch_size=1000, batch_sleep=0, api_key="k",
                      provision=False)
    first = fake_client.instances[-1]
    first_ids = {it["id"] for b in first.ingested for it in b}
    assert json.loads(state_path.read_text())["docs_processed"] == 2

    # ``limit`` is part of the resume identity, so each resumed run ingests the next ``limit`` docs
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, limit=2, batch_size=1000, batch_sleep=0,
                            api_key="k", provision=False)
    second = fake_client.instances[-1]
    second_ids = {it["id"] for b in second.ingested for it in b}
    assert first_ids.isdisjoint(second_ids)
    assert rec["corpus"]["documents_read"] == 2 and rec["corpus"]["documents_processed_total"] == 4

    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, limit=2, batch_size=1000, batch_sleep=0,
                            api_key="k", provision=False)
    third_ids = {it["id"] for b in fake_client.instances[-1].ingested for it in b}
    assert first_ids | second_ids | third_ids == {it["id"] for it in expected_items}
    assert rec["corpus"]["documents_read"] == 1 and rec["corpus"]["documents_processed_total"] == 5
    assert rec["items"]["sent"] == len(expected_items) and rec["ready"] is True

    # a further resume finds nothing left, sends nothing and is still ready
    rec = ingest.run_ingest(cfg, db_path, state_path, resume=True, limit=2, batch_size=1000, batch_sleep=0,
                            api_key="k", provision=False)
    assert fake_client.instances[-1].ingested == []
    assert rec["corpus"]["documents_read"] == 0 and rec["ready"] is True


def test_resume_refuses_a_checkpoint_from_another_target(tmp_path, cfg, db_path, fake_client):
    state_path = tmp_path / "s.json"
    ingest.run_ingest(cfg, db_path, state_path, limit=1, batch_sleep=0, api_key="k", provision=False)
    other = RunConfig(name="other")
    other.hydradb.database, other.hydradb.collection = cfg.hydradb.database, "another"
    with pytest.raises(RuntimeError, match="refusing to resume"):
        ingest.run_ingest(other, db_path, state_path, resume=True, limit=1, api_key="k", provision=False)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, limit=1, infer=True, api_key="k", provision=False)


def test_resume_refuses_a_different_source_scope_or_corpus(tmp_path, cfg, db_path, fake_client):
    state_path = tmp_path / "run" / "s.json"
    ingest.run_ingest(cfg, db_path, state_path, source_types=["slack", "gmail"], batch_sleep=0, api_key="k",
                      provision=False)
    assert json.loads(state_path.read_text())["source_types"] == ["gmail", "slack"]
    with pytest.raises(RuntimeError, match="refusing to resume.*source_types/limit"):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["slack"], api_key="k",
                          provision=False)
    with pytest.raises(RuntimeError, match="refusing to resume.*source_types/limit"):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["slack", "gmail"], limit=3,
                          api_key="k", provision=False)
    # same scope in a different order is the same scope
    ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["gmail", "slack"], batch_sleep=0,
                      api_key="k", provision=False)

    # a different corpus (here: a subset build) is refused too
    other_db = tmp_path / "other.sqlite"
    corpus.build_from_repo(ERB_REPO, other_db, limit=3, quiet=True)
    with pytest.raises(RuntimeError, match="refusing to resume.*corpus"):
        ingest.run_ingest(cfg, other_db, state_path, resume=True, source_types=["slack", "gmail"],
                          api_key="k", provision=False)

    # a pre-journal checkpoint cannot be reconciled
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"database": cfg.hydradb.database, "collection": cfg.hydradb.collection,
                                  "infer": False, "last_doc_id": None}))
    with pytest.raises(RuntimeError, match="predates the batch journal"):
        ingest.run_ingest(cfg, db_path, legacy, resume=True, api_key="k", provision=False)


def test_missing_api_key_is_a_clear_error(tmp_path, cfg, db_path, fake_client, monkeypatch):
    monkeypatch.delenv("HYDRADB_API_KEY", raising=False)
    monkeypatch.setattr(ingest, "load_dotenv", lambda *a, **kw: False)  # ignore any developer .env
    with pytest.raises(RuntimeError, match="HYDRADB_API_KEY"):
        ingest.run_ingest(cfg, db_path, tmp_path / "s.json", provision=False)
