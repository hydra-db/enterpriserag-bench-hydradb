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
    * ``pending_ids``: ids reported as never settling (status-poll timeout);
    * ``errored_ids``: ids reported ``errored`` every time they are polled;
    * ``flaky_ids``: ids reported ``errored`` the first time they are polled
      and ``completed`` afterwards (a retry succeeds);
    * ``error_all``: every id errors (the reviewer's all-failed batch).
    """

    instances: list["FakeAdminClient"] = []
    fail_send_on_call: int | None = None
    fail_wait_on_call: int | None = None
    pending_ids: set[str] = set()
    errored_ids: set[str] = set()
    flaky_ids: set[str] = set()
    error_all: bool = False
    polled: list[str] = []  # every id ever polled, across instances

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
        out = {}
        for i in source_ids:
            if i in FakeAdminClient.pending_ids:
                out[i] = "pending"
            elif FakeAdminClient.error_all or i in FakeAdminClient.errored_ids:
                out[i] = "errored"
            elif i in FakeAdminClient.flaky_ids and i not in FakeAdminClient.polled:
                out[i] = "errored"
            else:
                out[i] = "completed"
        FakeAdminClient.polled.extend(source_ids)
        return out

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
    FakeAdminClient.errored_ids = set()
    FakeAdminClient.flaky_ids = set()
    FakeAdminClient.error_all = False
    FakeAdminClient.polled = []
    monkeypatch.setattr(ingest, "HydraDBAdminClient", FakeAdminClient)
    return FakeAdminClient


def _sent_ids(clients) -> list[str]:
    return [it["id"] for c in clients for batch in c.ingested for it in batch]


def _waited_ids(clients) -> list[str]:
    return [i for c in clients for ids in c.waited for i in ids]


def _status_log(run_dir: Path) -> list[dict]:
    """The raw log lines (``ingest.read_status_log`` is the deduplicating reader)."""
    return [json.loads(line) for line in (run_dir / "ingest_status.jsonl").read_text().splitlines()]


def _small_corpus(path: Path, docs: list[tuple[str, str]], revision: str = "a") -> Path:
    """A ``documents.sqlite`` of one-item ``google_drive`` docs (item id == doc id) with meta identity."""
    corpus.build_documents_db(path, [(d, "google_drive", f"Title {d}", body, len(body)) for d, body in docs],
                              quiet=True, identity={"source": "repo", "revision": revision})
    return path


def _readiness(sent, settled, errored, pending, unresolved) -> dict:
    resolved = sent == settled + errored and pending == 0 and unresolved == 0
    return {"items_sent": sent, "items_settled": settled, "items_errored": errored, "items_pending": pending,
            "unresolved_batches": unresolved, "resolved": resolved,
            "ready": resolved and errored == 0 and sent == settled}


def _assert_converged(rec: dict, run_dir: Path, state_path: Path, db_path: Path, clients, expected_items,
                      expected_batches):
    """The accounting a run must show once every item is sent and settled, however it got there."""
    n = len(expected_items)
    assert rec["ready"] is True and rec["resolved"] is True
    assert rec["readiness"] == _readiness(n, n, 0, 0, 0)
    assert rec["statuses"] == {"settled": n, "errored": 0, "pending": 0}
    assert rec["failed_ids"] == {} and rec["accepted_failures"] == []
    assert rec["items"]["sent"] == n and rec["batches"]["sent"] == expected_batches
    # no document skipped, no item sent twice
    assert sorted(_sent_ids(clients)) == sorted(it["id"] for it in expected_items)
    # every id polled exactly once, and logged exactly once with a terminal status
    assert sorted(_waited_ids(clients)) == sorted(it["id"] for it in expected_items)
    log = _status_log(run_dir)
    assert len(log) == n and len({r["id"] for r in log}) == n
    assert all(r["status"] == "completed" and r["attempt"] == 1 for r in log)
    assert ingest.read_status_log(run_dir / "ingest_status.jsonl") == log
    state = json.loads(state_path.read_text())
    assert state["items_sent"] == n and state["batches_sent"] == expected_batches
    assert state["failed_ids"] == {}
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
    assert rec["ready"] is False and rec["resolved"] is False and rec["readiness"]["items_sent"] == 0  # nothing sent

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
    assert state["corpus"]["source"] == "repo" and state["converter"]["sha256"] == ingest.converter_sha256()
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
    assert exc.readiness == _readiness(n, 0, 0, n, batches)
    assert (exc.items_sent, exc.items_pending, exc.unresolved_batches) == (n, n, batches)
    assert exc.ready is False and exc.resolved is False and exc.reason == "unresolved"
    assert "pending" in str(exc) and "--resume" in str(exc) and "--retry-failed" not in str(exc)
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
    assert exc.readiness == _readiness(n, n - 3, 0, 3, 1)
    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["ready"] is False and on_disk["readiness"] == exc.readiness
    assert on_disk["batches"] == {"sent": batches, "retried": 0, "unresolved": 1}
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
    assert sorted(_waited_ids([first, resumed])) == sorted([it["id"] for it in expected_items] + sorted(stuck))
    assert rec["ready"] is True and rec["resolved"] is True
    assert rec["readiness"] == _readiness(n, n, 0, 0, 0) and rec["failed_ids"] == {}
    assert rec["batches"] == {"sent": batches, "retried": 0, "unresolved": 0}
    log = _status_log(run_dir)
    assert len(log) == n and len({r["id"] for r in log}) == n  # the stuck ids were logged once, on resume
    assert ingest.read_status_log(run_dir / "ingest_status.jsonl") == log


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
    with pytest.raises(RuntimeError, match="refusing to resume.*source_types="):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["slack"], api_key="k",
                          provision=False)
    with pytest.raises(RuntimeError, match="refusing to resume.*limit="):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["slack", "gmail"], limit=3,
                          api_key="k", provision=False)
    # same scope in a different order is the same scope
    ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["gmail", "slack"], batch_sleep=0,
                      api_key="k", provision=False)

    # a different corpus (here: a subset build) is refused too
    other_db = tmp_path / "other.sqlite"
    corpus.build_from_repo(ERB_REPO, other_db, limit=3, quiet=True)
    with pytest.raises(RuntimeError, match="refusing to resume.*corpus.documents="):
        ingest.run_ingest(cfg, other_db, state_path, resume=True, source_types=["slack", "gmail"],
                          api_key="k", provision=False)
    # ... and so is a checkpoint written by a different converter
    state = json.loads(state_path.read_text())
    state["converter"]["sha256"] = "0" * 64
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="refusing to resume.*converter.sha256="):
        ingest.run_ingest(cfg, db_path, state_path, resume=True, source_types=["slack", "gmail"],
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


# ---------------------------------------------------------------------------
# failed items (reviewer finding P1-1): resolved is not ready
# ---------------------------------------------------------------------------
def _run(cfg, db, state_path, **kw):
    kw.setdefault("batch_sleep", 0)
    kw.setdefault("provision", False)
    kw.setdefault("api_key", "k")
    return ingest.run_ingest(cfg, db, state_path, **kw)


def test_reviewer_fixture_all_errored_batch_is_resolved_but_not_ready(tmp_path, cfg, fake_client):
    """The reviewer's P1-1 fixture: one doc, its only item errors. Was: ready=True and a normal return."""
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "ORIGINAL BODY")], revision="revision-A")
    run_dir = tmp_path / "all-errored"
    state_path = run_dir / "ingest_state.json"
    fake_client.error_all = True
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db, state_path)
    exc = exc_info.value
    assert exc.readiness == _readiness(1, 0, 1, 0, 0)
    assert exc.ready is False and exc.resolved is True and exc.reason == "failed"
    assert exc.failed_ids == {"d1": {"batch": 1, "status": "errored", "attempts": 1}}
    assert "1 item(s) failed" in str(exc) and "--retry-failed" in str(exc)
    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["ready"] is False and on_disk["resolved"] is True
    assert on_disk["failed_ids"] == exc.failed_ids and on_disk["accepted_failures"] == []
    state = json.loads(state_path.read_text())
    assert state["journal"][0]["state"] == "settled" and state["journal"][0]["counts"] == {"settled": 0, "errored": 1}
    assert state["failed_ids"] == exc.failed_ids  # compaction keeps the failed-id record
    assert _status_log(run_dir) == [{"batch": 1, "id": "d1", "status": "errored", "attempt": 1}]
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert stages[-1]["ingest"]["ready"] is False and stages[-1]["ingest"]["resolved"] is True


def test_all_failed_batch_on_the_repo_corpus_raises_incomplete(tmp_path, cfg, db_path, expected_items, fake_client):
    state_path = tmp_path / "run" / "s.json"
    fake_client.error_all = True
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db_path, state_path, batch_size=10)
    exc = exc_info.value
    n = len(expected_items)
    assert exc.readiness == _readiness(n, 0, n, 0, 0)
    assert exc.ready is False and exc.resolved is True
    assert set(exc.failed_ids) == {it["id"] for it in expected_items}
    assert all(v["status"] == "errored" and v["attempts"] == 1 for v in exc.failed_ids.values())
    # a plain --resume finds nothing unresolved, sends nothing, and is still not ready
    with pytest.raises(ingest.IngestIncomplete) as again:
        _run(cfg, db_path, state_path, resume=True, batch_size=10)
    assert fake_client.instances[-1].ingested == [] and fake_client.instances[-1].waited == []
    assert again.value.readiness == exc.readiness and again.value.reason == "failed"


def test_mixed_batch_counts_are_exact(tmp_path, cfg, fake_client):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one"), ("d2", "two"), ("d3", "three")])
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    fake_client.errored_ids = {"d2"}
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db, state_path, batch_size=10)
    exc = exc_info.value
    assert exc.readiness == _readiness(3, 2, 1, 0, 0)
    assert (exc.items_sent, exc.items_settled, exc.items_errored, exc.items_pending) == (3, 2, 1, 0)
    assert exc.manifest["statuses"] == {"settled": 2, "errored": 1, "pending": 0}
    assert exc.failed_ids == {"d2": {"batch": 1, "status": "errored", "attempts": 1}}
    assert exc.manifest["batches"] == {"sent": 1, "retried": 0, "unresolved": 0}
    assert "1 item(s) failed" in str(exc) and "--retry-failed" in str(exc)
    [client] = fake_client.instances
    assert [it["id"] for b in client.ingested for it in b] == ["d1", "d2", "d3"]
    assert sorted(_status_log(run_dir), key=lambda r: r["id"]) == [
        {"batch": 1, "id": "d1", "status": "completed", "attempt": 1},
        {"batch": 1, "id": "d2", "status": "errored", "attempt": 1},
        {"batch": 1, "id": "d3", "status": "completed", "attempt": 1}]


def test_retry_failed_resends_exactly_the_failed_items_and_becomes_ready(tmp_path, cfg, db_path, expected_items,
                                                                          fake_client):
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    # every item of the slack thread plus one comment of the jira ticket fail on their first attempt
    slack = [it["id"] for it in expected_items if it["id"].split("_m")[0] != it["id"]]
    jira_comment = next(it["id"] for it in expected_items if "_c0" in it["id"])
    flaky = set(slack) | {jira_comment}
    assert len(flaky) >= 3
    fake_client.flaky_ids = set(flaky)
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db_path, state_path, batch_size=10)
    first = fake_client.instances[-1]
    n, batches = len(expected_items), len(first.ingested)
    assert set(exc_info.value.failed_ids) == flaky
    assert exc_info.value.readiness == _readiness(n, n - len(flaky), len(flaky), 0, 0)

    rec = _run(cfg, db_path, state_path, resume=True, retry_failed=True, batch_size=10)
    retried = fake_client.instances[-1]
    # exactly the failed items were re-sent (none of their siblings that had settled), each once
    assert sorted(it["id"] for b in retried.ingested for it in b) == sorted(flaky)
    assert sorted(_waited_ids([retried])) == sorted(flaky)
    assert all(it in expected_items for b in retried.ingested for it in b)  # item for item what convert produces
    # ... and the accounting converged without double counting
    assert rec["ready"] is True and rec["resolved"] is True
    assert rec["readiness"] == _readiness(n, n, 0, 0, 0)
    assert rec["failed_ids"] == {} and rec["accepted_failures"] == []
    assert rec["items"]["sent"] == n
    assert rec["batches"]["retried"] == len(retried.ingested) >= 1
    assert rec["batches"]["sent"] == batches + rec["batches"]["retried"]
    assert rec["retry"] == {"batches": rec["batches"]["retried"], "items": len(flaky), "recovered": len(flaky)}
    assert rec["corpus"]["documents_read"] == 0
    state = json.loads(state_path.read_text())
    assert state["failed_ids"] == {} and state["last_doc_id"] == max(d["doc_id"] for d in corpus.iter_documents(db_path))
    retry_entries = [e for e in state["journal"] if e.get("retry")]
    assert len(retry_entries) == rec["batches"]["retried"]
    assert all(e["state"] == "settled" and e["counts"]["errored"] == 0 and "doc_ids" in e for e in retry_entries)
    assert sum(e["n_items"] for e in retry_entries) == len(flaky)
    # the status log is the attempt history: one row per attempt, both attempts of every flaky id present
    log = ingest.read_status_log(run_dir / "ingest_status.jsonl")
    assert len(log) == n + len(flaky) == len(_status_log(run_dir))
    by_id = {}
    for r in log:
        by_id.setdefault(r["id"], []).append((r["attempt"], r["status"]))
    assert all(by_id[i] == [(1, "errored"), (2, "completed")] for i in flaky)
    assert all(by_id[i] == [(1, "completed")] for i in by_id if i not in flaky)


def test_retry_failed_keeps_ids_that_fail_again_with_attempts_bumped(tmp_path, cfg, fake_client):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one"), ("d2", "two"), ("d3", "three")])
    state_path = tmp_path / "run" / "s.json"
    fake_client.errored_ids = {"d2"}
    fake_client.flaky_ids = {"d3"}
    with pytest.raises(ingest.IngestIncomplete) as first:
        _run(cfg, db, state_path, batch_size=10)
    assert first.value.readiness == _readiness(3, 1, 2, 0, 0)
    # retry in the same run as the first attempt also works (no resume needed for this run's own failures)
    with pytest.raises(ingest.IngestIncomplete) as second:
        _run(cfg, db, state_path, resume=True, retry_failed=True, batch_size=10)
    exc = second.value
    assert [it["id"] for b in fake_client.instances[-1].ingested for it in b] == ["d2", "d3"]
    assert exc.readiness == _readiness(3, 2, 1, 0, 0)
    assert exc.failed_ids == {"d2": {"batch": 2, "status": "errored", "attempts": 2}}
    assert exc.manifest["retry"] == {"batches": 1, "items": 2, "recovered": 1}
    assert "1 item(s) failed" in str(exc)
    # a further retry re-sends only d2 again; attempts keep counting
    with pytest.raises(ingest.IngestIncomplete) as third:
        _run(cfg, db, state_path, resume=True, retry_failed=True, batch_size=10)
    assert [it["id"] for b in fake_client.instances[-1].ingested for it in b] == ["d2"]
    assert third.value.failed_ids == {"d2": {"batch": 3, "status": "errored", "attempts": 3}}
    assert third.value.readiness == _readiness(3, 2, 1, 0, 0)
    assert [r["attempt"] for r in ingest.read_status_log(tmp_path / "run" / "ingest_status.jsonl") if r["id"] == "d2"] == [1, 2, 3]


def test_retry_failed_from_scratch_retries_this_runs_failures_once(tmp_path, cfg, fake_client):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one"), ("d2", "two")])
    state_path = tmp_path / "run" / "s.json"
    fake_client.flaky_ids = {"d2"}
    rec = _run(cfg, db, state_path, retry_failed=True, batch_size=10)
    [client] = fake_client.instances
    assert [[it["id"] for it in b] for b in client.ingested] == [["d1", "d2"], ["d2"]]
    assert rec["ready"] is True and rec["readiness"] == _readiness(2, 2, 0, 0, 0)
    assert rec["retry"] == {"batches": 1, "items": 1, "recovered": 1}


def test_accept_failures_returns_normally_with_ready_false_recorded(tmp_path, cfg, fake_client):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one"), ("d2", "two"), ("d3", "three")])
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    fake_client.errored_ids = {"d3"}
    rec = _run(cfg, db, state_path, accept_failures=True, batch_size=10)
    assert rec["ready"] is False and rec["resolved"] is True
    assert rec["readiness"] == _readiness(3, 2, 1, 0, 0)
    assert rec["accept_failures"] is True and rec["accepted_failures"] == ["d3"]
    assert rec["failed_ids"] == {"d3": {"batch": 1, "status": "errored", "attempts": 1}}
    on_disk = json.loads((run_dir / "ingest_manifest.json").read_text())
    assert on_disk["ready"] is False and on_disk["accepted_failures"] == ["d3"]
    stages = json.loads((run_dir / "manifest.json").read_text())["stages"]
    assert stages[-1]["ingest"]["ready"] is False and stages[-1]["ingest"]["accepted_failures"] == ["d3"]

    # accept_failures never covers unresolved work
    fake_client.pending_ids = {"d1"}
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db, tmp_path / "run2" / "s.json", accept_failures=True, batch_size=10)
    assert exc_info.value.reason == "unresolved" and exc_info.value.manifest["accepted_failures"] == []


# ---------------------------------------------------------------------------
# resume identity (reviewer finding H1): same ids, different content
# ---------------------------------------------------------------------------
def test_resume_refuses_a_corpus_with_the_same_ids_but_another_revision(tmp_path, cfg, fake_client):
    original = _small_corpus(tmp_path / "docs.sqlite", [("d1", "ORIGINAL BODY")], revision="a")
    alternate = _small_corpus(tmp_path / "alternate.sqlite", [("d1", "DIFFERENT BODY")], revision="b")
    assert ingest.corpus_identity(original)["doc_ids_sha256"] == ingest.corpus_identity(alternate)["doc_ids_sha256"]
    run_dir = tmp_path / "corpus-swap"
    state_path = run_dir / "ingest_state.json"
    fake_client.fail_send_on_call = 1
    with pytest.raises(ConnectionError):
        _run(cfg, original, state_path)
    assert json.loads(state_path.read_text())["journal"][0]["state"] == "sending"

    fake_client.fail_send_on_call = None
    with pytest.raises(RuntimeError, match="refusing to resume.*corpus.revision='a'.*corpus.revision='b'"):
        _run(cfg, alternate, state_path, resume=True)
    assert fake_client.instances[-1].ingested == []  # the replacement body was never sent
    assert not (run_dir / "ingest_manifest.json").exists()

    # the original corpus resumes fine and sends the original body
    rec = _run(cfg, original, state_path, resume=True)
    [[item]] = fake_client.instances[-1].ingested
    assert "ORIGINAL BODY" in json.dumps(item) and rec["ready"] is True
    assert rec["corpus"]["source"] == "repo" and rec["corpus"]["revision"] == "a"
    assert rec["converter"]["sha256"] == ingest.converter_sha256()


# ---------------------------------------------------------------------------
# status log (reviewer finding H9): crash between the log append and the checkpoint
# ---------------------------------------------------------------------------
def test_crash_after_status_append_before_checkpoint_leaves_one_row_per_batch_and_id(tmp_path, cfg, fake_client,
                                                                                     monkeypatch):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one")])
    run_dir = tmp_path / "status-journal-crash"
    state_path = run_dir / "ingest_state.json"
    log_path = run_dir / "ingest_status.jsonl"
    original_write = ingest._atomic_write_json

    def crash_write(path, payload):
        if any(e["state"] == "settled" for e in payload["journal"]):
            raise OSError("crash after status log append before checkpoint")
        return original_write(path, payload)

    monkeypatch.setattr(ingest, "_atomic_write_json", crash_write)
    with pytest.raises(OSError):
        _run(cfg, db, state_path)
    assert _status_log(run_dir) == [{"batch": 1, "id": "d1", "status": "completed", "attempt": 1}]
    state = json.loads(state_path.read_text())
    assert state["journal"][0]["state"] == "sent" and state["journal"][0]["statuses"] == {}

    monkeypatch.setattr(ingest, "_atomic_write_json", original_write)
    rec = _run(cfg, db, state_path, resume=True)
    resumed = fake_client.instances[-1]
    assert resumed.ingested == [] and resumed.waited == []  # the logged status was adopted, not re-polled
    assert rec["ready"] is True and rec["readiness"] == _readiness(1, 1, 0, 0, 0)
    assert _status_log(run_dir) == [{"batch": 1, "id": "d1", "status": "completed", "attempt": 1}]
    assert ingest.read_status_log(log_path) == _status_log(run_dir)

    # even a log that did end up with a duplicate reads as one row per (batch, id), the last one winning
    with open(log_path, "a") as f:
        f.write(json.dumps({"batch": 1, "id": "d1", "status": "errored", "attempt": 1}) + "\n")
        f.write(json.dumps({"batch": 2, "id": "d1", "status": "completed", "attempt": 2}) + "\n")
    assert len(_status_log(run_dir)) == 3
    assert ingest.read_status_log(log_path) == [{"batch": 1, "id": "d1", "status": "errored", "attempt": 1},
                                                {"batch": 2, "id": "d1", "status": "completed", "attempt": 2}]


def test_crash_after_status_append_with_a_failed_id_keeps_it_failed_on_resume(tmp_path, cfg, fake_client, monkeypatch):
    db = _small_corpus(tmp_path / "docs.sqlite", [("d1", "one"), ("d2", "two")])
    run_dir = tmp_path / "run"
    state_path = run_dir / "s.json"
    fake_client.errored_ids = {"d2"}
    original_write = ingest._atomic_write_json

    def crash_write(path, payload):
        if any(e["state"] == "settled" for e in payload["journal"]):
            raise OSError("crash")
        return original_write(path, payload)

    monkeypatch.setattr(ingest, "_atomic_write_json", crash_write)
    with pytest.raises(OSError):
        _run(cfg, db, state_path)
    monkeypatch.setattr(ingest, "_atomic_write_json", original_write)
    with pytest.raises(ingest.IngestIncomplete) as exc_info:
        _run(cfg, db, state_path, resume=True)
    assert fake_client.instances[-1].waited == []
    assert exc_info.value.readiness == _readiness(2, 1, 1, 0, 0)
    assert exc_info.value.failed_ids == {"d2": {"batch": 1, "status": "errored", "attempts": 1}}
    assert len(ingest.read_status_log(run_dir / "ingest_status.jsonl")) == 2 == len(_status_log(run_dir))
