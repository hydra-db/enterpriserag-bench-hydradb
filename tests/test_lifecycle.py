"""Stage-transition lifecycle tests: a run must pass its own validator after
every transition (retrieve -> generate -> failure -> retry -> verify -> replay),
with exactly one authoritative context per question and accessible history.
All offline; the model and HydraDB are mocked."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from erb_hydradb import analysis, cli, corpus, generate, hydrate, manifest, validate
from erb_hydradb.config import RunConfig

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "run-2026-09-04"
QUESTIONS = REPO / "tests" / "data" / "questions.jsonl"
FIXTURE_REPO = REPO / "tests" / "data" / "erb_repo"


class FakeQuery:
    """Returns chunks for a document that exists in the fixture corpus."""
    calls = 0

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def query(self, *a, **kw):
        FakeQuery.calls += 1
        return [{"id": f"{DOC_ID}_m0000", "chunk_content": "retrieved chunk"}]


DOC_ID = None


@pytest.fixture(scope="module")
def fixture_store(tmp_path_factory):
    global DOC_ID
    db = tmp_path_factory.mktemp("corpus") / "documents.sqlite"
    corpus.build_from_repo(FIXTURE_REPO, db, quiet=True)
    DOC_ID = next(corpus.iter_documents(db))["doc_id"]
    return hydrate.DocumentStore(db, None)


def _questions(tmp_path, n=1):
    rows = [{"question_id": f"q{i}", "question": f"Q{i}", "question_type": "basic", "expected_doc_ids": [DOC_ID]}
            for i in range(1, n + 1)]
    p = tmp_path / "questions.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p, rows


def _run(cfg, run_dir, questions, qpath, store, **kw):
    with patch("erb_hydradb.hydradb_client.HydraDBQueryClient", FakeQuery):
        return asyncio.run(generate.run_generate(cfg, run_dir, questions, api_key="mock", store=store,
                                                 questions_path=qpath, **kw))


def test_retrieve_generate_fail_retry_verify_replay(tmp_path, fixture_store):
    cfg = RunConfig()
    qpath, questions = _questions(tmp_path)
    run = tmp_path / "run"

    # 1. retrieval only
    rows, inc = _run(cfg, run, questions, qpath, fixture_store, retrieval_only=True)
    assert inc is None and rows[0]["stage"] == "retrieved"
    assert validate.check_sums_and_manifest(run, ("SHA256SUMS",)) == []

    # 2. generation from the saved context, model fails
    with patch.object(generate, "generate_answer", side_effect=RuntimeError("provider down")):
        rows, inc = _run(cfg, run, questions, qpath, fixture_store)
    assert inc is not None and rows[0]["stage"] == "error"

    # 3. retry succeeds: exactly one authoritative context, history keeps both attempts
    with patch.object(generate, "generate_answer", return_value=("REAL ANSWER", {"pass1": True})) as call:
        rows, inc = _run(cfg, run, questions, qpath, fixture_store)
    assert inc is None and rows[0]["stage"] == "answered" and call.call_count == 1
    ctx_rows = validate.read_jsonl_rows(run / "contexts.jsonl.gz")
    assert [r["question_id"] for r in ctx_rows] == ["q1"]
    assert validate.check_context_rows(ctx_rows, {"q1"}) == []
    attempts = validate.read_jsonl_rows(run / generate.ATTEMPTS)
    assert len(attempts) >= 2 and {a["question_id"] for a in attempts} == {"q1"}
    # the model saw the saved retrieval-only context, not an empty rebuild
    assert rows[0]["context_sha256"] == ctx_rows[0]["context_sha256"] == validate.sha256_text(ctx_rows[0]["context"])

    # 4. the completed run passes its own validator (manifest: latest writer wins)
    report = validate.validate_run(run, qpath, required=("SHA256SUMS",), contexts=True, store=fixture_store)
    assert report["ok"], validate.format_report(report)
    assert report["checks"]["context_reconstruction"]["reproduced"] == 1

    # 5. replay of the run's own contexts regenerates without HydraDB
    replay = tmp_path / "replay"
    with patch.object(generate, "generate_answer", return_value=("REPLAYED", {})):
        rows2, inc2 = asyncio.run(generate.run_generate(cfg, replay, questions, api_key=None, store=None,
                                                        questions_path=qpath, from_contexts=run / "contexts.jsonl.gz"))
    assert inc2 is None and rows2[0]["answer"] == "REPLAYED"
    assert validate.validate_run(replay, qpath, required=("SHA256SUMS",))["ok"]


def test_generation_identity_binds_document_store(tmp_path, fixture_store, tmp_path_factory):
    cfg = RunConfig()
    qpath, questions = _questions(tmp_path)
    run = tmp_path / "run"
    with patch.object(generate, "generate_answer", return_value=("A", {})):
        _run(cfg, run, questions, qpath, fixture_store, store_identity={"backend": "sqlite", "sqlite": {"source": "repo", "revision": "a"}})
    with patch.object(generate, "generate_answer", return_value=("B", {})) as call, pytest.raises(SystemExit):
        _run(cfg, run, questions, qpath, fixture_store, store_identity={"backend": "sqlite", "sqlite": {"source": "repo", "revision": "b"}})
    assert call.call_count == 0


def test_finishing_a_retrieved_row_uses_the_saved_context(tmp_path):
    """Chunk-fallback evidence must survive the retrieval-only -> generation transition."""
    cfg = RunConfig()

    class MissingStore:
        backend = "test"
        identity = {"backend": "test"}

        def get(self, d):
            return None

    qpath, questions = _questions(tmp_path)
    run = tmp_path / "run"
    rows, _ = _run(cfg, run, questions, qpath, MissingStore(), retrieval_only=True)
    original_sha = rows[0]["context_sha256"]
    with patch.object(generate, "generate_answer", return_value=("A", {})) as call:
        rows, inc = _run(cfg, run, questions, qpath, MissingStore())
    assert inc is None
    assert "retrieved chunk" in call.call_args.args[1]
    assert rows[0]["context_sha256"] == original_sha


def test_manifest_latest_record_wins(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("v1")
    manifest.write_manifest(tmp_path, "s1", None, [f])
    f.write_text("v2")
    manifest.write_manifest(tmp_path, "s2", None, [f])
    assert validate.check_sums_and_manifest(tmp_path, ("SHA256SUMS",)) == []
    f.write_text("v3")
    manifest.write_sums(tmp_path)
    assert any("hash differs" in p for p in validate.check_sums_and_manifest(tmp_path, ("SHA256SUMS",)))


# ------------------------------------------------------------- recall -----

def test_recall_is_set_based_and_matches_the_evaluator():
    qs = analysis.load_jsonl(QUESTIONS)
    assert len(qs["qst_0413"]["expected_doc_ids"]) == 2 and len(set(qs["qst_0413"]["expected_doc_ids"])) == 1
    ranked, _ = analysis.load_retrieval(ART / "gen_checkpoint.json", None)
    ids = [i for i in qs if qs[i].get("expected_doc_ids")]
    r10 = analysis.recall_at(ranked, qs, ids, 10)
    strict = analysis.load_results(ART / "official_results_strict.json")["aggregate_stats"]
    assert r10["recall_pct"] == strict["average_recall_pct"] == 96.63
    assert r10["invalid_extra"] == strict["average_invalid_extra_docs"]
    rep = analysis.recall_compare(ranked, ranked, QUESTIONS)
    assert all(row["a"] == row["b"] for row in rep["rows"])
    assert next(row for row in rep["rows"] if row["k"] == 10)["a"] == 96.63
    assert not rep["full_coverage_gained"] and not rep["full_coverage_lost"]


def test_load_retrieval_rejects_duplicate_rows(tmp_path):
    _, rows = validate.read_checkpoint(ART / "gen_checkpoint.json")
    (tmp_path / "ck.json").write_text(json.dumps(rows[:2] + rows[:1]))
    with pytest.raises(SystemExit):
        analysis.load_retrieval(tmp_path / "ck.json", None)


# ------------------------------------------------------ artifact semantics -----

def _strict():
    return analysis.load_results(ART / "official_results_strict.json")


def test_nan_and_out_of_range_scores_are_rejected(tmp_path):
    d = _strict()
    d["questions"][0]["completeness_pct"] = math.nan
    p = tmp_path / "nan.json"
    p.write_text(json.dumps(d))
    assert any("finite" in x for x in validate.check_results_file(p, None)[0])
    d = _strict()
    d["questions"][0]["completeness_pct"] = 200
    p = tmp_path / "over.json"
    p.write_text(json.dumps(d))
    assert any("[0, 100]" in x for x in validate.check_results_file(p, None)[0])


def test_exported_answers_must_match_checkpoint_and_retrieval_metrics(tmp_path):
    import shutil
    run = tmp_path / "run"
    shutil.copytree(ART, run)
    rows = validate.read_jsonl_rows(run / "answers.jsonl")
    rows[0]["answer"] = ""
    rows[0]["document_ids"] = []
    (run / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = json.loads((run / "manifest.json").read_text())
    for st in m["stages"]:
        if "answers.jsonl" in st.get("files", {}):
            st["files"]["answers.jsonl"] = manifest.sha256_file(run / "answers.jsonl")
    (run / "manifest.json").write_text(json.dumps(m))
    manifest.write_sums(run)
    report = validate.validate_run(run, QUESTIONS)
    assert report["checks"]["answers_vs_checkpoint"]["status"] == "fail"
    assert report["checks"]["retrieval_metrics"]["status"] == "fail"
    assert not report["ok"]


def test_published_run_retrieval_metrics_follow_from_submissions():
    assert validate.check_retrieval_metrics(ART, QUESTIONS) == []


def test_corrections_audit_detects_tampered_after_state(tmp_path):
    import shutil
    run = tmp_path / "run"
    run.mkdir()
    for name in ("official_results_protocol.json", "answers.jsonl"):
        shutil.copy2(ART / name, run / name)
    records = validate.read_jsonl_rows(ART / "corrections.jsonl")
    records[0] = copy.deepcopy(records[0])
    records[0]["after"]["expected_doc_ids"] = ["dsid_not_in_submission"]
    records[0]["doc_set_changed"] = False
    records[0]["gold_answer_changed"] = False
    (run / "corrections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    assert cli.main(["audit-corrections", "--run-dir", str(run), "--questions", str(QUESTIONS)]) == 1
