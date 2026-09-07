"""Regression tests for every false-success and data-loss case found in review.

Each test reproduces an adversarial input from the 2026-09-07 public-release
review and asserts the harness now fails loudly. All offline; external calls
are mocked.
"""

from __future__ import annotations

import asyncio
import copy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from erb_hydradb import analysis, cli, generate, hydrate, manifest, validate
from erb_hydradb.config import RunConfig

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "run-2026-09-04"
QUESTIONS = REPO / "tests" / "data" / "questions.jsonl"


def _results():
    with open(ART / "official_results_strict.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------ verify / stats -----

def test_verify_rejects_empty_run_dir(tmp_path):
    manifest.write_sums(tmp_path)  # an empty SHA256SUMS
    assert cli.main(["verify", "--run-dir", str(tmp_path)]) == 1


def test_stats_reports_tampered_category_statistic(tmp_path):
    data = _results()
    data["question_type_stats"]["basic"]["combined_correctness_completeness_score"] = -999
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    problems, _ = validate.check_results_file(p, None)
    assert any("question_type_stats.basic.combined_correctness_completeness_score" in x for x in problems)
    assert cli.main(["stats", "--results", str(p)]) == 1


def test_stats_reports_missing_aggregate_fields(tmp_path):
    data = _results()
    data["aggregate_stats"] = {}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    problems, _ = validate.check_results_file(p, None)
    assert any("aggregate_stats." in x and "missing" in x for x in problems)


def test_compare_rejects_duplicate_raw_rows(tmp_path):
    data = _results()
    extra = copy.deepcopy(next(r for r in data["questions"] if r["answer_correct"] and r["completeness_pct"] == 100))
    extra["answer_correct"] = False
    data["questions"].append(extra)
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        analysis.compare(ART / "official_results_strict.json", p, QUESTIONS, expect_n=500)


def test_compare_rejects_wrong_expect_n():
    with pytest.raises(SystemExit):
        analysis.compare(ART / "official_results_strict.json", ART / "official_results_strict.json", QUESTIONS, expect_n=999)


def test_verify_contexts_rejects_tampered_text_with_stale_hash(tmp_path):
    with gzip.open(ART / "contexts.jsonl.gz", "rt", encoding="utf-8") as f:
        c = json.loads(next(f))
    _, ck = validate.read_checkpoint(ART / "gen_checkpoint.json")
    (tmp_path / "gen_checkpoint.json").write_text(json.dumps([r for r in ck if r["question_id"] == c["question_id"]]))
    c["context"] = "THIS IS NOT THE PUBLISHED CONTEXT"
    with gzip.open(tmp_path / "contexts.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(c) + "\n")
    manifest.write_sums(tmp_path)
    report = validate.validate_run(tmp_path, None, required=("SHA256SUMS",), contexts=False)
    assert report["checks"]["contexts"]["status"] == "fail"
    assert any("does not match the actual text" in p for p in report["checks"]["contexts"]["problems"])


def test_verify_contexts_rejects_zero_contexts(tmp_path):
    _, ck = validate.read_checkpoint(ART / "gen_checkpoint.json")
    (tmp_path / "gen_checkpoint.json").write_text(json.dumps(ck[:1]))
    with gzip.open(tmp_path / "contexts.jsonl.gz", "wt", encoding="utf-8"):
        pass
    manifest.write_sums(tmp_path)
    report = validate.validate_run(tmp_path, None, required=("SHA256SUMS",), contexts=False)
    assert report["checks"]["contexts"]["status"] == "fail"


def test_verify_contexts_requested_but_no_corpus_is_incomplete_not_verified(tmp_path):
    report = validate.validate_run(ART, QUESTIONS, contexts=True, store=None)
    assert report["checks"]["context_reconstruction"]["status"] == "incomplete"
    assert report["ok"] is False


def test_verify_detects_manifest_inner_hash_mismatch(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "manifest.json").write_text(json.dumps({"stages": [{"stage": "s", "files": {"a.txt": "0" * 64}}]}))
    manifest.write_sums(tmp_path)
    problems = validate.check_sums_and_manifest(tmp_path, ("SHA256SUMS",))
    assert any("hash differs" in p for p in problems)


def test_published_run_passes_full_validation():
    report = validate.validate_run(ART, QUESTIONS)
    assert report["ok"], validate.format_report(report)


# ----------------------------------------------------------- generate -----

def _fake_context(text="True context"):
    return hydrate.build_context(["d1"], [{"id": "d1", "chunk_content": text}],
                                 type("S", (), {"get": lambda self, d: None, "backend": "test", "identity": {}})(), 12, 240000)


def _write_contexts(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _run(cfg, run_dir, questions, contexts, **kw):
    return asyncio.run(generate.run_generate(cfg, run_dir, questions, api_key=None, store=None,
                                             from_contexts=contexts, **kw))


def test_all_model_failures_are_incomplete_and_cli_exits_2(tmp_path):
    cfg = RunConfig()
    ctx = _fake_context()
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": "q1", **ctx, "retrieved_doc_ids": ["d1"]}])
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text(json.dumps({"question_id": "q1", "question": "Q1"}) + "\n")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.save(cfg_path)
    with patch("erb_hydradb.llm.provider_available", return_value=True), \
         patch.object(generate, "generate_answer", side_effect=RuntimeError("provider down")):
        rc = cli.main(["generate", "--allow-unpinned", "--config", str(cfg_path), "--questions", str(qpath),
                       "--run-dir", str(tmp_path / "run"), "--from-contexts", str(contexts)])
    assert rc == cli.EXIT_INCOMPLETE
    m = json.loads((tmp_path / "run" / "manifest.json").read_text())["stages"][-1]
    assert m["complete"] is False and m["errors"] == 1
    with patch("erb_hydradb.llm.provider_available", return_value=True), \
         patch.object(generate, "generate_answer", side_effect=RuntimeError("provider down")):
        rc = cli.main(["generate", "--allow-unpinned", "--allow-partial", "--config", str(cfg_path), "--questions",
                       str(qpath), "--run-dir", str(tmp_path / "run2"), "--from-contexts", str(contexts)])
    assert rc == 0


def test_retrieval_only_then_generation_actually_generates(tmp_path):
    cfg = RunConfig()
    ctx = _fake_context()
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": "q1", **ctx, "retrieved_doc_ids": ["d1"]}])
    questions = [{"question_id": "q1", "question": "Q1"}]
    run = tmp_path / "run"
    with patch.object(generate, "generate_answer", return_value=("REAL", {"pass1": True})) as call:
        rows, inc = _run(cfg, run, questions, contexts, retrieval_only=True)
        assert inc is None and rows[0]["stage"] == "retrieved" and call.call_count == 0
        rows, inc = _run(cfg, run, questions, contexts)
        assert inc is None and rows[0]["stage"] == "answered" and rows[0]["answer"] == "REAL" and call.call_count == 1
    m = json.loads((run / "manifest.json").read_text())["stages"][-1]
    assert m["retrieval_only"] is False and m["complete"] is True


def test_replay_missing_context_is_refused_before_any_model_call(tmp_path):
    cfg = RunConfig()
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": "q1", **_fake_context(), "retrieved_doc_ids": ["d1"]}])
    questions = [{"question_id": "q1", "question": "Q1"}, {"question_id": "q2", "question": "Q2"}]
    with patch.object(generate, "generate_answer", return_value=("REAL", {})) as call:
        with pytest.raises(SystemExit):
            _run(cfg, tmp_path / "run", questions, contexts)
        assert call.call_count == 0


def test_replay_forged_hash_is_refused_before_any_model_call(tmp_path):
    cfg = RunConfig()
    ctx = _fake_context()
    forged = {**ctx, "context": "CHANGED CONTENT"}  # keeps the old sha256 label and char count
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": "q1", **forged, "retrieved_doc_ids": ["d1"]}])
    with patch.object(generate, "generate_answer", return_value=("REAL", {})) as call:
        with pytest.raises(SystemExit):
            _run(cfg, tmp_path / "run", [{"question_id": "q1", "question": "Q1"}], contexts)
        assert call.call_count == 0


def test_replay_duplicate_context_rows_are_refused(tmp_path):
    cfg = RunConfig()
    row = {"question_id": "q1", **_fake_context(), "retrieved_doc_ids": ["d1"]}
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [row, row])
    with pytest.raises(SystemExit):
        _run(cfg, tmp_path / "run", [{"question_id": "q1", "question": "Q1"}], contexts)


def test_rejected_resume_does_not_mutate_config_file(tmp_path):
    cfg = RunConfig()
    ctx = _fake_context()
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": "q1", **ctx, "retrieved_doc_ids": ["d1"]}])
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text(json.dumps({"question_id": "q1", "question": "Q1"}) + "\n")
    old_cfg = tmp_path / "old.yaml"
    cfg.save(old_cfg)
    run = tmp_path / "run"
    with patch("erb_hydradb.llm.provider_available", return_value=True), \
         patch.object(generate, "generate_answer", return_value=("A", {"pass1": True})):
        assert cli.main(["generate", "--allow-unpinned", "--config", str(old_cfg), "--questions", str(qpath),
                         "--run-dir", str(run), "--from-contexts", str(contexts)]) == 0
        cfg.retrieval.mode = "fast"
        new_cfg = tmp_path / "new.yaml"
        cfg.save(new_cfg)
        with pytest.raises(SystemExit):
            cli.main(["generate", "--allow-unpinned", "--config", str(new_cfg), "--questions", str(qpath),
                      "--run-dir", str(run), "--from-contexts", str(contexts)])
    assert RunConfig.load(run / "config.yaml").retrieval.mode == "thinking"


def test_targets_with_unknown_ids_is_refused(tmp_path):
    t = tmp_path / "t.txt"
    t.write_text("qst_0001\nnot_a_question\n")
    with pytest.raises(SystemExit):
        generate.select_questions(generate.load_questions(QUESTIONS), t, None)


# ------------------------------------------------------- corrections -----

def test_published_corrections_audit_passes(tmp_path):
    out = tmp_path / "audit.json"
    assert cli.main(["audit-corrections", "--run-dir", str(ART), "--questions", str(QUESTIONS), "--out", str(out)]) == 0
    a = json.loads(out.read_text())
    assert a["flagged_corrected"] == 14 == a["records"] and a["ok"]


def test_corrections_audit_detects_missing_record(tmp_path):
    for name in ("official_results_protocol.json", "answers.jsonl"):
        (tmp_path / name).write_bytes((ART / name).read_bytes())
    rows = validate.read_jsonl_rows(ART / "corrections.jsonl")[1:]
    _write_contexts(tmp_path / "corrections.jsonl", rows)
    assert cli.main(["audit-corrections", "--run-dir", str(tmp_path), "--questions", str(QUESTIONS)]) == 1


# ------------------------------------------------- release re-review (0d7c6ba) -----

def _fixture_run(tmp_path, n=3):
    """A small completed generation run against a mocked HydraDB (no store: chunk fallback)."""
    from test_lifecycle import FakeQuery

    class NoStore:
        backend = "test"
        identity = {"backend": "test"}

        def get(self, d):
            return None

    rows = [{"question_id": f"q{i}", "question": f"Q{i}", "question_type": "basic"} for i in range(1, n + 1)]
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "run"
    with patch("erb_hydradb.hydradb_client.HydraDBQueryClient", FakeQuery), \
            patch.object(generate, "generate_answer", return_value=("A", {})):
        asyncio.run(generate.run_generate(RunConfig(), run, rows, api_key="mock", store=NoStore(),
                                          questions_path=qpath))
    return run, qpath, rows, NoStore()


def _rerun(run, qpath, rows, store, **kw):
    from test_lifecycle import FakeQuery
    with patch("erb_hydradb.hydradb_client.HydraDBQueryClient", FakeQuery), \
            patch.object(generate, "generate_answer", return_value=("A2", {})):
        return asyncio.run(generate.run_generate(RunConfig(), run, rows, api_key="mock", store=store,
                                                 questions_path=qpath, **kw))


def test_torn_attempt_record_is_quarantined_and_resume_converges(tmp_path):
    """P2-3: an interrupted append leaves half a JSON line; resume must not crash,
    must keep the committed attempts, and must end with a run that verifies."""
    run, qpath, rows, store = _fixture_run(tmp_path)
    attempts = run / generate.ATTEMPTS
    before = validate.read_jsonl_rows(attempts)
    # simulate a crash mid-append: partial record without newline, and drop q3 from the checkpoint
    with open(attempts, "a", encoding="utf-8") as f:
        f.write('{"question_id": "q3", "attempt": 99, "context": "half wri')
    ck = json.loads((run / "gen_checkpoint.json").read_text())
    ck = [ck[0]] + [r for r in ck[1:] if r["question_id"] != "q3"]
    (run / "gen_checkpoint.json").write_text(json.dumps(ck))
    with pytest.raises(ValueError, match="invalid JSON"):
        validate.read_jsonl_rows(attempts)   # the strict reader sees the torn record
    out, inc = _rerun(run, qpath, rows, store)
    assert inc is None and {r["question_id"] for r in out} == {"q1", "q2", "q3"}
    torn = list(run.glob(generate.ATTEMPTS + ".torn-*"))
    assert len(torn) == 1 and "half wri" in torn[0].read_text()
    after = validate.read_jsonl_rows(attempts)
    assert after[:len(before)] == before and after[-1]["question_id"] == "q3" and after[-1]["attempt"] == len(before) + 1
    report = validate.validate_run(run, qpath, required=("SHA256SUMS",))
    assert report["ok"], validate.format_report(report)


def test_mid_journal_corruption_is_an_error_not_skipped(tmp_path):
    from erb_hydradb import identity
    j = tmp_path / "j.jsonl"
    j.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    with pytest.raises(ValueError, match="malformed record at line 2"):
        identity.read_journal(j)
    assert j.read_text() == '{"a": 1}\nnot json\n{"a": 2}\n'   # untouched


def test_narrowed_selection_is_refused_before_writing(tmp_path):
    """P2-5: a checkpoint with q1..q3 must not be resumed with --n 1 / --targets q1:
    answers.jsonl and contexts.jsonl.gz would export fewer rows than the checkpoint."""
    # CLI path: replay run over three questions, then the same run narrowed to one
    ctx = _fake_context()
    contexts = tmp_path / "contexts.jsonl"
    _write_contexts(contexts, [{"question_id": f"q{i}", **ctx, "retrieved_doc_ids": ["d1"]} for i in (1, 2, 3)])
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text("".join(json.dumps({"question_id": f"q{i}", "question": f"Q{i}"}) + "\n" for i in (1, 2, 3)))
    cfg_path = tmp_path / "cfg.yaml"
    RunConfig().save(cfg_path)
    run = tmp_path / "run_cli"
    base = ["generate", "--allow-unpinned", "--config", str(cfg_path), "--questions", str(qpath),
            "--run-dir", str(run), "--from-contexts", str(contexts)]
    with patch("erb_hydradb.llm.provider_available", return_value=True), \
            patch.object(generate, "generate_answer", return_value=("A", {"pass1": True})):
        assert cli.main(base) == 0
        sums_before = (run / "SHA256SUMS").read_bytes()
        (run / "config.yaml").unlink()
        with patch.object(generate, "generate_answer") as call, pytest.raises(SystemExit, match="narrower selection"):
            cli.main(base + ["--n", "1"])
        assert call.call_count == 0
        assert not (run / "config.yaml").exists()          # refused before the CLI wrote anything
        assert (run / "SHA256SUMS").read_bytes() == sums_before

    # library path, forced: the extra rows stay exported and the run still verifies
    run, qpath, rows, store = _fixture_run(tmp_path)
    with pytest.raises(SystemExit, match="narrower selection"):
        _rerun(run, qpath, rows[:1], store)
    out, inc = _rerun(run, qpath, rows[:1], store, force_resume=True)
    assert inc is None and [r["question_id"] for r in out] == ["q1", "q2", "q3"]
    assert {r["question_id"] for r in validate.read_jsonl_rows(run / "answers.jsonl")} == {"q1", "q2", "q3"}
    assert validate.validate_run(run, qpath, required=("SHA256SUMS",))["ok"]
    # the same run dir with the full selection again is fine without force
    out, inc = _rerun(run, qpath, rows, store)
    assert inc is None and len(out) == 3


def test_narrowing_with_force_still_refuses_incomplete_extra_rows(tmp_path):
    run, qpath, rows, store = _fixture_run(tmp_path)
    ck = json.loads((run / "gen_checkpoint.json").read_text())
    for r in ck[1:]:
        if r["question_id"] == "q3":
            r["stage"] = "retrieved"
    (run / "gen_checkpoint.json").write_text(json.dumps(ck))
    with pytest.raises(SystemExit, match="not complete"):
        _rerun(run, qpath, rows[:2], store, force_resume=True)


@pytest.mark.parametrize("field,value,needle", [
    ("count", float("nan"), "not a non-negative integer"),
    ("count", True, "not a non-negative integer"),
    ("count", "12", "not a non-negative integer"),
    ("average_recall_pct", float("nan"), "not a finite number"),
    ("average_recall_pct", None, "not a finite number"),
    ("average_recall_pct", "96.63", "not a finite number"),
    ("average_recall_pct", 101, "out of range"),
    ("combined_correctness_completeness_score", -1, "out of range"),
])
def test_category_statistics_must_be_typed_finite_and_bounded(tmp_path, field, value, needle):
    """P2-7: NaN compares unequal to everything, so a tolerance test alone let it through."""
    data = _results()
    data["question_type_stats"]["basic"][field] = value
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    problems, _ = validate.check_results_file(p, None)
    assert any(f"question_type_stats.basic.{field}" in x and needle in x for x in problems), problems


def test_aggregate_statistics_must_be_typed_finite_and_bounded(tmp_path):
    data = _results()
    data["aggregate_stats"]["average_completeness_pct"] = float("nan")
    data["aggregate_stats"]["completed_questions"] = 1.5
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    problems, _ = validate.check_results_file(p, None)
    assert any("aggregate_stats.average_completeness_pct" in x and "finite" in x for x in problems)
    assert any("aggregate_stats.completed_questions" in x for x in problems)


def test_null_retrieval_metrics_with_gold_documents_fail_verification(tmp_path):
    """P2-7: a null document_recall_pct on a question that has gold documents is not
    'not applicable', it is a missing measurement."""
    import shutil
    run = tmp_path / "run"
    shutil.copytree(ART, run)
    p = run / "official_results_strict.json"
    d = json.loads(p.read_text())
    victim = next(q for q in d["questions"] if q.get("document_recall_pct") is not None)
    victim["document_recall_pct"] = None
    victim["invalid_extra_docs"] = None
    p.write_text(json.dumps(d))
    problems = validate.check_retrieval_metrics(run, QUESTIONS)
    assert any(victim["question_id"] in x and "null retrieval metrics" in x for x in problems), problems


def test_gold_sets_carry_the_pinned_valid_doc_ids(tmp_path):
    """valid_doc_ids from the questions file must count as valid, not only those from corrections."""
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"question_id": "q1", "expected_doc_ids": ["g1"], "valid_doc_ids": ["v1", "v2"]}) + "\n")
    golds = validate.gold_sets(q, None)
    assert golds["q1"] == ({"g1"}, {"v1", "v2"})
    assert validate.retrieval_metrics(["g1", "v1", "x"], *golds["q1"]) == (100.0, 1)
    c = tmp_path / "corrections.jsonl"
    c.write_text(json.dumps({"question_id": "q1", "after": {"expected_doc_ids": ["g2"], "valid_doc_ids": ["v3"]}}) + "\n")
    assert validate.gold_sets(q, c)["q1"] == ({"g2"}, {"v3"})


def test_valid_set_only_correction_records_are_accepted_by_the_audit(tmp_path):
    """The evaluator marks a question `updated` when it only gained valid_doc_ids; the
    results row is then NOT flagged corrected. Such a record must be exported (it changes
    invalid_extra_docs) and must pass the audit; a record that changes nothing must not."""
    import shutil
    run = tmp_path / "run"
    run.mkdir()
    for name in ("official_results_protocol.json", "answers.jsonl"):
        shutil.copy2(ART / name, run / name)
    records = validate.read_jsonl_rows(ART / "corrections.jsonl")
    results = {r["question_id"]: r for r in analysis.load_results(run / "official_results_protocol.json")["questions"]}
    qs = {q["question_id"]: q for q in validate.read_jsonl_rows(QUESTIONS)}
    victim = next(q for q, r in results.items() if not r.get("corrected") and qs[q].get("expected_doc_ids"))
    orig = qs[victim]
    rec = {"question_id": victim, "question_type": orig["question_type"],
           "before": {k: orig.get(k) for k in ("expected_doc_ids", "gold_answer", "answer_facts")},
           "after": {**{k: orig.get(k) for k in ("expected_doc_ids", "gold_answer", "answer_facts")},
                     "valid_doc_ids": ["dsid_extra_valid"]},
           "update_reasons": ["valid only"], "doc_set_changed": False, "gold_answer_changed": False, "source": "test"}
    (run / "corrections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records + [rec]))
    assert cli.main(["audit-corrections", "--run-dir", str(run), "--questions", str(QUESTIONS)]) == 0
    rec["after"]["valid_doc_ids"] = []
    (run / "corrections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records + [rec]))
    assert cli.main(["audit-corrections", "--run-dir", str(run), "--questions", str(QUESTIONS)]) == 1
