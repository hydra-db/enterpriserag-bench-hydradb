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
