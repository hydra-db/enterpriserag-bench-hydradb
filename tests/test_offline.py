"""Offline tests: no network, no keys. Run with `pytest`."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from erb_hydradb import analysis, hydrate
from erb_hydradb.config import RunConfig

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "run-2026-09-04"
BASE = REPO / "artifacts" / "baseline-2026-09-04"
CFG = REPO / "config" / "run-2026-09-04.yaml"


# ---------------------------------------------------------------- hydrate --

def test_canonical_extraction_single_and_multi_field():
    single = {"title_field_name": "t", "content_field_names": ["body"], "t": "T", "body": "hello"}
    assert hydrate.extract_document_content(single) == ("T", "hello")
    multi = {"title_field_name": "subject", "content_field_names": ["a", "items"],
             "subject": "S", "a": "x", "items": ["one", "two"], "dataset_noise_document": True}
    assert hydrate.extract_document_content(multi) == ("S", "a:\nx\n\nitems:\none\ntwo")


def test_doc_id_suffix_stripping():
    assert hydrate.erb_doc_id({"id": "dsid_abc_m0001"}) == "dsid_abc"
    assert hydrate.erb_doc_id("dsid_abc_c12") == "dsid_abc"
    assert hydrate.erb_doc_id("dsid_abc_e3") == "dsid_abc"
    assert hydrate.erb_doc_id("dsid_abc") == "dsid_abc"
    assert hydrate.distinct_docs([{"id": "dsid_a_m1"}, {"id": "dsid_b"}, {"id": "dsid_a_m2"}]) == ["dsid_a", "dsid_b"]


class _Store:
    backend = "test"

    def __init__(self, docs):
        self.docs = docs

    def get(self, did):
        return self.docs.get(did)


def test_build_context_budget_drops_whole_docs_and_never_truncates():
    docs = {f"d{i}": hydrate.Document(f"d{i}", "slack", f"title{i}", "x" * 1000) for i in range(5)}
    ctx = hydrate.build_context([f"d{i}" for i in range(5)], [], _Store(docs), 5, 2500)
    assert ctx["context_docs"] == ["d0", "d1"]
    assert ctx["dropped_docs"] == ["d2", "d3", "d4"]
    assert "x" * 1000 in ctx["context"] and "[document 1 | source=slack | title=title0 | id=d0]" in ctx["context"]


def test_build_context_leak_assertion():
    docs = {"d0": hydrate.Document("d0", "slack", "t", "dataset_noise_document: true")}
    with pytest.raises(hydrate.ContextLeakError):
        hydrate.build_context(["d0"], [], _Store(docs), 1, 10_000)


def test_build_context_chunk_fallback_for_unknown_doc():
    ctx = hydrate.build_context(["missing"], [{"id": "missing_m0001", "chunk_content": "chunk text"}], _Store({}), 1, 10_000)
    assert ctx["chunk_fallback_docs"] == ["missing"] and "chunk text" in ctx["context"]


# ----------------------------------------------------------------- config --

def test_config_fingerprint_changes_with_generation_fields_only():
    cfg = RunConfig.load(CFG)
    fp = cfg.generation_fingerprint()
    cfg.generation.workers = 1
    assert cfg.generation_fingerprint() == fp
    cfg.retrieval.mode = "fast"
    assert cfg.generation_fingerprint() != fp


# --------------------------------------------------------------- analysis --

@pytest.mark.parametrize("name", ["official_results_strict.json", "official_results_protocol.json"])
def test_published_results_recompute_exactly(name):
    rc = analysis.recompute_stats(ART / name)
    assert rc["n"] == 500 and rc["diffs"] == []


def test_published_headline_numbers():
    strict = analysis.recompute_stats(ART / "official_results_strict.json")["recomputed"]["aggregate_stats"]
    proto = analysis.recompute_stats(ART / "official_results_protocol.json")["recomputed"]["aggregate_stats"]
    assert strict["combined_correctness_completeness_score"] == 88.3
    assert proto["combined_correctness_completeness_score"] == 88.73
    assert proto["average_correctness_pct"] == 92.0
    base = analysis.recompute_stats(BASE / "official_results_strict.json")["recomputed"]["aggregate_stats"]
    assert base["combined_correctness_completeness_score"] == 64.23


def test_answers_submission_format_and_coverage():
    rows = analysis.load_jsonl(ART / "answers.jsonl")
    assert len(rows) == 500
    for r in rows.values():
        assert set(r) == {"question_id", "answer", "document_ids"}
        assert r["answer"] and len(r["document_ids"]) <= 10


def test_contexts_match_checkpoint_hashes_and_carry_no_forbidden_markers():
    with open(ART / "gen_checkpoint.json", "r", encoding="utf-8") as f:
        ck = {r["question_id"]: r for r in json.load(f) if "question_id" in r}
    seen = 0
    with gzip.open(ART / "contexts.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            assert c["context_sha256"] == ck[c["question_id"]]["context_sha256"]
            for m in hydrate.FORBIDDEN_MARKERS:
                assert m not in c["context"]
            seen += 1
    assert seen == 500


def test_paired_compare_against_baseline(tmp_path):
    q = REPO / "tests" / "data" / "questions.jsonl"
    if not q.exists():
        pytest.skip("questions.jsonl not staged in tests/data")
    rep = analysis.compare(BASE / "official_results_strict.json", ART / "official_results_strict.json", q,
                           expect_n=500, b_checkpoint=ART / "gen_checkpoint.json", a_answers=BASE / "answers.jsonl")
    assert rep["paired_bootstrap"]["delta_mean"] == 24.07
    assert rep["flips"]["F->T"]["n"] == 124 and rep["flips"]["T->F"]["n"] == 10
