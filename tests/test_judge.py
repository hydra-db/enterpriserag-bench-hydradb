"""Judge wrapper tests: no network, no keys, no real git, no real evaluator.

``subprocess.run`` / ``subprocess.Popen`` are replaced by a fake that answers
git queries from canned state and "runs" a fake evaluator that honours the
upstream CLI (``--answers-file``, ``--results-file``, ``--resume``, ...) by
writing results files the way the real one does.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from erb_hydradb import judge, paths
from erb_hydradb.config import RunConfig

ORIG_OPENAI = b"# pristine upstream openai_llm.py\nclass OpenAILLM: ...\n"
ORIG_FACTORY = b"# pristine upstream factory.py\ndef get_llm(): ...\n"
LOCAL_EDIT = b"# USER'S LOCAL EXPERIMENT -- must never be touched\nclass OpenAILLM: pass\n"
N_ANSWERS = 40

FAKE_EVAL = '''
def compute_stats_for_group(results):
    n = len(results)
    if n == 0:
        return {"count": 0, "average_correctness_pct": 0.0, "average_completeness_pct": 0.0,
                "combined_correctness_completeness_score": 0.0, "average_recall_pct": 0.0,
                "average_invalid_extra_docs": 0.0}
    corr = 100.0 * sum(1 for r in results if r["answer_correct"]) / n
    comp = sum(r["completeness_pct"] for r in results) / n
    return {"count": n, "average_correctness_pct": corr, "average_completeness_pct": comp,
            "combined_correctness_completeness_score": corr * comp / 100.0,
            "average_recall_pct": 100.0, "average_invalid_extra_docs": 0.0}
'''


def _row(qid: str) -> dict:
    return {"question_id": qid, "corrected": False, "question_type": "basic", "answer_correct": True,
            "correctness_reasoning": "fake", "completeness_pct": 100.0, "document_recall_pct": 100.0,
            "invalid_extra_docs": 0}


class FakeProcs:
    """Stand-in for subprocess.run / subprocess.Popen."""

    def __init__(self, checkout: Path, head: str = paths.ERB_COMMIT, porcelain: str = ""):
        self.checkout = checkout
        self.head = head
        self.porcelain = porcelain
        self.calls: list[dict] = []
        self.evaluated: list[str] = []      # every question id the fake evaluator "spent money on"
        self.fail_shards: dict[str, int] = {}   # answers file stem -> rows to write before failing
        self.drop_ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.malformed: dict[str, dict] = {}    # qid -> field overrides
        self.fail_strict = False

    # -- git ---------------------------------------------------------------
    def _git(self, cmd, cwd):
        if cmd[1] == "rev-parse":
            return subprocess.CompletedProcess(cmd, 0, self.head + "\n", "")
        if cmd[1] == "status":
            return subprocess.CompletedProcess(cmd, 0, self.porcelain, "")
        raise AssertionError(f"unexpected git command (checkout must never be modified): {cmd}")

    # -- evaluator ---------------------------------------------------------
    def _evaluate(self, cmd, cwd, env) -> int:
        args = {cmd[i]: cmd[i + 1] for i in range(2, len(cmd) - 1) if cmd[i].startswith("--") and not cmd[i + 1].startswith("--")}
        flags = {c for c in cmd[2:] if c.startswith("--") and c not in args}
        overlay = Path(env["PYTHONPATH"])
        assert Path(cwd) == overlay, "evaluator must run from the overlay"
        assert (overlay / judge.EVAL_SCRIPT).exists()
        assert not overlay.is_relative_to(self.checkout), "overlay must not live in the user's checkout"
        assert (overlay / "generated_data" / "uuid_index.json").exists()
        assert (overlay / "questions.jsonl").exists()
        answers = Path(args["--answers-file"])
        results = Path(args["--results-file"])
        rows = [json.loads(line) for line in answers.read_text().splitlines() if line.strip()]
        existing: list[dict] = []
        if "--resume" in flags and results.exists():
            existing = json.load(open(results))["questions"]
        done = {r["question_id"] for r in existing}
        todo = [r["question_id"] for r in rows if r["question_id"] not in done]
        limit = self.fail_shards.get(answers.stem)
        if self.fail_strict and "--no-correction" in flags:
            limit = 0
        out = list(existing)
        for qid in todo[: limit if limit is not None else None]:
            self.evaluated.append(qid)
            if qid in self.drop_ids:
                continue
            out.append({**_row(qid), **self.malformed.get(qid, {})})
            if qid in self.duplicate_ids:
                out.append(_row(qid))
        results.write_text(json.dumps({"aggregate_stats": {}, "question_type_stats": {}, "questions": out}))
        return 1 if limit is not None else 0

    def run(self, cmd, cwd=None, env=None, **kw):
        self.calls.append({"cmd": list(cmd), "cwd": cwd})
        if cmd[0] == "git":
            return self._git(cmd, cwd)
        return subprocess.CompletedProcess(cmd, self._evaluate(cmd, cwd, env), "", "")

    def Popen(self, cmd, cwd=None, env=None, stdout=None, **kw):  # noqa: N802
        self.calls.append({"cmd": list(cmd), "cwd": cwd})
        rc = self._evaluate(cmd, cwd, env)
        if stdout is not None:
            stdout.write("fake evaluator output\n")
            stdout.flush()

        class P:
            returncode = rc

            def wait(self):
                return rc
        return P()

    def evaluator_calls(self) -> list[list[str]]:
        return [c["cmd"] for c in self.calls if c["cmd"][0] != "git"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    erb = tmp_path / "EnterpriseRAG-Bench"
    for rel, content in {
        "src/__init__.py": b"", "src/llm/__init__.py": b"", "src/llm/openai_llm.py": ORIG_OPENAI,
        "src/llm/factory.py": ORIG_FACTORY, "src/utils/__init__.py": b"", "src/utils/helpers.py": b"X = 1\n",
        "src/scripts/__init__.py": b"", "src/scripts/answer_evaluation/__init__.py": b"",
        "src/scripts/answer_evaluation/metrics_based_eval.py": FAKE_EVAL.encode(),
        "generated_data/uuid_index.json": b"{}",
    }.items():
        (erb / rel).parent.mkdir(parents=True, exist_ok=True)
        (erb / rel).write_bytes(content)
    (erb / "src" / "llm" / "__pycache__").mkdir()
    (erb / "src" / "llm" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    ids = [f"qst_{i:04d}" for i in range(1, N_ANSWERS + 1)]
    questions = erb / "questions.jsonl"
    questions.write_text("".join(json.dumps({"question_id": q, "question_type": "basic"}) + "\n" for q in ids))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    answers = run_dir / "answers.jsonl"
    answers.write_text("".join(json.dumps({"question_id": q, "answer": "a", "document_ids": []}) + "\n" for q in ids))
    monkeypatch.setenv("ERB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "oa-test")
    fake = FakeProcs(erb)
    monkeypatch.setattr(judge.subprocess, "run", fake.run)
    monkeypatch.setattr(judge.subprocess, "Popen", fake.Popen)
    cfg = RunConfig()
    cfg.judge.provider = "openrouter"
    cfg.judge.shards = 20
    return {"erb": erb, "run_dir": run_dir, "answers": answers, "questions": questions, "fake": fake,
            "cfg": cfg, "ids": ids, "data": tmp_path / "data"}


def _strict(w, **kw):
    return judge.run_strict(w["cfg"], w["run_dir"], w["erb"], w["questions"], w["answers"], None, **kw)


def _official(w, **kw):
    return judge.run_official(w["cfg"], w["run_dir"], w["erb"], w["questions"], w["answers"], None, **kw)


def _make_dirty(w, content=LOCAL_EDIT):
    (w["erb"] / "src" / "llm" / "openai_llm.py").write_bytes(content)
    w["fake"].porcelain = " M src/llm/openai_llm.py\n"


def _git_checkouts(fake):
    return [c["cmd"] for c in fake.calls if c["cmd"][:2] == ["git", "checkout"]]


def _llm_bytes(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((root / "src" / "llm").glob("*.py"))}


# ------------------------------------------------- (a) checkout never touched --

def test_missing_key_is_refused_before_any_filesystem_work(world, monkeypatch):
    w = world
    monkeypatch.delenv("OPENROUTER_API_KEY")
    _make_dirty(w)
    before = _llm_bytes(w["erb"])
    with pytest.raises(judge.MissingCredentials):
        _strict(w, allow_unpinned=True)
    with pytest.raises(judge.MissingCredentials):
        _official(w, allow_unpinned=True)
    assert _llm_bytes(w["erb"]) == before
    assert w["fake"].calls == []
    assert not (w["data"] / "evaluator").exists()
    assert not (w["run_dir"] / "protocol_shards").exists()


def test_dirty_checkout_is_refused_before_any_copy(world):
    w = world
    _make_dirty(w)
    with pytest.raises(judge.CheckoutError, match="local modifications"):
        _strict(w)
    with pytest.raises(judge.CheckoutError):
        _official(w)
    assert _llm_bytes(w["erb"])["openai_llm.py"] == LOCAL_EDIT
    assert not (w["data"] / "evaluator").exists()
    assert w["fake"].evaluator_calls() == []
    assert _git_checkouts(w["fake"]) == []


def test_unpinned_head_is_refused_and_recorded_when_allowed(world):
    w = world
    w["fake"].head = "deadbeef" * 5
    with pytest.raises(judge.CheckoutError, match="not the pinned"):
        _strict(w)
    assert not (w["data"] / "evaluator").exists()
    _strict(w, allow_unpinned=True)
    stage = json.load(open(w["run_dir"] / "manifest.json"))["stages"][-1]
    assert stage["evaluator_overlay"]["checkout_head"] == "deadbeef" * 5
    assert stage["evaluator_overlay"]["dirty_files"] == []


def test_local_modification_survives_openrouter_success_and_failure(world):
    w, fake = world, world["fake"]
    _make_dirty(w)
    before = _llm_bytes(w["erb"])
    out = _strict(w, allow_unpinned=True)
    assert out.exists()
    fake.fail_strict = True
    with pytest.raises(judge.EvaluatorFailed):
        _strict(w, allow_unpinned=True)
    assert _llm_bytes(w["erb"]) == before, "the user's checkout was modified"
    assert _git_checkouts(fake) == []
    # the overlay, not the checkout, carries the patch bytes
    ov = _llm_bytes(w["data"] / "evaluator" / "openrouter")
    assert ov["openai_llm.py"] == (judge.PATCH_DIR / "openai_llm.py").read_bytes()
    assert ov["factory.py"] == (judge.PATCH_DIR / "factory.py").read_bytes()
    stage = json.load(open(w["run_dir"] / "manifest.json"))["stages"][-1]
    assert stage["evaluator_overlay"]["dirty_files"] == ["src/llm/openai_llm.py"]
    assert stage["evaluator_overlay"]["patch_sha256"] == judge.patch_sha256s()
    assert set(stage["evaluator_overlay"]["src_llm_sha256"]) == {"src/llm/__init__.py", "src/llm/openai_llm.py", "src/llm/factory.py"}


def test_reviewer_probe_modified_llm_file_never_triggers_git_restore(world):
    """Reviewer probe P1-1: with provider=openai and ' M src/llm/openai_llm.py' the old
    code ran `git checkout -- src/llm/openai_llm.py src/llm/factory.py`."""
    w, fake = world, world["fake"]
    w["cfg"].judge.provider = "openai"
    _make_dirty(w)
    _strict(w, allow_unpinned=True)
    assert _git_checkouts(fake) == []
    assert all(c["cmd"][:2] in (["git", "rev-parse"], ["git", "status"]) for c in fake.calls if c["cmd"][0] == "git")
    assert _llm_bytes(w["erb"])["openai_llm.py"] == LOCAL_EDIT
    assert not hasattr(judge, "restore_patches") and not hasattr(judge, "apply_patches")


# ------------------------------------------------------------ (b) overlays --

def test_overlays_per_provider_coexist(world):
    w = world
    oa = judge.evaluator_overlay(w["erb"], "openai")
    orr = judge.evaluator_overlay(w["erb"], "openrouter")
    assert oa.path != orr.path and oa.path.parent == orr.path.parent == w["data"] / "evaluator"
    assert _llm_bytes(oa.path) == {"__init__.py": b"", "openai_llm.py": ORIG_OPENAI, "factory.py": ORIG_FACTORY}
    assert _llm_bytes(orr.path)["openai_llm.py"] == (judge.PATCH_DIR / "openai_llm.py").read_bytes()
    assert _llm_bytes(orr.path)["factory.py"] == (judge.PATCH_DIR / "factory.py").read_bytes()
    assert oa.patch_sha256 == {} and set(orr.patch_sha256) == set(judge.PATCH_TARGETS)
    for ov in (oa, orr):
        assert (ov.path / "generated_data").is_symlink() and (ov.path / "questions.jsonl").is_symlink()
        assert (ov.path / "generated_data").resolve() == (w["erb"] / "generated_data").resolve()
        assert ov.cwd == ov.path
        assert not list(ov.path.rglob("__pycache__"))
        state = json.load(open(ov.path / "overlay.json"))
        assert state["src_tree_sha256"] == ov.src_tree_sha256 and state["provider"] == ov.provider
    # building one after the other left both intact
    assert _llm_bytes(oa.path)["openai_llm.py"] == ORIG_OPENAI
    assert _llm_bytes(w["erb"]) == {"__init__.py": b"", "openai_llm.py": ORIG_OPENAI, "factory.py": ORIG_FACTORY}


def test_overlay_is_reused_until_source_or_patch_changes(world, monkeypatch):
    w = world
    first = judge.evaluator_overlay(w["erb"], "openrouter")
    assert first.rebuilt
    helper = first.path / "src" / "utils" / "helpers.py"
    mtime = helper.stat().st_mtime_ns
    again = judge.evaluator_overlay(w["erb"], "openrouter")
    assert not again.rebuilt and helper.stat().st_mtime_ns == mtime
    (w["erb"] / "src" / "utils" / "helpers.py").write_bytes(b"X = 2\n")
    w["fake"].porcelain = " M src/utils/helpers.py\n"
    third = judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True)
    assert third.rebuilt and helper.read_bytes() == b"X = 2\n"
    assert third.src_tree_sha256 != first.src_tree_sha256
    monkeypatch.setattr(judge, "patch_sha256s", lambda: {k: "0" * 64 for k in judge.PATCH_TARGETS})
    assert judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True).rebuilt


def test_overlay_falls_back_to_recorded_paths_without_symlinks(world, monkeypatch):
    w = world

    def no_symlink(*a, **k):
        raise OSError("symlinks unavailable")
    monkeypatch.setattr(judge.os, "symlink", no_symlink)
    ov = judge.evaluator_overlay(w["erb"], "openai")
    assert ov.links == {"generated_data": {"mode": "path", "target": str(w["erb"] / "generated_data")},
                        "questions.jsonl": {"mode": "path", "target": str(w["erb"] / "questions.jsonl")}}
    assert ov.cwd == w["erb"].resolve()
    assert json.load(open(ov.path / "overlay.json"))["links"] == ov.links


# --------------------------------------------- (c)/(d) non-destructive retries --

def _snapshot(shard_dir: Path) -> dict[str, tuple[int, bytes]]:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in shard_dir.glob("results_*.json")}


def test_official_retry_reruns_only_the_failed_shard(world):
    w, fake = world, world["fake"]
    fake.fail_shards = {"answers_07": 1}          # shard 07 writes 1 of its 2 rows, then dies
    with pytest.raises(judge.EvaluatorFailed, match=r"\['07'\]"):
        _official(w, expect_n=N_ANSWERS)
    shard_dir = w["run_dir"] / "protocol_shards"
    plan = json.load(open(shard_dir / "shards.json"))
    assert plan["shard_count"] == 20 and len(plan["shards"]) == 20
    assert plan["shards"][7]["expected_ids"] == ["qst_0015", "qst_0016"]
    assert [s["id"] for s in plan["shards"]] == [f"{i:02d}" for i in range(20)]
    assert len(fake.evaluator_calls()) == 20
    assert all("--resume" not in c for c in fake.evaluator_calls())
    first = _snapshot(shard_dir)
    assert len(first) == 20
    assert sorted(p.name for p in shard_dir.glob("log_*")) == sorted(f"log_{i:02d}.attempt1.txt" for i in range(20))
    spent_first = list(fake.evaluated)
    assert len(spent_first) == N_ANSWERS - 1

    # retry: same plan, no fresh, same key
    fake.fail_shards = {}
    fake.calls.clear()
    out = _official(w, expect_n=N_ANSWERS)
    calls = fake.evaluator_calls()
    assert len(calls) == 1 and calls[0][calls[0].index("--answers-file") + 1].endswith("answers_07.jsonl")
    assert "--resume" in calls[0]
    assert fake.evaluated[len(spent_first):] == ["qst_0016"], "only the missing row was re-judged"
    second = _snapshot(shard_dir)
    for sid in first:
        if sid != "results_07.json":
            assert second[sid] == first[sid], f"{sid} was rewritten"
    assert (shard_dir / "log_07.attempt1.txt").exists() and (shard_dir / "log_07.attempt2.txt").exists()
    assert "--resume" in (shard_dir / "log_07.attempt2.txt").read_text()
    merged = json.load(open(out))
    assert [q["question_id"] for q in merged["questions"]] == w["ids"]
    assert merged["aggregate_stats"]["average_correctness_pct"] == 100.0
    stage = json.load(open(w["run_dir"] / "manifest.json"))["stages"][-1]
    assert stage["stage"] == "judge_official" and stage["shards_run"] == {"07": 2}
    assert len(stage["shards_skipped"]) == 19 and stage["archived_shards"] is None
    # a third invocation with everything complete runs nothing at all
    fake.calls.clear()
    _official(w, expect_n=N_ANSWERS)
    assert fake.evaluator_calls() == []


def test_official_changed_answers_refused_without_fresh_and_archived_with_it(world):
    w, fake = world, world["fake"]
    _official(w)
    shard_dir = w["run_dir"] / "protocol_shards"
    before = _snapshot(shard_dir)
    w["answers"].write_text("".join(json.dumps({"question_id": q, "answer": "changed", "document_ids": []}) + "\n"
                                    for q in w["ids"]))
    with pytest.raises(judge.PlanMismatch, match="answers_sha256"):
        _official(w)
    assert _snapshot(shard_dir) == before
    fake.calls.clear()
    _official(w, fresh=True)
    archived = [p for p in w["run_dir"].glob("protocol_shards.*") if p.is_dir()]
    assert len(archived) == 1
    assert _snapshot(archived[0]) == before, "archive must hold the old results byte-for-byte"
    assert (archived[0] / "shards.json").exists() and (archived[0] / "log_00.attempt1.txt").exists()
    assert len(fake.evaluator_calls()) == 20
    assert json.load(open(shard_dir / "shards.json"))["answers_sha256"] != json.load(open(archived[0] / "shards.json"))["answers_sha256"]


def test_official_changed_model_or_shard_layout_is_a_different_plan(world):
    w = world
    _official(w)
    w["cfg"].judge.model = "openai/other"
    with pytest.raises(judge.PlanMismatch, match="model"):
        _official(w)
    w["cfg"].judge.model = RunConfig().judge.model
    w["cfg"].judge.shards = 10
    with pytest.raises(judge.PlanMismatch, match="shard_count"):
        _official(w)


def test_official_legacy_shard_dir_without_plan_is_refused(world):
    w = world
    shard_dir = w["run_dir"] / "protocol_shards"
    shard_dir.mkdir()
    (shard_dir / "results_00.json").write_text("{}")
    with pytest.raises(judge.PlanMismatch):
        _official(w)
    assert (shard_dir / "results_00.json").exists()
    _official(w, fresh=True)
    assert any(p.is_dir() for p in w["run_dir"].glob("protocol_shards.*"))


# ------------------------------------------------------------- (e) coverage --

def test_strict_missing_question_id_raises(world):
    w = world
    w["fake"].drop_ids = {"qst_0003"}
    with pytest.raises(judge.CoverageError, match="qst_0003"):
        _strict(w)


def test_strict_duplicate_id_raises(world):
    w = world
    w["fake"].duplicate_ids = {"qst_0005"}
    with pytest.raises(judge.CoverageError, match="duplicate"):
        _strict(w)


def test_strict_malformed_rows_raise(world):
    w = world
    w["fake"].malformed = {"qst_0002": {"answer_correct": "yes"}}
    with pytest.raises(judge.CoverageError, match="malformed"):
        _strict(w)
    w["fake"].malformed = {"qst_0002": {"completeness_pct": None}}
    with pytest.raises(judge.CoverageError, match="malformed"):
        _strict(w)


def test_reviewer_probe_expect_n_is_enforced_for_strict(world):
    """Reviewer probe P1-6: `--expect-n 999` used to be ignored for --protocol strict."""
    w = world
    with pytest.raises(judge.CoverageError, match="999"):
        _strict(w, expect_n=999)
    assert _strict(w, expect_n=N_ANSWERS).exists()
    assert json.load(open(w["run_dir"] / "manifest.json"))["stages"][-1]["expect_n"] == N_ANSWERS


def test_strict_single_question_mode_keeps_other_rows(world):
    w = world
    _strict(w)
    w["fake"].calls.clear()
    _strict(w, question_id="qst_0009")
    cmd = w["fake"].evaluator_calls()[0]
    assert cmd[cmd.index("--question-id") + 1] == "qst_0009"
    with pytest.raises(judge.CoverageError):
        _strict(w, question_id="qst_9999")


def test_official_coverage_rejects_shard_that_dropped_a_row(world):
    w = world
    w["fake"].drop_ids = {"qst_0020"}
    with pytest.raises(judge.EvaluatorFailed, match=r"\['09'\]"):
        _official(w)


# ---------------------------------------------------------------- (f) merge --

def test_merge_rejects_shard_whose_ids_do_not_match_expected(world):
    w = world
    out = _official(w)
    shard_dir = w["run_dir"] / "protocol_shards"
    plan = json.load(open(shard_dir / "shards.json"))
    ov = judge.evaluator_overlay(w["erb"], "openrouter")
    # swap one id in shard 03's results for one that belongs to shard 04
    res = shard_dir / "results_03.json"
    data = json.load(open(res))
    data["questions"][0]["question_id"] = plan["shards"][4]["expected_ids"][0]
    res.write_text(json.dumps(data))
    with pytest.raises(judge.CoverageError, match="shard 03"):
        judge.merge_shards(ov, shard_dir, plan, out, None)
    # and a shard with an extra, duplicated row
    data["questions"][0]["question_id"] = plan["shards"][3]["expected_ids"][0]
    data["questions"].append(dict(data["questions"][0]))
    res.write_text(json.dumps(data))
    with pytest.raises(judge.CoverageError, match="duplicate"):
        judge.merge_shards(ov, shard_dir, plan, out, None)
    # and expect_n against a valid set
    data["questions"].pop()
    res.write_text(json.dumps(data))
    with pytest.raises(judge.CoverageError, match="expected 41"):
        judge.merge_shards(ov, shard_dir, plan, out, 41)
    merged = judge.merge_shards(ov, shard_dir, plan, out, N_ANSWERS)
    assert len(merged["questions"]) == N_ANSWERS and merged["aggregate_stats"]["num_corrected_questions"] == 0


def test_judge_env_sets_provider_and_optional_cheap_model(world, monkeypatch):
    w = world
    env = judge.judge_env(w["cfg"])
    assert env["LLM_PROVIDER"] == "openrouter" and env["CHEAP_LLM_MODEL_NAME"] == w["cfg"].judge.cheap_model
    monkeypatch.delenv("CHEAP_LLM_MODEL_NAME", raising=False)
    w["cfg"].judge.cheap_model = ""
    assert "CHEAP_LLM_MODEL_NAME" not in judge.judge_env(w["cfg"])
    w["cfg"].judge.provider = "openai"
    env = judge.judge_env(w["cfg"])
    assert env["LLM_API_KEY"] == "oa-test" and "OPENROUTER_API_KEY" not in env
