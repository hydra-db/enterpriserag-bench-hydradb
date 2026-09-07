"""Judge wrapper tests: no network, no keys, no real git, no real evaluator.

``subprocess.run`` / ``subprocess.Popen`` are replaced by a fake that answers
git queries from canned state (or, for the corpus-identity tests, routes them
to real git against a tiny committed fixture repository) and "runs" a fake
evaluator that honours the upstream CLI (``--answers-file``, ``--results-file``,
``--updated-questions-file``, ``--resume``, ...) by writing results and
updated-questions files the way the real one does.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from erb_hydradb import identity, judge, paths
from erb_hydradb.config import RunConfig

REAL_POPEN = subprocess.Popen       # captured before any monkeypatching (subprocess.run calls Popen)


def real_git(cmd, cwd, env, check=False, **_):
    """Run a real git command through the original Popen (the fake replaces both
    ``subprocess.run`` and ``subprocess.Popen``)."""
    with REAL_POPEN(list(cmd), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        out, err = proc.communicate()
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)

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
        self.real_git: dict | None = None   # env -> route git to the real binary (git-backed fixture)
        self.calls: list[dict] = []
        self.evaluated: list[str] = []      # every question id the fake evaluator "spent money on"
        self.fail_shards: dict[str, int] = {}   # answers file stem -> rows to write before failing
        self.drop_ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.malformed: dict[str, dict] = {}    # qid -> field overrides
        self.updates: dict[str, dict] = {}      # qid -> updated-question row overrides (official protocol)
        self.fail_strict = False

    # -- git ---------------------------------------------------------------
    def _git(self, cmd, cwd, kw):
        if self.real_git is not None:
            return real_git(cmd, cwd, self.real_git, **kw)
        if cmd[1] == "rev-parse":
            return subprocess.CompletedProcess(cmd, 0, self.head + "\n", "")
        if cmd[1] == "status":
            # honour the pathspec: `-- src` and `-- generated_data questions.jsonl` see different files
            specs = cmd[cmd.index("--") + 1:] if "--" in cmd else []
            lines = [line for line in self.porcelain.splitlines() if line.strip()
                     and (not specs or any(line[3:].startswith(sp) for sp in specs))]
            return subprocess.CompletedProcess(cmd, 0, "".join(line + "\n" for line in lines), "")
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
        # the real evaluator may regenerate its UUID index cache: write it like it would
        cache = Path(args["--uuid-index-cache-file"])
        assert cache.is_relative_to(overlay), "uuid index cache must be harness-owned (inside the overlay)"
        cache.write_text('{"regenerated_by": "fake evaluator"}')
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
            row = {**_row(qid), **self.malformed.get(qid, {})}
            if qid in self.updates:
                row["corrected"] = True
            out.append(row)
            if qid in self.duplicate_ids:
                out.append(_row(qid))
        results.write_text(json.dumps({"aggregate_stats": {}, "question_type_stats": {}, "questions": out}))
        rc = 1 if limit is not None else 0
        if rc == 0 and "--no-correction" not in flags:
            # like the real evaluator: the WHOLE questions file, with this shard's own corrections applied
            mine = {r["question_id"] for r in rows}
            updated = []
            for line in Path(args["--questions-file"]).read_text().splitlines():
                if line.strip():
                    q = json.loads(line)
                    if q["question_id"] in mine and q["question_id"] in self.updates:
                        q = {**q, "updated": True, **self.updates[q["question_id"]]}
                    updated.append(q)
            Path(args["--updated-questions-file"]).write_text("".join(json.dumps(q) + "\n" for q in updated))
        return rc

    def run(self, cmd, cwd=None, env=None, **kw):
        self.calls.append({"cmd": list(cmd), "cwd": cwd})
        if cmd[0] == "git":
            return self._git(cmd, cwd, kw)
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


def _overlays(w, provider: str) -> list[Path]:
    """Overlay directories built for ``provider`` (named ``<provider>-<identity hash>``)."""
    root = w["data"] / "evaluator"
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{provider}-")) if root.exists() else []


def _stage(w) -> dict:
    return json.load(open(w["run_dir"] / "manifest.json"))["stages"][-1]


def _corpus_identity(w) -> str:
    return identity.checkout_state(w["erb"])["corpus_identity"]


def _git_world(w, monkeypatch) -> str:
    """Turn the fixture checkout into a real (tiny) git repository with everything
    committed, pin ``paths.ERB_COMMIT`` to its HEAD and route git through the
    real binary. The fake evaluator is unchanged. Returns HEAD."""
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("GIT_DIR", None)

    def git(*args):
        out = real_git(["git", *args], w["erb"], env)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()
    git("init", "-q")
    git("add", "-A")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture checkout")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(paths, "ERB_COMMIT", head)
    w["fake"].real_git = env
    w["git"] = git
    return head


def _corrections(w) -> list[dict]:
    return judge.validate.read_jsonl_rows(w["run_dir"] / "corrections.jsonl")


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
    [overlay] = _overlays(w, "openrouter")
    ov = _llm_bytes(overlay)
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
    """An overlay is immutable: the same identity is reused untouched; a changed
    source tree or patch set is a *new* directory and the old one is left intact."""
    w = world
    first = judge.evaluator_overlay(w["erb"], "openrouter")
    assert first.rebuilt and first.path.name.startswith("openrouter-")
    helper = first.path / "src" / "utils" / "helpers.py"
    mtime = helper.stat().st_mtime_ns
    state = (first.path / "overlay.json").read_bytes()
    again = judge.evaluator_overlay(w["erb"], "openrouter")
    assert not again.rebuilt and again.path == first.path and helper.stat().st_mtime_ns == mtime
    assert (first.path / "overlay.json").read_bytes() == state, "reuse must not rewrite the overlay"
    (w["erb"] / "src" / "utils" / "helpers.py").write_bytes(b"X = 2\n")
    w["fake"].porcelain = " M src/utils/helpers.py\n"
    third = judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True)
    assert third.rebuilt and third.path != first.path
    assert (third.path / "src" / "utils" / "helpers.py").read_bytes() == b"X = 2\n"
    assert helper.read_bytes() == b"X = 1\n", "the first overlay was modified"
    assert third.src_tree_sha256 != first.src_tree_sha256
    monkeypatch.setattr(judge, "patch_sha256s", lambda: {k: "0" * 64 for k in judge.PATCH_TARGETS})
    fourth = judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True)
    assert fourth.rebuilt and fourth.path not in (first.path, third.path)
    assert len(_overlays(w, "openrouter")) == 3


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
    corr = w["run_dir"] / "corrections.jsonl"
    assert corr.exists() and corr.read_bytes() == b"", "no corrections: still an (empty) file, so verify knows"
    assert "corrections.jsonl" in stage["files"] and "questions_effective.jsonl" in stage["files"]
    assert len(judge.validate.read_jsonl_rows(w["run_dir"] / "questions_effective.jsonl")) == N_ANSWERS
    # a third invocation with everything complete runs nothing at all
    fake.calls.clear()
    _official(w, expect_n=N_ANSWERS)
    assert fake.evaluator_calls() == []
    assert corr.exists() and corr.read_bytes() == b""


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


# ------------------------------------ (g) 2026-09-07 re-review: plan identity --

def test_reviewer_probe_same_plan_fresh_reruns_everything_and_archives(world):
    """Reviewer probe P1-3b: with an identical plan `--fresh` used to be ignored
    (reuse branch won, zero evaluator calls, nothing archived)."""
    w, fake = world, world["fake"]
    _official(w)
    shard_dir = w["run_dir"] / "protocol_shards"
    before = _snapshot(shard_dir)
    fake.calls.clear()
    spent = len(fake.evaluated)
    _official(w, fresh=True)
    archived = [p for p in w["run_dir"].glob("protocol_shards.*") if p.is_dir()]
    assert len(archived) == 1 and _snapshot(archived[0]) == before
    assert len(fake.evaluator_calls()) == 20 and not any("--resume" in c for c in fake.evaluator_calls())
    assert len(fake.evaluated) == spent + N_ANSWERS, "every question must be judged again"
    stage = _stage(w)
    assert stage["archived_shards"] == str(archived[0]) and stage["shards_skipped"] == []
    assert len(stage["shards_run"]) == 20
    assert stage["plan_fingerprint"] == json.load(open(shard_dir / "shards.json"))["fingerprint"]
    assert not (w["run_dir"] / judge.LOCK_FILE).exists()
    assert judge.LOCK_FILE not in (w["run_dir"] / "SHA256SUMS").read_text()


def test_reviewer_probe_changed_evaluator_source_is_a_different_plan(world):
    """Reviewer probe P1-3a: after the evaluator source changed under allow_unpinned
    a resume used to reuse every shard (zero calls) while recording the new hash."""
    w, fake = world, world["fake"]
    _official(w)
    old = _stage(w)
    assert old["overlay_identity"]["src_tree_sha256"] == old["evaluator_overlay"]["src_tree_sha256"]
    (w["erb"] / "src" / "utils" / "helpers.py").write_bytes(b"X = 2\n")
    fake.porcelain = " M src/utils/helpers.py\n"
    fake.calls.clear()
    with pytest.raises(judge.PlanMismatch, match="overlay_identity"):
        _official(w, allow_unpinned=True)
    assert fake.evaluator_calls() == []
    assert _stage(w) == old, "no manifest stage may be written for a refused run"
    _official(w, allow_unpinned=True, fresh=True)
    new = _stage(w)
    assert new["overlay_identity"]["src_tree_sha256"] != old["overlay_identity"]["src_tree_sha256"]
    assert new["plan_fingerprint"] != old["plan_fingerprint"] and new["overlay_dir"] != old["overlay_dir"]
    assert len(fake.evaluator_calls()) == 20 and new["archived_shards"]
    plan = json.load(open(w["run_dir"] / "protocol_shards" / "shards.json"))
    assert set(judge.FINGERPRINT_KEYS) <= set(plan)
    assert set(plan["overlay_identity"]) == {"src_tree_sha256", "patch_sha256", "checkout_head"}
    assert plan["corpus_identity"] == _corpus_identity(w) == plan["corpus_state"]["corpus_identity"]
    assert set(plan["corpus_state"]) == set(judge.CORPUS_STATE_KEYS)


def test_changed_corpus_is_a_different_plan(world):
    """Corpus identity is HEAD + the content of every dirty/untracked file under the
    corpus paths -- the index file is just one of them, and its sha256 is no longer
    the identity (it is kept only as provenance of the overlay's cache copy)."""
    w, fake = world, world["fake"]
    _official(w)
    start = _stage(w)
    assert start["corpus_identity"] == _corpus_identity(w) and start["corpus_state"]["dirty"] == []
    (w["erb"] / "generated_data" / "uuid_index.json").write_text('{"doc-1": "uuid-1"}')
    fake.porcelain = " M generated_data/uuid_index.json\n"      # src/ stays clean: no CheckoutError
    fake.calls.clear()
    with pytest.raises(judge.PlanMismatch, match="corpus_identity"):
        _official(w)
    assert fake.evaluator_calls() == []
    _official(w, fresh=True)
    assert len(fake.evaluator_calls()) == 20
    stage = _stage(w)
    assert stage["corpus_identity"] != start["corpus_identity"]
    assert stage["corpus_state"]["dirty"] == ["generated_data/uuid_index.json"]
    assert stage["evaluator_overlay"]["uuid_index_sha256"] == judge.manifest.sha256_file(
        w["erb"] / "generated_data" / "uuid_index.json")
    assert stage["overlay_identity"]["corpus_identity"] == stage["corpus_identity"]
    assert "corpus_sha256" not in stage["overlay_identity"]


def test_reviewer_probe_two_checkouts_same_provider_get_distinct_immutable_overlays(world):
    """Reviewer probe P1-3c: preparing a second checkout used to repoint the first
    overlay's corpus symlink (data/evaluator/<provider>/ was shared and mutable)."""
    import shutil
    w = world
    first = judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True)
    first_state = (first.path / "overlay.json").read_bytes()
    first_links = {n: os.readlink(first.path / n) for n in judge.OVERLAY_LINKS}
    second_checkout = w["erb"].parent / "second-checkout"
    shutil.copytree(w["erb"], second_checkout)
    w["fake"].checkout = second_checkout          # fake git answers for either checkout
    second = judge.evaluator_overlay(second_checkout, "openrouter", allow_unpinned=True)
    assert first.path != second.path and second.path.parent == first.path.parent
    assert first.path.name.startswith("openrouter-") and second.path.name.startswith("openrouter-")
    assert (first.path / "generated_data").resolve() == (w["erb"] / "generated_data").resolve()
    assert (second.path / "generated_data").resolve() == (second_checkout / "generated_data").resolve()
    assert {n: os.readlink(first.path / n) for n in judge.OVERLAY_LINKS} == first_links
    assert (first.path / "overlay.json").read_bytes() == first_state
    assert first.identity["checkout"] != second.identity["checkout"]
    assert judge.overlay_name(first.identity) == first.path.name
    # a different revision of the same checkout is yet another overlay
    w["fake"].checkout, w["fake"].head = w["erb"], "cafebabe" * 5
    third = judge.evaluator_overlay(w["erb"], "openrouter", allow_unpinned=True)
    assert third.path not in (first.path, second.path) and third.identity["checkout_head"] == "cafebabe" * 5
    assert len(_overlays(w, "openrouter")) == 3


def test_overlay_build_is_atomic(world, monkeypatch):
    w = world
    real_copyfile = judge.shutil.copyfile
    copied = []

    def die_mid_copy(src, dst, *a, **k):
        copied.append(dst)
        if len(copied) == 1:
            raise OSError("disk full (simulated) mid-copy")
        return real_copyfile(src, dst, *a, **k)
    monkeypatch.setattr(judge.shutil, "copyfile", die_mid_copy)
    with pytest.raises(OSError, match="simulated"):
        judge.evaluator_overlay(w["erb"], "openrouter")
    root = w["data"] / "evaluator"
    assert list(root.iterdir()) == [], f"partial overlay left behind: {list(root.iterdir())}"
    monkeypatch.setattr(judge.shutil, "copyfile", real_copyfile)
    ov = judge.evaluator_overlay(w["erb"], "openrouter")
    assert ov.rebuilt and [p.name for p in root.iterdir()] == [ov.path.name]
    assert (ov.path / "overlay.json").exists() and (ov.path / "uuid_index.json").exists()
    assert json.load(open(ov.path / "overlay.json"))["identity"] == ov.identity


def test_run_dir_lock_refuses_concurrent_judge_and_clears_stale_lock(world, monkeypatch, capsys):
    w, fake = world, world["fake"]
    lock = w["run_dir"] / judge.LOCK_FILE
    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": "2026-09-07T00:00:00+00:00", "host": "h"}))
    with pytest.raises(judge.RunLocked, match=str(os.getpid())):
        _official(w)
    assert fake.evaluator_calls() == [] and not (w["run_dir"] / "protocol_shards").exists()
    assert lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid(), "a live lock must not be touched"
    # stale: the recorded pid is gone
    lock.write_text(json.dumps({"pid": 4242, "started_at": "2026-09-06T00:00:00+00:00", "host": "h"}))
    monkeypatch.setattr(identity, "_pid_alive", lambda pid: pid != 4242)
    _official(w)
    assert "stale" in capsys.readouterr().out
    assert not lock.exists() and len(fake.evaluator_calls()) == 20
    # a lock is held for the whole run and released even when the evaluator fails
    fake.fail_shards = {"answers_03": 0}
    seen = {}
    real_run = judge._run_shards

    def spy(*a, **k):
        seen["locked"] = lock.exists() and json.loads(lock.read_text())["pid"] == os.getpid()
        return real_run(*a, **k)
    monkeypatch.setattr(judge, "_run_shards", spy)
    (w["run_dir"] / "protocol_shards" / "results_03.json").unlink()
    with pytest.raises(judge.EvaluatorFailed):
        _official(w)
    assert seen["locked"] and not lock.exists()


def test_reviewer_probe_strict_refuses_bad_answers_before_any_subprocess(world):
    """Reviewer probe H6a: a duplicate row used to reach the evaluator (money spent)
    and was only caught by the results validation afterwards."""
    w, fake = world, world["fake"]
    good = w["answers"].read_text()

    def refused(match):
        with pytest.raises(judge.CoverageError, match=match):
            _strict(w, expect_n=N_ANSWERS)
        with pytest.raises(judge.CoverageError, match=match):
            _official(w, expect_n=N_ANSWERS)
        assert fake.calls == [], "no git and no evaluator call may happen before the answers file is validated"
        assert not (w["data"] / "evaluator").exists() and not (w["run_dir"] / "protocol_shards").exists()

    with w["answers"].open("a") as f:
        f.write(json.dumps({"question_id": w["ids"][0], "answer": "conflicting duplicate", "document_ids": []}) + "\n")
    refused("duplicate question ids")
    w["answers"].write_text(good + json.dumps({"question_id": "qst_9999", "answer": "x", "document_ids": []}) + "\n")
    refused("not in questions.jsonl")
    w["answers"].write_text(good.replace('"document_ids": []', '"docs": []', 1))
    refused("expected \\['answer', 'document_ids', 'question_id'\\]")
    w["answers"].write_text(good.replace('"answer": "a"', '"answer": 7', 1))
    refused("answer must be a string")
    w["answers"].write_text(good + "{not json\n")
    refused("invalid JSON")
    w["answers"].write_text(good)
    with pytest.raises(judge.CoverageError, match="expect_n 41"):
        _official(w, expect_n=41)
    assert fake.calls == []
    _strict(w, expect_n=N_ANSWERS)
    assert len(fake.evaluator_calls()) == 1


def test_reviewer_probe_uuid_index_cache_is_harness_owned(world):
    """Reviewer probe H6b: `--uuid-index-cache-file` used to point into the checkout,
    which the evaluator may regenerate/write."""
    w, fake = world, world["fake"]
    checkout_copy = w["erb"] / "generated_data" / "uuid_index.json"
    assert checkout_copy.read_bytes() == b"{}"
    _strict(w)
    _official(w)
    for cmd in fake.evaluator_calls():
        cache = Path(cmd[cmd.index("--uuid-index-cache-file") + 1])
        overlay = Path(_stage(w)["overlay_dir"])
        assert cache == overlay / "uuid_index.json" and not cache.is_relative_to(w["erb"])
    assert checkout_copy.read_bytes() == b"{}", "the checkout's uuid_index.json was written"
    assert (Path(_stage(w)["overlay_dir"]) / "uuid_index.json").read_text() == '{"regenerated_by": "fake evaluator"}'
    assert _stage(w)["evaluator_overlay"]["uuid_index_cache"] == str(Path(_stage(w)["overlay_dir"]) / "uuid_index.json")


# ---------------------------- (h) 2026-09-07 release re-review: corpus identity --

def _doc(w) -> Path:
    return w["erb"] / "generated_data" / "sources" / "doc.json"


def _seed_document(w) -> Path:
    doc = _doc(w)
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "original body"}))
    (w["erb"] / "generated_data" / "uuid_index.json").write_text(json.dumps({"doc-1": "sources/doc.json"}))
    return doc


def test_reviewer_probe_changed_document_body_is_a_different_plan(world, monkeypatch):
    """Reviewer probe P1-1: the corpus identity used to be the sha256 of
    uuid_index.json, so editing a document BODY (index unchanged) resumed the old
    plan with zero evaluator calls while the overlay's link read the new body."""
    w, fake = world, world["fake"]
    doc = _seed_document(w)
    _git_world(w, monkeypatch)
    _official(w)
    before = _stage(w)
    overlay = Path(before["overlay_dir"])
    assert before["corpus_state"]["dirty"] == [] and before["corpus_identity"] == _corpus_identity(w)
    assert before["corpus_state"]["head"] == paths.ERB_COMMIT
    assert json.load(open(overlay / "overlay.json"))["corpus_state"] == before["corpus_state"]
    # the overlay still links to the mutable checkout: a changed body IS what the evaluator would read
    doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "REPLACEMENT body"}))
    assert json.loads((overlay / "generated_data" / "sources" / "doc.json").read_text())["content"] == "REPLACEMENT body"
    assert judge.uuid_index_sha256(w["erb"]) == before["evaluator_overlay"]["uuid_index_sha256"], "index unchanged"
    fake.calls.clear()
    with pytest.raises(judge.PlanMismatch, match="corpus_identity"):
        _official(w)
    assert fake.evaluator_calls() == [], "a changed corpus must never be resumed for free"
    assert _stage(w) == before, "no manifest stage may be written for a refused run"
    # restoring the exact bytes restores the identity: the plan is the same again and nothing is re-judged
    doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "original body"}))
    _official(w)
    assert fake.evaluator_calls() == [] and _stage(w)["corpus_identity"] == before["corpus_identity"]
    # a changed body under --fresh is a new plan, a new overlay and a full re-judge
    doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "REPLACEMENT body"}))
    fake.calls.clear()
    _official(w, fresh=True)
    after = _stage(w)
    assert len(fake.evaluator_calls()) == 20
    assert after["plan_fingerprint"] != before["plan_fingerprint"] and after["overlay_dir"] != before["overlay_dir"]
    assert after["corpus_state"]["dirty"] == ["generated_data/sources/doc.json"]
    assert after["corpus_state"]["head"] == before["corpus_state"]["head"]
    assert after["corpus_identity"] == _corpus_identity(w) != before["corpus_identity"]
    plan = json.load(open(w["run_dir"] / "protocol_shards" / "shards.json"))
    assert plan["corpus_identity"] == after["corpus_identity"] and plan["corpus_state"] == after["corpus_state"]
    # an untracked file under the corpus paths counts too
    (w["erb"] / "generated_data" / "sources" / "extra.json").write_text("{}")
    with pytest.raises(judge.PlanMismatch, match="corpus_identity"):
        _official(w)


def test_corpus_changed_during_official_run_fails_and_flags_the_manifest(world, monkeypatch):
    w, fake = world, world["fake"]
    doc = _seed_document(w)
    _git_world(w, monkeypatch)
    real_run_shards = judge._run_shards

    def mutate_between_launch_and_merge(*a, **k):
        out = real_run_shards(*a, **k)
        doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "edited while judging"}))
        return out
    monkeypatch.setattr(judge, "_run_shards", mutate_between_launch_and_merge)
    with pytest.raises(judge.EvaluatorFailed, match="corpus changed during judging"):
        _official(w)
    stage = _stage(w)
    assert stage["stage"] == "judge_official" and stage["corpus_changed_during_run"] is True
    assert stage["corpus_state"]["dirty"] == [] and stage["corpus_state_at_end"]["dirty"] == ["generated_data/sources/doc.json"]
    assert stage["corpus_state_at_end"]["corpus_identity"] != stage["corpus_identity"]
    assert (w["run_dir"] / "official_results_protocol.json").exists(), "results are left in place"
    assert (w["run_dir"] / "corrections.jsonl").exists()
    assert not (w["run_dir"] / judge.LOCK_FILE).exists()
    # the plan was made against the start state, so a plain resume is refused now ...
    monkeypatch.setattr(judge, "_run_shards", real_run_shards)
    fake.calls.clear()
    with pytest.raises(judge.PlanMismatch, match="corpus_identity"):
        _official(w)
    assert fake.evaluator_calls() == []
    # ... and restoring the checkout makes the completed shards usable again, with a clean flag
    w["git"]("checkout", "--", "generated_data")
    _official(w)
    assert fake.evaluator_calls() == [] and _stage(w)["corpus_changed_during_run"] is False


def test_corpus_changed_during_strict_run_fails_and_flags_the_manifest(world, monkeypatch):
    w, fake = world, world["fake"]
    doc = _seed_document(w)
    _git_world(w, monkeypatch)
    evaluate = fake._evaluate

    def evaluate_then_mutate(cmd, cwd, env):
        rc = evaluate(cmd, cwd, env)
        doc.write_text(json.dumps({"dataset_doc_uuid": "doc-1", "content": "edited while judging"}))
        return rc
    monkeypatch.setattr(fake, "_evaluate", evaluate_then_mutate)
    with pytest.raises(judge.EvaluatorFailed, match="corpus changed during judging"):
        _strict(w)
    stage = _stage(w)
    assert stage["stage"] == "judge_strict" and stage["corpus_changed_during_run"] is True
    assert stage["corpus_state_at_end"]["dirty"] == ["generated_data/sources/doc.json"]
    assert (w["run_dir"] / "official_results_strict.json").exists()


def test_end_of_run_corpus_state_is_recorded_for_strict_and_official(world):
    w = world
    _strict(w)
    strict = _stage(w)
    assert strict["stage"] == "judge_strict" and strict["corpus_changed_during_run"] is False
    assert set(strict["corpus_state"]) == set(strict["corpus_state_at_end"]) == set(judge.CORPUS_STATE_KEYS)
    assert strict["corpus_state"] == strict["corpus_state_at_end"]
    assert strict["corpus_identity"] == strict["corpus_state"]["corpus_identity"] == _corpus_identity(w)
    assert strict["corpus_state"]["head"] == paths.ERB_COMMIT
    assert strict["evaluator_overlay"]["corpus_state"] == strict["corpus_state"]
    assert json.load(open(Path(strict["overlay_dir"]) / "overlay.json"))["corpus_state"] == strict["corpus_state"]
    _official(w)
    official = _stage(w)
    assert official["stage"] == "judge_official" and official["corpus_changed_during_run"] is False
    assert official["corpus_state"] == official["corpus_state_at_end"] == strict["corpus_state"]
    assert json.load(open(w["run_dir"] / "protocol_shards" / "shards.json"))["corpus_state"] == strict["corpus_state"]


# ------------------------------------- (i) release re-review: corrections export --

def test_reviewer_probe_lifecycle_official_corrections_are_materialised_and_verify_passes(world, monkeypatch, tmp_path):
    """Reviewer probe P1-2: official judging wrote per-shard questions_updated files but
    never the run's corrections.jsonl, so `verify` scored corrected results against
    the pinned gold and failed retrieval_metrics."""
    from unittest.mock import patch

    import test_lifecycle as tl

    from erb_hydradb import analysis, cli, corpus, generate, hydrate, validate
    w, fake = world, world["fake"]
    db = tmp_path / "documents.sqlite"
    corpus.build_from_repo(tl.FIXTURE_REPO, db, quiet=True)
    doc_id = next(corpus.iter_documents(db))["doc_id"]
    store = hydrate.DocumentStore(db, None)
    monkeypatch.setattr(tl, "DOC_ID", doc_id)
    qs = [{"question_id": q, "question": "Q", "question_type": "basic", "expected_doc_ids": [doc_id],
           "gold_answer": "gold", "answer_facts": ["fact"]} for q in w["ids"]]
    qs[0]["expected_doc_ids"] = ["original-required"]          # the judges will replace it with the submitted doc
    w["questions"].write_text("".join(json.dumps(q) + "\n" for q in qs))
    with patch.object(generate, "generate_answer", return_value=("GENERATED ANSWER", {})):
        rows, incomplete = tl._run(w["cfg"], w["run_dir"], qs, w["questions"], store)
    assert incomplete is None and all(r["stage"] == "answered" for r in rows)
    pre = validate.validate_run(w["run_dir"], w["questions"], required=("SHA256SUMS",))
    assert pre["ok"], validate.format_report(pre)
    submitted = validate.read_jsonl_rows(w["answers"])[0]["document_ids"]
    assert submitted == [doc_id]
    reasons = {doc_id: {"classification": "required", "reason": "mocked consensus"},
               "original-required": {"classification": "invalid", "reason": "mocked consensus"}}
    fake.updates[w["ids"][0]] = {"expected_doc_ids": [doc_id], "gold_answer": "corrected gold",
                                 "answer_facts": ["corrected fact"], "update_reasons": reasons}
    monkeypatch.setattr(judge, "_load_compute_stats", lambda ov: analysis.stats_for_group)
    _official(w)
    corr = w["run_dir"] / "corrections.jsonl"
    records = _corrections(w)
    assert records == [{
        "question_id": w["ids"][0], "question_type": "basic",
        "before": {"expected_doc_ids": ["original-required"], "gold_answer": "gold", "answer_facts": ["fact"]},
        "after": {"expected_doc_ids": [doc_id], "gold_answer": "corrected gold", "answer_facts": ["corrected fact"],
                  "valid_doc_ids": []},
        "update_reasons": reasons, "doc_set_changed": True, "gold_answer_changed": True,
        "source": "questions_updated_00.jsonl written by the evaluator during this run"}]
    assert list(records[0]) == ["question_id", "question_type", "before", "after", "update_reasons",
                                "doc_set_changed", "gold_answer_changed", "source"], "published record layout"
    effective = validate.read_jsonl_rows(w["run_dir"] / "questions_effective.jsonl")
    assert [q["question_id"] for q in effective] == w["ids"]
    assert effective[0]["expected_doc_ids"] == [doc_id] and effective[0]["updated"] is True and effective[1] == qs[1]
    stage = _stage(w)
    assert {"corrections.jsonl", "questions_effective.jsonl", "official_results_protocol.json"} <= set(stage["files"])
    assert validate.gold_sets(w["questions"], corr)[w["ids"][0]] == ({doc_id}, set())
    report = validate.validate_run(w["run_dir"], w["questions"], required=("SHA256SUMS",))
    assert report["ok"], validate.format_report(report)
    assert report["checks"]["retrieval_metrics"]["status"] == "ok"
    assert cli.main(["audit-corrections", "--run-dir", str(w["run_dir"]), "--questions", str(w["questions"])]) == 0
    # a no-op retry re-materialises the same file byte for byte
    before = corr.read_bytes()
    fake.calls.clear()
    _official(w)
    assert fake.evaluator_calls() == [] and corr.read_bytes() == before
    assert validate.validate_run(w["run_dir"], w["questions"], required=("SHA256SUMS",))["ok"]
    # --fresh re-judges everything and rewrites the record from the new shard files
    fake.updates[w["ids"][0]]["gold_answer"] = "corrected gold v2"
    fake.calls.clear()
    _official(w, fresh=True)
    assert len(fake.evaluator_calls()) == 20
    assert [r["after"]["gold_answer"] for r in _corrections(w)] == ["corrected gold v2"]
    assert validate.read_jsonl_rows(w["run_dir"] / "questions_effective.jsonl")[0]["gold_answer"] == "corrected gold v2"
    assert validate.validate_run(w["run_dir"], w["questions"], required=("SHA256SUMS",))["ok"]
    assert cli.main(["audit-corrections", "--run-dir", str(w["run_dir"]), "--questions", str(w["questions"])]) == 0
    assert len([p for p in w["run_dir"].glob("protocol_shards.*") if p.is_dir()]) == 1


def test_corrections_are_taken_by_shard_ownership(world):
    """Every shard's questions_updated file holds all questions: a global
    last-record-wins merge would let shard 01's untouched copy of qst_0001 erase
    shard 00's correction, and an `updated` row for a question a shard does not
    own must be ignored."""
    w = world
    qs = [{"question_id": q, "question_type": "basic", "expected_doc_ids": ["gold-" + q], "gold_answer": "g " + q,
           "answer_facts": ["f " + q]} for q in w["ids"]]
    w["questions"].write_text("".join(json.dumps(q) + "\n" for q in qs))
    shard_dir = w["run_dir"] / "protocol_shards"
    shard_dir.mkdir()
    half = N_ANSWERS // 2
    plan = {"shards": [{"id": "00", "updated": "questions_updated_00.jsonl", "expected_ids": w["ids"][:half]},
                       {"id": "01", "updated": "questions_updated_01.jsonl", "expected_ids": w["ids"][half:]}]}
    own, foreign = w["ids"][0], w["ids"][-1]           # own -> shard 00; foreign -> shard 01

    def updated(q, **fields):
        return {**q, "updated": True, "update_reasons": {"d": {"classification": "required", "reason": "r"}}, **fields}
    shard00 = [updated(q, expected_doc_ids=["new-doc"], valid_doc_ids=["also-fine"]) if q["question_id"] in (own, foreign)
               else q for q in qs]
    shard00[2] = {**qs[2], "updated": False}            # explicit false is not a correction
    shard01 = list(qs)                                  # shard 01 saw nothing to correct
    (shard_dir / "questions_updated_00.jsonl").write_text("".join(json.dumps(q) + "\n" for q in shard00))
    (shard_dir / "questions_updated_01.jsonl").write_text("".join(json.dumps(q) + "\n" for q in shard01))
    # shard 01 written LAST still must not erase shard 00's correction, and shard 00 must not speak for shard 01
    records, effective = judge.collect_corrections(shard_dir, plan, w["questions"])
    assert [r["question_id"] for r in records] == [own]
    assert records[0]["before"] == {"expected_doc_ids": ["gold-" + own], "gold_answer": "g " + own, "answer_facts": ["f " + own]}
    assert records[0]["after"] == {"expected_doc_ids": ["new-doc"], "gold_answer": "g " + own, "answer_facts": ["f " + own],
                                   "valid_doc_ids": ["also-fine"]}
    assert records[0]["doc_set_changed"] is True and records[0]["gold_answer_changed"] is False
    assert records[0]["source"] == "questions_updated_00.jsonl written by the evaluator during this run"
    assert effective[0]["expected_doc_ids"] == ["new-doc"] and effective[-1] == qs[-1] and effective[2] == qs[2]
    # the export writes both files atomically and an empty corrections file when nothing was corrected
    corr, eff = judge.export_corrections(w["run_dir"], shard_dir, plan, w["questions"])
    assert judge.validate.read_jsonl_rows(corr) == records and len(judge.validate.read_jsonl_rows(eff)) == N_ANSWERS
    assert not list(w["run_dir"].glob("*.tmp-*"))
    (shard_dir / "questions_updated_00.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
    corr, _ = judge.export_corrections(w["run_dir"], shard_dir, plan, w["questions"])
    assert corr.exists() and corr.read_bytes() == b""
    # a shard whose updated file is missing cannot have its corrections recovered
    (shard_dir / "questions_updated_01.jsonl").unlink()
    with pytest.raises(judge.EvaluatorFailed, match="questions_updated_01.jsonl"):
        judge.collect_corrections(shard_dir, plan, w["questions"])


# ------------------------------------------------ (j) release re-review: the lock --

def test_reviewer_probe_lock_double_acquire_exactly_one_wins_and_release_is_owner_checked(world, monkeypatch):
    """Reviewer probe P2-4: the old create-then-write lock let a second acquirer in
    during the create-before-write interval (both "succeeded"), and releasing one
    removed the other's lock. The lock now publishes its owner record atomically
    (hard link) and release is owner-checked."""
    w = world
    lock = w["run_dir"] / judge.LOCK_FILE
    real_link = os.link
    nested: dict = {}

    def interleave(src, dst, *a, **k):
        # B runs its WHOLE acquisition after A has written its owner record but
        # before A publishes the lock; A then continues.
        if "outcome" not in nested and Path(dst) == lock:
            assert not lock.exists(), "the lock must not be observable before its record is complete"
            nested["outcome"] = "running"
            try:
                nested["outcome"] = judge.acquire_lock(w["run_dir"])
            except judge.RunLocked as exc:
                nested["outcome"] = exc
        return real_link(src, dst, *a, **k)
    monkeypatch.setattr(identity.os, "link", interleave)
    outcomes = []
    try:
        outcomes.append(judge.acquire_lock(w["run_dir"]))
    except judge.RunLocked as exc:
        outcomes.append(exc)
    outcomes.append(nested["outcome"])
    winners = [o for o in outcomes if isinstance(o, tuple)]
    losers = [o for o in outcomes if isinstance(o, judge.RunLocked)]
    assert len(winners) == 1 and len(losers) == 1, outcomes
    (lock_path, token) = winners[0]
    assert lock_path == lock and lock.exists()
    owner = json.loads(lock.read_text())
    assert owner["pid"] == os.getpid() and owner["token"] == token and owner["what"] == "judge_official"
    assert str(os.getpid()) in str(losers[0])
    assert not list(w["run_dir"].glob(f"{judge.LOCK_FILE}.*.tmp")), "no staging record left behind"
    # the loser (no token of its own) cannot remove the winner's lock ...
    judge.release_lock(lock, "not-the-owner-token")
    assert lock.exists() and json.loads(lock.read_text()) == owner
    # ... a third acquirer is still refused while it is held ...
    monkeypatch.setattr(identity.os, "link", real_link)
    with pytest.raises(judge.RunLocked):
        judge.acquire_lock(w["run_dir"])
    # ... and only the owner's release removes it
    judge.release_lock(lock, token)
    assert not lock.exists()
    judge.release_lock(lock, token)              # idempotent
    # a lock held by another pid whose owner is gone is taken over with a note
    lock.write_text(json.dumps({"pid": 4242, "token": "x", "started_at": "2026-09-06T00:00:00Z"}))
    monkeypatch.setattr(identity, "_pid_alive", lambda pid: pid != 4242)
    lock_path, token = judge.acquire_lock(w["run_dir"])
    assert json.loads(lock.read_text())["token"] == token
    judge.release_lock(lock_path, token)
    assert not lock.exists()
