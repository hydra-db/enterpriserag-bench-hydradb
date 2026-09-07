"""Scoring with the benchmark's own evaluator.

The evaluator is ``src/scripts/answer_evaluation/metrics_based_eval.py`` in the
EnterpriseRAG-Bench checkout. This module never reimplements any scoring; it
only sets up the environment, shards the work, and merges shard outputs with
the evaluator's own ``compute_stats_for_group``.

The evaluator is **never run from the user's checkout**. It runs from a
harness-owned *overlay* under ``data/evaluator/<provider>/``:

* ``src/`` — a copy of the checkout's ``src`` tree (pristine for ``openai``;
  for ``openrouter`` the two files ``src/llm/openai_llm.py`` and
  ``src/llm/factory.py`` are replaced by ``erb_patches/`` so the judge is called
  through OpenRouter's chat-completions endpoint);
* ``generated_data`` and ``questions.jsonl`` — symlinks into the checkout (the
  evaluator resolves corpus paths relative to its working directory; where
  symlinks cannot be created the absolute paths are recorded instead and the
  evaluator runs with the checkout as its working directory, still importing
  code from the overlay via ``PYTHONPATH``);
* ``overlay.json`` — the hashes the overlay was built from; the copy is
  refreshed only when the checkout's ``src`` tree or the patches change.

The checkout itself is only read. Before building an overlay the checkout must
be at the pinned commit with a clean ``src/``; otherwise the run is refused
unless ``allow_unpinned=True``, in which case the actual HEAD and dirty files
are recorded in the manifest. Credentials are validated before any of this.

Two protocols:

* ``strict`` — ``--no-correction --skip-citation-stripping``. One process.
  Comparable across runs because the gold set never changes.
* ``official`` — the full protocol: citation stripping and the three-judge
  document-correction flow (which may regenerate gold answers). Run as N
  concurrent shards. Shard state lives in ``<run_dir>/protocol_shards/`` with a
  ``shards.json`` plan. Re-running the same plan re-runs only the shards that
  are missing or incomplete, passing the evaluator's own ``--resume`` (which
  keeps the rows already in that shard's results file and evaluates the rest);
  previous results and logs are never deleted. A different plan (other answers,
  questions, provider, model, shard layout) is refused unless ``fresh=True``,
  which renames the old directory to ``protocol_shards.<timestamp>/``.

Both protocols verify coverage after the evaluator exits: the set of judged
question ids must equal the set of ids in the answers file (and ``expect_n``
when given), with no duplicates and well-formed rows.

Two judge providers:

* ``openai`` — the evaluator unmodified: OpenAI Responses API, reasoning effort
  "medium", model ``LLM_MODEL_NAME`` (default gpt-5.4). This is the maintainers'
  path and the one to use for an exact protocol reproduction.
* ``openrouter`` — the patched overlay described above. This is what the
  published run used. Reasoning-effort parity with the Responses API is not
  guaranteed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import manifest, paths
from .config import RunConfig

EVAL_SCRIPT = Path("src") / "scripts" / "answer_evaluation" / "metrics_based_eval.py"
PATCH_TARGETS = {"src/llm/openai_llm.py": "openai_llm.py", "src/llm/factory.py": "factory.py"}
PATCH_DIR = Path(__file__).parent / "erb_patches"
OVERLAY_LINKS = ("generated_data", "questions.jsonl")
PROVIDERS = ("openai", "openrouter")
SHARD_PLAN = "shards.json"
OVERLAY_STATE = "overlay.json"
FINGERPRINT_KEYS = ("answers_sha256", "questions_sha256", "provider", "model", "cheap_model",
                    "shard_parallelism", "shard_count")


# ------------------------------------------------------------- exceptions --

class JudgeError(RuntimeError):
    """Base class for every refusal or failure raised by this module."""


class MissingCredentials(JudgeError):
    """The provider's API key is not in the environment."""


class CheckoutError(JudgeError):
    """The benchmark checkout is dirty, not at the pinned commit, or not a checkout at all."""


class PlanMismatch(JudgeError):
    """``protocol_shards/`` holds state from a different official-protocol plan."""


class EvaluatorFailed(JudgeError):
    """The evaluator process (or one or more shards) exited non-zero or left no usable results."""


class CoverageError(JudgeError):
    """A results file does not cover exactly the expected question ids, or has malformed rows."""


# -------------------------------------------------------------- utilities --

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise CheckoutError(f"git {' '.join(args)} failed in {root}: {(out.stderr or '').strip()}")
    return out.stdout or ""


def _iter_src_files(src_dir: Path):
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".pyc", ".pyo"):
            continue
        yield p


def _src_tree_sha256(src_dir: Path) -> str:
    """One digest over the relative path and bytes of every file under ``src``."""
    h = hashlib.sha256()
    for p in _iter_src_files(src_dir):
        h.update(p.relative_to(src_dir).as_posix().encode("utf-8") + b"\0")
        h.update(manifest.sha256_file(p).encode("ascii") + b"\n")
    return h.hexdigest()


def patch_sha256s() -> dict[str, str]:
    """sha256 of each shipped patch file, keyed by the checkout-relative path it replaces."""
    return {rel: manifest.sha256_file(PATCH_DIR / fname) for rel, fname in PATCH_TARGETS.items()}


def answer_ids(answers: Path) -> list[str]:
    ids = []
    with open(answers, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["question_id"])
    return ids


# ------------------------------------------------------------ credentials --

def judge_env(cfg: RunConfig) -> dict:
    """Environment for the evaluator. Raises ``MissingCredentials`` if the
    provider's key is absent; called before any filesystem work."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # CHEAP_LLM_MODEL_NAME is read by the evaluator's src/utils/json_recovery.py
    # (get_cheap_llm). It is set from judge.cheap_model so that the id is valid
    # for the provider; whether the evaluator ever reaches that path is up to it.
    if cfg.judge.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise MissingCredentials("OPENROUTER_API_KEY is not set")
        env.update({"LLM_PROVIDER": "openrouter", "LLM_MODEL_NAME": cfg.judge.model, "OPENROUTER_API_KEY": key})
    elif cfg.judge.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise MissingCredentials("OPENAI_API_KEY is not set")
        env.update({"LLM_PROVIDER": "openai", "LLM_MODEL_NAME": cfg.judge.model, "LLM_API_KEY": key})
        env.pop("OPENROUTER_API_KEY", None)
    else:
        raise ValueError(f"unknown judge provider {cfg.judge.provider!r}")
    if cfg.judge.cheap_model:
        env["CHEAP_LLM_MODEL_NAME"] = cfg.judge.cheap_model
    return env


# ---------------------------------------------------------------- overlay --

@dataclass
class CheckoutState:
    root: Path
    head: str
    dirty_files: list[str]
    pinned: bool

    @property
    def clean(self) -> bool:
        return self.pinned and not self.dirty_files


def checkout_state(erb_root: Path) -> CheckoutState:
    """HEAD and the ``git status --porcelain -- src`` list of the checkout. Read-only."""
    head = _git(erb_root, "rev-parse", "HEAD").strip()
    porcelain = _git(erb_root, "status", "--porcelain", "--", "src")
    dirty = [line[3:].rstrip() for line in porcelain.splitlines() if line.strip()]
    return CheckoutState(erb_root, head, dirty, head == paths.ERB_COMMIT)


@dataclass
class Overlay:
    path: Path
    cwd: Path                       # the overlay when links are symlinks, else the checkout
    provider: str
    checkout: str
    checkout_head: str
    dirty_files: list[str]
    src_tree_sha256: str
    patch_sha256: dict[str, str]
    src_llm_sha256: dict[str, str]
    links: dict[str, dict] = field(default_factory=dict)
    rebuilt: bool = False

    def manifest_record(self) -> dict:
        d = asdict(self)
        d["path"], d["cwd"] = str(self.path), str(self.cwd)
        return d


def overlay_dir(provider: str) -> Path:
    return paths.data_dir() / "evaluator" / provider


def _link(overlay: Path, name: str, target: Path) -> dict:
    link = overlay / name
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
        return {"mode": "symlink", "target": str(target)}
    except (OSError, NotImplementedError):
        return {"mode": "path", "target": str(target)}


def evaluator_overlay(erb_root: Path, provider: str, *, allow_unpinned: bool = False) -> Overlay:
    """Build (or reuse) ``data/evaluator/<provider>/`` from the checkout at ``erb_root``.

    The checkout is only read. Refuses a dirty ``src/`` or a HEAD other than
    ``paths.ERB_COMMIT`` unless ``allow_unpinned`` is set.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown judge provider {provider!r}")
    erb_root = Path(erb_root).resolve()
    erb_src = erb_root / "src"
    if not erb_src.is_dir():
        raise CheckoutError(f"{erb_root} has no src/ directory; is it an EnterpriseRAG-Bench checkout?")
    state = checkout_state(erb_root)
    if not state.clean:
        why = []
        if not state.pinned:
            why.append(f"HEAD {state.head[:12]} is not the pinned {paths.ERB_COMMIT[:12]}")
        if state.dirty_files:
            why.append(f"src/ has local modifications: {state.dirty_files}")
        msg = f"checkout {erb_root}: " + "; ".join(why)
        if not allow_unpinned:
            raise CheckoutError(msg + " (pass allow_unpinned=True / --allow-unpinned to judge with it anyway)")
        print(f"WARNING: {msg}; recorded in the manifest", file=sys.stderr)

    patches = patch_sha256s() if provider == "openrouter" else {}
    tree = _src_tree_sha256(erb_src)
    root = overlay_dir(provider)
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / OVERLAY_STATE
    previous: dict = {}
    if state_file.exists():
        try:
            previous = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    reuse = ((root / "src").is_dir() and previous.get("src_tree_sha256") == tree
             and previous.get("patch_sha256") == patches and previous.get("checkout") == str(erb_root)
             and previous.get("provider") == provider)

    if not reuse:
        staging = root / f"src.build-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(erb_src, staging, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        if provider == "openrouter":
            for rel, fname in PATCH_TARGETS.items():
                shutil.copyfile(PATCH_DIR / fname, staging / Path(rel).relative_to("src"))
        if (root / "src").exists():
            shutil.rmtree(root / "src")
        os.replace(staging, root / "src")

    links = {name: _link(root, name, erb_root / name) for name in OVERLAY_LINKS}
    cwd = root if all(v["mode"] == "symlink" for v in links.values()) else erb_root
    llm_dir = root / "src" / "llm"
    src_llm = {p.relative_to(root).as_posix(): manifest.sha256_file(p) for p in _iter_src_files(llm_dir)}
    ov = Overlay(path=root, cwd=cwd, provider=provider, checkout=str(erb_root), checkout_head=state.head,
                 dirty_files=state.dirty_files, src_tree_sha256=tree, patch_sha256=patches,
                 src_llm_sha256=src_llm, links=links, rebuilt=not reuse)
    record = ov.manifest_record()
    record["built_at"] = previous.get("built_at") if reuse else manifest.now_iso()
    state_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return ov


# ------------------------------------------------------------- validation --

def load_result_ids(results: Path) -> list[str]:
    """Question ids of a results file, after checking every row is well-formed.
    Raises ``CoverageError`` on duplicates, non-bool ``answer_correct`` or
    missing ``completeness_pct``."""
    try:
        with open(results, "r", encoding="utf-8") as fh:
            rows = json.load(fh).get("questions", [])
    except FileNotFoundError:
        raise CoverageError(f"results file not written: {results}") from None
    except ValueError as e:
        raise CoverageError(f"results file {results} is not valid JSON: {e}") from None
    ids, bad = [], []
    for r in rows:
        qid = r.get("question_id")
        if not qid:
            bad.append("<missing question_id>")
            continue
        if not isinstance(r.get("answer_correct"), bool) or r.get("completeness_pct") is None:
            bad.append(qid)
        ids.append(qid)
    if bad:
        raise CoverageError(f"{results}: {len(bad)} malformed rows (answer_correct not bool or "
                            f"completeness_pct missing): {bad[:10]}")
    seen, dups = set(), set()
    for i in ids:
        (dups if i in seen else seen).add(i)
    if dups:
        raise CoverageError(f"{results}: duplicate question ids: {sorted(dups)[:10]}")
    return ids


def check_coverage(results: Path, expected: set[str], expect_n: int | None = None, *, label: str = "") -> list[str]:
    """Raise ``CoverageError`` unless ``results`` holds exactly ``expected`` (and ``expect_n`` rows)."""
    ids = load_result_ids(results)
    got = set(ids)
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise CoverageError(f"COVERAGE FAILURE {label}{results}: {len(got)} judged ids vs {len(expected)} expected; "
                            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''} extra={extra[:10]}")
    if expect_n is not None and len(ids) != expect_n:
        raise CoverageError(f"COVERAGE FAILURE {label}{results}: {len(ids)} judged rows, --expect-n {expect_n}")
    return ids


# ----------------------------------------------------------------- strict --

def _cmd(erb_root: Path, answers: Path, questions: Path, results: Path, updated: Path,
         parallelism: int, strict: bool, question_id: str | None, resume: bool = False) -> list[str]:
    cmd = [sys.executable, str(EVAL_SCRIPT), "--parallelism", str(parallelism),
           "--answers-file", str(answers), "--questions-file", str(questions),
           "--results-file", str(results), "--updated-questions-file", str(updated),
           "--uuid-index-cache-file", str(erb_root / "generated_data" / "uuid_index.json")]
    if strict:
        cmd += ["--no-correction", "--skip-citation-stripping"]
    if question_id:
        cmd += ["--question-id", question_id]
    if resume:
        cmd += ["--resume"]
    return cmd


def _overlay_env(env: dict, ov: Overlay) -> dict:
    env = dict(env)
    env["PYTHONPATH"] = str(ov.path)
    return env


def run_strict(cfg: RunConfig, run_dir: Path, erb_root: Path, questions: Path, answers: Path,
               results: Path | None = None, question_id: str | None = None, *, expect_n: int | None = None,
               allow_unpinned: bool = False, fresh: bool = False) -> Path:
    """Strict protocol in one evaluator process run from the overlay.

    ``fresh`` is accepted for parity with ``run_official`` and has no effect:
    strict keeps no shard state, and the evaluator rewrites the results file
    itself (with ``question_id`` it keeps the other rows and re-judges that one).
    """
    results = results or run_dir / "official_results_strict.json"
    env = judge_env(cfg)                     # credentials first: no filesystem work without a key
    ov = evaluator_overlay(Path(erb_root), cfg.judge.provider, allow_unpinned=allow_unpinned)
    cmd = _cmd(Path(erb_root).resolve(), answers.resolve(), questions.resolve(), results.resolve(),
               run_dir / "questions_updated_strict.jsonl", cfg.judge.strict_parallelism, True, question_id)
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ov.cwd, env=_overlay_env(env, ov), stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        raise EvaluatorFailed(f"evaluator exited {proc.returncode} (strict); results file: {results}")
    expected = set(answer_ids(answers))
    if question_id:
        if question_id not in expected:
            raise CoverageError(f"question id {question_id!r} is not in {answers}")
        ids = load_result_ids(results)
        if question_id not in ids or not set(ids) <= expected:
            raise CoverageError(f"COVERAGE FAILURE {results}: expected {question_id!r} among ids from {answers}")
    else:
        check_coverage(results, expected, expect_n)
    manifest.write_manifest(run_dir, "judge_strict", cfg.to_dict(), [results],
                            extra={"provider": cfg.judge.provider, "judge_model": cfg.judge.model,
                                   "question_id": question_id, "expect_n": expect_n,
                                   "evaluator_overlay": ov.manifest_record()})
    return results


# --------------------------------------------------------------- official --

def shard_plan(cfg: RunConfig, answers: Path, questions: Path) -> dict:
    """The official-protocol plan: fingerprint plus per-shard inputs and expected ids."""
    with open(answers, "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    n_shards = max(1, cfg.judge.shards)
    per = max(1, -(-len(rows) // n_shards))
    shards = []
    for i in range(0, len(rows), per):
        sid = f"{i // per:02d}"
        chunk = rows[i:i + per]
        shards.append({"id": sid, "input": f"answers_{sid}.jsonl", "result": f"results_{sid}.json",
                       "updated": f"questions_updated_{sid}.jsonl",
                       "expected_ids": [json.loads(r)["question_id"] for r in chunk],
                       "input_sha256": _sha256_bytes("".join(chunk).encode("utf-8"))})
    return {"answers_sha256": manifest.sha256_file(answers), "questions_sha256": manifest.sha256_file(questions),
            "provider": cfg.judge.provider, "model": cfg.judge.model, "cheap_model": cfg.judge.cheap_model,
            "shard_parallelism": cfg.judge.shard_parallelism, "shard_count": len(shards),
            "answers": str(answers.resolve()), "questions": str(questions.resolve()), "shards": shards}


def _fingerprint(plan: dict) -> dict:
    return {k: plan.get(k) for k in FINGERPRINT_KEYS}


def _archive(shard_dir: Path) -> Path:
    stamp = _utc_stamp()
    dest = shard_dir.with_name(f"{shard_dir.name}.{stamp}")
    n = 1
    while dest.exists():
        dest = shard_dir.with_name(f"{shard_dir.name}.{stamp}-{n}")
        n += 1
    shard_dir.rename(dest)
    return dest


def prepare_shards(cfg: RunConfig, run_dir: Path, answers: Path, questions: Path, *,
                   fresh: bool = False) -> tuple[Path, dict, Path | None]:
    """Create or reuse ``<run_dir>/protocol_shards/`` for this plan. Never deletes.

    Returns ``(shard_dir, plan, archived_dir)``. A non-empty directory holding a
    different fingerprint (or no ``shards.json``) raises ``PlanMismatch`` unless
    ``fresh``, in which case it is renamed to ``protocol_shards.<timestamp>/``.
    """
    shard_dir = run_dir / "protocol_shards"
    plan = shard_plan(cfg, answers, questions)
    plan_file = shard_dir / SHARD_PLAN
    archived = None
    if shard_dir.exists() and any(shard_dir.iterdir()):
        existing = None
        if plan_file.exists():
            try:
                existing = json.loads(plan_file.read_text(encoding="utf-8"))
            except ValueError:
                existing = None
        if existing is not None and _fingerprint(existing) == _fingerprint(plan):
            plan = existing
        elif fresh:
            archived = _archive(shard_dir)
            print(f"  archived previous shard state to {archived}", flush=True)
        else:
            old = existing or {}
            diff = {k: old.get(k) for k in FINGERPRINT_KEYS if old.get(k) != plan[k]}
            raise PlanMismatch(f"{shard_dir} holds state for a different plan (differing existing values: {diff}); "
                               "pass fresh=True / --fresh to archive it and start over")
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(answers, "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    pos = 0
    for s in plan["shards"]:
        chunk = "".join(rows[pos:pos + len(s["expected_ids"])])
        pos += len(s["expected_ids"])
        p = shard_dir / s["input"]
        if not p.exists() or p.read_text(encoding="utf-8") != chunk:
            p.write_text(chunk, encoding="utf-8")
    if not plan_file.exists():
        plan["created_at"] = manifest.now_iso()
        plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return shard_dir, plan, archived


def shard_complete(shard_dir: Path, s: dict) -> bool:
    """True when the shard's results file exists, is well-formed and holds exactly its expected ids."""
    res = shard_dir / s["result"]
    if not res.exists():
        return False
    try:
        return set(load_result_ids(res)) == set(s["expected_ids"])
    except CoverageError:
        return False


def _next_attempt(shard_dir: Path, sid: str) -> int:
    return 1 + len(list(shard_dir.glob(f"log_{sid}.attempt*.txt")))


def _load_compute_stats(ov: Overlay):
    """Import ``compute_stats_for_group`` from the overlay's copy of the evaluator."""
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    sys.path.insert(0, str(ov.path))
    try:
        from src.scripts.answer_evaluation.metrics_based_eval import compute_stats_for_group  # type: ignore
    finally:
        sys.path.pop(0)
    return compute_stats_for_group


def merge_shards(ov: Overlay, shard_dir: Path, plan: dict, out: Path, expect_n: int | None) -> dict:
    """Validate every shard against its expected ids and merge with the evaluator's stats code."""
    questions: list[dict] = []
    files: list[Path] = []
    for s in plan["shards"]:
        res = shard_dir / s["result"]
        check_coverage(res, set(s["expected_ids"]), label=f"shard {s['id']} ")
        with open(res, "r", encoding="utf-8") as fh:
            questions.extend(json.load(fh)["questions"])
        files.append(res)
    ids = [q["question_id"] for q in questions]
    expected = {qid for s in plan["shards"] for qid in s["expected_ids"]}
    if len(ids) != len(set(ids)):
        raise CoverageError(f"duplicate question ids across shards: {len(ids) - len(set(ids))}")
    if set(ids) != expected:
        raise CoverageError(f"COVERAGE FAILURE: shard union {len(set(ids))} ids vs {len(expected)} expected")
    if expect_n is not None and len(ids) != expect_n:
        raise CoverageError(f"COVERAGE FAILURE: {len(ids)} judged rows from {len(files)} shards, expected {expect_n}")
    compute_stats_for_group = _load_compute_stats(ov)
    by_type = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)
    merged = {"aggregate_stats": compute_stats_for_group(questions),
              "question_type_stats": {t: compute_stats_for_group(v) for t, v in by_type.items()},
              "questions": questions, "merged_from": [str(f) for f in files]}
    merged["aggregate_stats"]["num_corrected_questions"] = sum(1 for q in questions if q.get("corrected"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return merged


def run_official(cfg: RunConfig, run_dir: Path, erb_root: Path, questions: Path, answers: Path,
                 results: Path | None = None, expect_n: int | None = None, *,
                 allow_unpinned: bool = False, fresh: bool = False) -> Path:
    """Official protocol as concurrent shards run from the overlay; retries are non-destructive."""
    results = results or run_dir / "official_results_protocol.json"
    env = judge_env(cfg)                     # credentials first: no filesystem work without a key
    ov = evaluator_overlay(Path(erb_root), cfg.judge.provider, allow_unpinned=allow_unpinned)
    shard_dir, plan, archived = prepare_shards(cfg, run_dir, answers, questions, fresh=fresh)
    done = [s["id"] for s in plan["shards"] if shard_complete(shard_dir, s)]
    todo = [s for s in plan["shards"] if s["id"] not in done]
    if done:
        print(f"  {len(done)} shards already complete, skipped: {done}", flush=True)
    root = Path(erb_root).resolve()
    procs = []
    attempts: dict[str, int] = {}
    for s in todo:
        sid = s["id"]
        res, upd = shard_dir / s["result"], shard_dir / s["updated"]
        attempts[sid] = _next_attempt(shard_dir, sid)
        log = open(shard_dir / f"log_{sid}.attempt{attempts[sid]}.txt", "w", encoding="utf-8")
        cmd = _cmd(root, (shard_dir / s["input"]).resolve(), questions.resolve(), res.resolve(), upd.resolve(),
                   cfg.judge.shard_parallelism, False, None, resume=res.exists())
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        procs.append((s, subprocess.Popen(cmd, cwd=ov.cwd, env=_overlay_env(env, ov), stdout=log,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), log))
    print(f"  launched {len(procs)} shards x {cfg.judge.shard_parallelism} workers", flush=True)
    failed = []
    for s, p, log in procs:
        p.wait()
        log.close()
        if p.returncode != 0 or not shard_complete(shard_dir, s):
            failed.append(s["id"])
    if failed:
        raise EvaluatorFailed(f"shards failed or incomplete: {failed} (see {shard_dir}/log_<id>.attempt<N>.txt); "
                              "re-run to retry only these; completed shards are kept")
    merged = merge_shards(ov, shard_dir, plan, results, expect_n)
    check_coverage(results, set(answer_ids(answers)), expect_n)
    a = merged["aggregate_stats"]
    print(f"  merged: combined={a['combined_correctness_completeness_score']} correctness={a['average_correctness_pct']} "
          f"completeness={a['average_completeness_pct']} recall={a['average_recall_pct']} "
          f"corrected={a['num_corrected_questions']}", flush=True)
    manifest.write_manifest(run_dir, "judge_official", cfg.to_dict(), [results, shard_dir / SHARD_PLAN],
                            extra={"provider": cfg.judge.provider, "judge_model": cfg.judge.model,
                                   "shards": plan["shard_count"], "shards_skipped": done,
                                   "shards_run": attempts, "archived_shards": str(archived) if archived else None,
                                   "expect_n": expect_n, "evaluator_overlay": ov.manifest_record()})
    return results
