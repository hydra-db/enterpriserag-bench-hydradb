"""Scoring with the benchmark's own evaluator.

The evaluator is ``src/scripts/answer_evaluation/metrics_based_eval.py`` in the
EnterpriseRAG-Bench checkout, invoked as a subprocess from the checkout root
(its document-correction flow resolves corpus paths relative to that root).
This module never reimplements any scoring; it only sets up the environment,
shards the work, and merges shard outputs with the evaluator's own
``compute_stats_for_group``.

Two protocols:

* ``strict`` — ``--no-correction --skip-citation-stripping``. One process.
  Comparable across runs because the gold set never changes.
* ``official`` — the full protocol: citation stripping and the three-judge
  document-correction flow (which may regenerate gold answers). Run as N
  concurrent shards because the evaluator writes its results file only at the
  end of a process.

Two judge providers:

* ``openai`` — the evaluator unmodified: OpenAI Responses API, reasoning effort
  "medium", model ``LLM_MODEL_NAME`` (default gpt-5.4). This is the maintainers'
  path and the one to use for an exact protocol reproduction.
* ``openrouter`` — two files in the checkout (``src/llm/openai_llm.py``,
  ``src/llm/factory.py``) are replaced by ``erb_patches/`` so the judge is called
  through OpenRouter's chat-completions endpoint. This is what the published
  run used. Reasoning-effort parity with the Responses API is not guaranteed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from . import manifest
from .config import RunConfig

EVAL_SCRIPT = Path("src") / "scripts" / "answer_evaluation" / "metrics_based_eval.py"
PATCH_TARGETS = {"src/llm/openai_llm.py": "openai_llm.py", "src/llm/factory.py": "factory.py"}


def apply_patches(erb_root: Path) -> None:
    src_dir = Path(__file__).parent / "erb_patches"
    for rel, fname in PATCH_TARGETS.items():
        shutil.copy2(src_dir / fname, erb_root / rel)


def restore_patches(erb_root: Path) -> None:
    """Restore the evaluator's original LLM files from git."""
    subprocess.run(["git", "checkout", "--", *PATCH_TARGETS.keys()], cwd=erb_root, check=True)


def patches_applied(erb_root: Path) -> bool:
    out = subprocess.run(["git", "status", "--porcelain", *PATCH_TARGETS.keys()], cwd=erb_root,
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def judge_env(cfg: RunConfig) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    # CHEAP_LLM_MODEL_NAME is used by the evaluator's JSON-recovery helper
    # (src/utils/json_recovery.py -> get_cheap_llm). Left unset, it defaults to
    # "gpt-5-mini", which OpenRouter does not resolve; the published run left it
    # unset (see METHODOLOGY section 6).
    if cfg.judge.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        env.update({"LLM_PROVIDER": "openrouter", "LLM_MODEL_NAME": cfg.judge.model,
                    "CHEAP_LLM_MODEL_NAME": cfg.judge.cheap_model, "OPENROUTER_API_KEY": key})
    elif cfg.judge.provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        env.update({"LLM_PROVIDER": "openai", "LLM_MODEL_NAME": cfg.judge.model,
                    "CHEAP_LLM_MODEL_NAME": cfg.judge.cheap_model, "LLM_API_KEY": key})
        env.pop("OPENROUTER_API_KEY", None)
    else:
        raise ValueError(f"unknown judge provider {cfg.judge.provider!r}")
    return env


def _prepare_checkout(cfg: RunConfig, erb_root: Path) -> None:
    if cfg.judge.provider == "openrouter":
        apply_patches(erb_root)
    elif patches_applied(erb_root):
        restore_patches(erb_root)


def _cmd(erb_root: Path, answers: Path, questions: Path, results: Path, updated: Path,
         parallelism: int, strict: bool, question_id: str | None) -> list[str]:
    cmd = [sys.executable, str(EVAL_SCRIPT), "--parallelism", str(parallelism),
           "--answers-file", str(answers), "--questions-file", str(questions),
           "--results-file", str(results), "--updated-questions-file", str(updated),
           "--uuid-index-cache-file", str(erb_root / "generated_data" / "uuid_index.json")]
    if strict:
        cmd += ["--no-correction", "--skip-citation-stripping"]
    if question_id:
        cmd += ["--question-id", question_id]
    return cmd


def run_strict(cfg: RunConfig, run_dir: Path, erb_root: Path, questions: Path, answers: Path,
               results: Path | None = None, question_id: str | None = None) -> Path:
    results = results or run_dir / "official_results_strict.json"
    _prepare_checkout(cfg, erb_root)
    env = judge_env(cfg)
    env["PYTHONPATH"] = str(erb_root)
    cmd = _cmd(erb_root, answers.resolve(), questions.resolve(), results.resolve(),
               run_dir / "questions_updated_strict.jsonl", cfg.judge.strict_parallelism, True, question_id)
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=erb_root, env=env, check=True, stdin=subprocess.DEVNULL)
    manifest.write_manifest(run_dir, "judge_strict", cfg.to_dict(), [results],
                            extra={"provider": cfg.judge.provider, "judge_model": cfg.judge.model,
                                   "question_id": question_id})
    return results


def _shard(answers: Path, shard_dir: Path, n_shards: int) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old in shard_dir.glob("*"):
        old.unlink()
    with open(answers, "r", encoding="utf-8") as f:
        rows = [line for line in f if line.strip()]
    per = max(1, -(-len(rows) // n_shards))
    files = []
    for i in range(0, len(rows), per):
        p = shard_dir / f"answers_{i // per:02d}.jsonl"
        p.write_text("".join(rows[i:i + per]), encoding="utf-8")
        files.append(p)
    return files


def merge_shards(erb_root: Path, shard_results: list[Path], out: Path, expect_n: int | None) -> dict:
    sys.path.insert(0, str(erb_root))
    try:
        from src.scripts.answer_evaluation.metrics_based_eval import compute_stats_for_group  # type: ignore
    finally:
        sys.path.pop(0)
    questions: list[dict] = []
    for f in shard_results:
        with open(f, "r", encoding="utf-8") as fh:
            questions.extend(json.load(fh).get("questions", []))
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate question ids across shards: {len(ids) - len(set(ids))}")
    if expect_n is not None and len(ids) != expect_n:
        raise SystemExit(f"COVERAGE FAILURE: {len(ids)} judged rows from {len(shard_results)} shards, expected {expect_n}")
    by_type = defaultdict(list)
    for q in questions:
        by_type[q["question_type"]].append(q)
    merged = {"aggregate_stats": compute_stats_for_group(questions),
              "question_type_stats": {t: compute_stats_for_group(v) for t, v in by_type.items()},
              "questions": questions, "merged_from": [str(f) for f in shard_results]}
    merged["aggregate_stats"]["num_corrected_questions"] = sum(1 for q in questions if q.get("corrected"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return merged


def run_official(cfg: RunConfig, run_dir: Path, erb_root: Path, questions: Path, answers: Path,
                 results: Path | None = None, expect_n: int | None = None) -> Path:
    results = results or run_dir / "official_results_protocol.json"
    _prepare_checkout(cfg, erb_root)
    env = judge_env(cfg)
    env["PYTHONPATH"] = str(erb_root)
    shard_dir = run_dir / "protocol_shards"
    shards = _shard(answers, shard_dir, cfg.judge.shards)
    procs = []
    for shard in shards:
        sid = shard.stem.replace("answers_", "")
        res = shard_dir / f"results_{sid}.json"
        upd = shard_dir / f"questions_updated_{sid}.jsonl"
        log = open(shard_dir / f"log_{sid}.txt", "w", encoding="utf-8")
        cmd = _cmd(erb_root, shard.resolve(), questions.resolve(), res.resolve(), upd.resolve(),
                   cfg.judge.shard_parallelism, False, None)
        procs.append((sid, res, subprocess.Popen(cmd, cwd=erb_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                  stdin=subprocess.DEVNULL), log))
    print(f"  launched {len(procs)} shards x {cfg.judge.shard_parallelism} workers", flush=True)
    failed = []
    for sid, res, p, log in procs:
        p.wait()
        log.close()
        if p.returncode != 0 or not res.exists():
            failed.append(sid)
    if failed:
        raise SystemExit(f"shards failed: {failed} (see {shard_dir}/log_*.txt); re-run to retry")
    merged = merge_shards(erb_root, [r for _, r, _, _ in procs], results, expect_n)
    a = merged["aggregate_stats"]
    print(f"  merged: combined={a['combined_correctness_completeness_score']} correctness={a['average_correctness_pct']} "
          f"completeness={a['average_completeness_pct']} recall={a['average_recall_pct']} "
          f"corrected={a['num_corrected_questions']}", flush=True)
    manifest.write_manifest(run_dir, "judge_official", cfg.to_dict(), [results],
                            extra={"provider": cfg.judge.provider, "judge_model": cfg.judge.model,
                                   "shards": len(shards)})
    return results
