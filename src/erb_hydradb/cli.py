"""``erb-hydradb`` command line.

    setup      clone EnterpriseRAG-Bench at the pinned commit, verify questions.jsonl
    download   build data/documents.sqlite (from the checkout, or from Hugging Face)
    ingest     load the corpus into a HydraDB collection as typed app sources
    generate   retrieve + answer (or regenerate from published contexts)
    judge      score answers with the benchmark's evaluator (strict | official)
    report     write RESULTS.md for a run directory
    compare    paired comparison of two results files
    recall     paired document-recall comparison of two retrieval runs
    verify     checksums + recomputed statistics for a run directory
    stats      recompute the evaluator's aggregates from a results file

Reproduction levels are described in REPRODUCE.md.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import analysis, manifest, paths
from .config import RunConfig


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _checkout_commit(root: Path) -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return out.stdout.strip()


def _erb_root(arg: str | None, allow_unpinned: bool = False) -> Path:
    """The benchmark checkout. Refuses a checkout that is not at the pinned
    commit unless --allow-unpinned is given (and then says so loudly)."""
    root = Path(arg).expanduser().resolve() if arg else paths.erb_repo()
    if not root or not (root / "questions.jsonl").exists():
        _die("EnterpriseRAG-Bench checkout not found; run `erb-hydradb setup` or set ERB_REPO")
    head = _checkout_commit(root)
    if head != paths.ERB_COMMIT:
        msg = f"checkout {root} is at {head[:12]}, not the pinned {paths.ERB_COMMIT[:12]}"
        if not allow_unpinned:
            _die(msg + " (run `erb-hydradb setup`, or pass --allow-unpinned to proceed anyway)")
        print(f"WARNING: {msg}; results will not be comparable to the published run", file=sys.stderr)
    return root


def _questions(arg: str | None, allow_unpinned: bool = False) -> Path:
    """The official questions file: --questions, else data/questions.jsonl (from
    setup), else the copy shipped in tests/data. Its sha256 must match the pin."""
    candidates = [Path(arg).expanduser().resolve()] if arg else [paths.questions_path(),
                                                                  paths.repo_root() / "tests" / "data" / "questions.jsonl"]
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        _die("questions.jsonl not found; run `erb-hydradb setup`")
    digest = _sha(p)
    if digest != paths.QUESTIONS_SHA256:
        msg = f"{p} sha256 {digest[:12]}... does not match the pinned questions file"
        if not allow_unpinned:
            _die(msg + " (pass --allow-unpinned to proceed anyway)")
        print(f"WARNING: {msg}", file=sys.stderr)
    return p


def _sha(p: Path) -> str:
    return manifest.sha256_file(p)


# ------------------------------------------------------------------ setup --

def cmd_setup(a: argparse.Namespace) -> int:
    root = Path(a.erb_dir).expanduser().resolve() if a.erb_dir else paths.repo_root() / "EnterpriseRAG-Bench"
    if not root.exists():
        print(f"cloning {paths.ERB_REPO_URL} -> {root}")
        subprocess.run(["git", "clone", "--filter=blob:none", paths.ERB_REPO_URL, str(root)], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    if head != paths.ERB_COMMIT:
        print(f"checking out pinned commit {paths.ERB_COMMIT}")
        subprocess.run(["git", "fetch", "--depth", "1", "origin", paths.ERB_COMMIT], cwd=root, check=False)
        subprocess.run(["git", "checkout", "-q", paths.ERB_COMMIT], cwd=root, check=True)
    q = root / "questions.jsonl"
    digest = _sha(q)
    if digest != paths.QUESTIONS_SHA256:
        _die(f"questions.jsonl sha256 {digest} != pinned {paths.QUESTIONS_SHA256}; refusing to continue")
    dest = paths.questions_path()
    shutil.copy2(q, dest)
    print(f"ok: checkout at {paths.ERB_COMMIT[:12]}, questions.jsonl verified and copied to {dest}")
    print(f"set ERB_REPO={root} in .env (or export it) so later commands find the checkout")
    return 0


# --------------------------------------------------------------- download --

def cmd_download(a: argparse.Namespace) -> int:
    from . import corpus
    db = paths.documents_db_path()
    if a.from_checkout:
        root = _erb_root(a.erb)
        n = corpus.build_from_repo(root, db, limit=a.limit)
    else:
        n = corpus.build_from_hf(db, limit=a.limit)
    print(f"ok: {n} documents in {db}")
    return 0


# ----------------------------------------------------------------- ingest --

def cmd_ingest(a: argparse.Namespace) -> int:
    from . import ingest
    cfg = RunConfig.load(a.config)
    key = os.environ.get("HYDRADB_API_KEY")
    if not key and not a.dry_run:
        _die("HYDRADB_API_KEY is not set")
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ingest.run_ingest(cfg, paths.documents_db_path(), run_dir / "ingest_state.json", infer=not a.no_infer,
                      resume=a.resume, dry_run=a.dry_run, limit=a.limit, source_types=a.source_types,
                      batch_size=a.batch_size, batch_sleep=a.batch_sleep, provision=not a.no_provision,
                      wait=not a.no_wait, api_key=key)
    return 0


# --------------------------------------------------------------- generate --

def cmd_generate(a: argparse.Namespace) -> int:
    from . import generate, hydrate, llm
    cfg = RunConfig.load(a.config)
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    questions = generate.select_questions(generate.load_questions(_questions(a.questions, a.allow_unpinned)),
                                          a.targets, a.n)
    if not a.retrieval_only and not llm.provider_available(cfg.generation.provider):
        _die(f"no key for generation provider {cfg.generation.provider!r}")
    from_contexts = None
    store = None
    if a.from_contexts:
        src = Path(a.from_contexts)
        if src.suffix == ".gz":
            plain = run_dir / "published_contexts.jsonl"
            with gzip.open(src, "rt", encoding="utf-8") as fi, open(plain, "w", encoding="utf-8") as fo:
                shutil.copyfileobj(fi, fo)
            src = plain
        from_contexts = src
    else:
        erb = paths.erb_repo()
        if erb and not paths.documents_db_path().exists():
            _erb_root(str(erb), a.allow_unpinned)   # enforce the pin when hydrating from the checkout
        store = hydrate.DocumentStore(paths.documents_db_path(), erb)
        print(f"document store: {store.backend}")
    asyncio.run(generate.run_generate(cfg, run_dir, questions, api_key=os.environ.get("HYDRADB_API_KEY"),
                                      store=store, retrieval_only=a.retrieval_only, from_contexts=from_contexts,
                                      force_resume=a.force_resume))
    return 0


# ------------------------------------------------------------------ judge --

def cmd_judge(a: argparse.Namespace) -> int:
    from . import judge
    cfg = RunConfig.load(a.config)
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    answers = Path(a.answers).resolve() if a.answers else run_dir / "answers.jsonl"
    if not answers.exists():
        _die(f"answers file not found: {answers}")
    root = _erb_root(a.erb, a.allow_unpinned)
    q = _questions(a.questions, a.allow_unpinned)
    results = Path(a.results).resolve() if a.results else None
    if a.protocol == "strict":
        out = judge.run_strict(cfg, run_dir, root, q, answers, results, question_id=a.question_id)
    else:
        if a.question_id:
            _die("--question-id is only supported with --protocol strict")
        out = judge.run_official(cfg, run_dir, root, q, answers, results, expect_n=a.expect_n)
    rc = analysis.recompute_stats(out)
    agg = rc["recomputed"]["aggregate_stats"]
    print(f"\n{a.protocol}: n={rc['n']} combined={agg['combined_correctness_completeness_score']} "
          f"correctness={agg['average_correctness_pct']} completeness={agg['average_completeness_pct']} "
          f"recall={agg['average_recall_pct']} invalid_extra={agg['average_invalid_extra_docs']}")
    if rc["diffs"]:
        print(f"WARNING recomputation mismatches: {rc['diffs']}")
    print(f"results -> {out}")
    return 0


# ----------------------------------------------------------------- report --

def cmd_report(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir).resolve()
    lb = None
    if a.leaderboard:
        with open(a.leaderboard, "r", encoding="utf-8") as f:
            lb = json.load(f)
    title = a.title or f"HydraDB on EnterpriseRAG-Bench: {run_dir.name}"
    md = analysis.render_results_md(run_dir, _questions(a.questions, True), title, lb)
    out = Path(a.out) if a.out else run_dir / "RESULTS.md"
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------- compare --

def cmd_compare(a: argparse.Namespace) -> int:
    rep = analysis.compare(a.a, a.b, _questions(a.questions, True), ids_file=a.ids, expect_n=a.expect_n,
                           a_checkpoint=a.checkpoint_a, a_answers=a.answers_a,
                           b_checkpoint=a.checkpoint_b, b_answers=a.answers_b)
    analysis.print_compare(rep)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=1)
        print(f"wrote {a.json}")
    return 0


def cmd_recall(a: argparse.Namespace) -> int:
    ra, sa = analysis.load_retrieval(a.a, None)
    rb, sb = analysis.load_retrieval(a.b, None)
    rep = analysis.recall_compare(ra, rb, _questions(a.questions, True), ctx=a.ctx, ids_file=a.ids)
    print(f"n={rep['n']} questions with gold docs in both runs (A: {sa}; B: {sb})")
    print(f"  {'k':>4} {'A':>8} {'B':>8} {'delta':>8} {'95% CI':>18}")
    for r in rep["rows"]:
        print(f"  {r['k']:>4} {r['a']:>8} {r['b']:>8} {r['delta_mean']:>+8} {str(r['ci95']):>18}")
    print(f"  full gold coverage inside the first {rep['ctx']} docs: gained {len(rep['full_coverage_gained'])}, "
          f"lost {len(rep['full_coverage_lost'])}")
    return 0


# ----------------------------------------------------------------- verify --

def cmd_verify(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir).resolve()
    ok = True
    problems = manifest.verify_sums(run_dir)
    if problems:
        ok = False
        print("checksums: FAIL"); [print("  " + p) for p in problems]
    else:
        print("checksums: ok")
    for name in ("official_results_strict.json", "official_results_protocol.json"):
        p = run_dir / name
        if p.exists():
            rc = analysis.recompute_stats(p)
            agg = rc["recomputed"]["aggregate_stats"]
            status = "ok" if not rc["diffs"] else f"MISMATCH {rc['diffs']}"
            ok = ok and not rc["diffs"]
            print(f"{name}: n={rc['n']} combined={agg['combined_correctness_completeness_score']} "
                  f"correctness={agg['average_correctness_pct']} completeness={agg['average_completeness_pct']} "
                  f"recall={agg['average_recall_pct']} -> {status}")
    root = paths.erb_repo()
    if root and (root / "questions.jsonl").exists():
        head = _checkout_commit(root)
        qh = _sha(root / "questions.jsonl")
        print(f"checkout commit: {head[:12]} ({'ok' if head == paths.ERB_COMMIT else 'DIFFERS from pinned ' + paths.ERB_COMMIT[:12]})")
        print(f"questions.jsonl sha256: {'ok' if qh == paths.QUESTIONS_SHA256 else 'DIFFERS'}")
        ok = ok and head == paths.ERB_COMMIT and qh == paths.QUESTIONS_SHA256
    else:
        print("checkout: not present (run `erb-hydradb setup` to verify the pinned commit)")
    if a.contexts:
        res = _context_equivalence(run_dir, root)
        if res is None:
            print("context equivalence: skipped (needs data/documents.sqlite or a checkout, and contexts.jsonl.gz)")
        else:
            print(f"context equivalence: {res['reproduced']}/{res['checked']} published context hashes reproduced "
                  f"from the retrieval order and the corpus ({res['backend']})")
            ok = ok and res["reproduced"] == res["checked"]
            if a.log:
                with open(a.log, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=1)
                print(f"wrote {a.log}")
    print("VERIFIED" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def _context_equivalence(run_dir: Path, root: Path | None) -> dict | None:
    """Rebuild every published context from the saved retrieval order and the
    corpus, and compare sha256 with the checkpoint. Offline; no LLM, no HydraDB."""
    from . import hydrate
    ck_path, ctx_path = run_dir / "gen_checkpoint.json", run_dir / "contexts.jsonl.gz"
    if not ck_path.exists() or not ctx_path.exists():
        return None
    db = paths.documents_db_path()
    if not db.exists() and not (root and (root / "generated_data" / "uuid_index.json").exists()):
        return None
    try:
        store = hydrate.DocumentStore(db if db.exists() else None, root)
    except FileNotFoundError:
        return None
    cfg = RunConfig.load(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else RunConfig()
    with open(ck_path, "r", encoding="utf-8") as f:
        rows = [r for r in json.load(f) if "question_id" in r]
    published: dict[str, str] = {}
    with gzip.open(ctx_path, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            published[c["question_id"]] = c["context_sha256"]
    checked = reproduced = 0
    mismatched: list[str] = []
    for r in rows:
        if r["question_id"] not in published:
            continue
        ctx = hydrate.build_context(r["retrieved_doc_ids"], [], store, cfg.generation.docs_in_context,
                                    cfg.generation.max_context_chars)
        checked += 1
        if ctx["context_sha256"] == published[r["question_id"]] == r.get("context_sha256"):
            reproduced += 1
        else:
            mismatched.append(r["question_id"])
    return {"checked": checked, "reproduced": reproduced, "mismatched": mismatched, "backend": store.backend,
            "docs_in_context": cfg.generation.docs_in_context, "max_context_chars": cfg.generation.max_context_chars,
            "erb_commit": _checkout_commit(root) if root else None, "finished_at": manifest.now_iso()}


def cmd_stats(a: argparse.Namespace) -> int:
    rc = analysis.recompute_stats(a.results)
    print(json.dumps(rc["recomputed"], indent=1))
    if rc["diffs"]:
        print(f"MISMATCH vs file: {rc['diffs']}")
        return 1
    print(f"ok: n={rc['n']}, aggregates in the file match the recomputation")
    return 0


# ------------------------------------------------------------------- main --

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="erb-hydradb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup"); s.add_argument("--erb-dir"); s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("download"); s.add_argument("--from-checkout", action="store_true")
    s.add_argument("--erb"); s.add_argument("--limit", type=int); s.set_defaults(fn=cmd_download)

    s = sub.add_parser("ingest"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--no-infer", action="store_true"); s.add_argument("--resume", action="store_true")
    s.add_argument("--dry-run", action="store_true"); s.add_argument("--limit", type=int)
    s.add_argument("--source-types", nargs="+"); s.add_argument("--batch-size", type=int, default=80)
    s.add_argument("--batch-sleep", type=float, default=1.0); s.add_argument("--no-provision", action="store_true")
    s.add_argument("--no-wait", action="store_true"); s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("generate"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--questions"); s.add_argument("--targets"); s.add_argument("--n", type=int)
    s.add_argument("--retrieval-only", action="store_true"); s.add_argument("--from-contexts")
    s.add_argument("--force-resume", action="store_true"); s.add_argument("--allow-unpinned", action="store_true")
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("judge"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--protocol", choices=["strict", "official"], default="strict")
    s.add_argument("--answers"); s.add_argument("--results"); s.add_argument("--erb"); s.add_argument("--questions")
    s.add_argument("--question-id"); s.add_argument("--expect-n", type=int, default=None)
    s.add_argument("--allow-unpinned", action="store_true"); s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("report"); s.add_argument("--run-dir", required=True); s.add_argument("--title")
    s.add_argument("--questions"); s.add_argument("--leaderboard"); s.add_argument("--out"); s.set_defaults(fn=cmd_report)

    s = sub.add_parser("compare"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s.add_argument("--questions"); s.add_argument("--ids"); s.add_argument("--expect-n", type=int)
    s.add_argument("--checkpoint-a"); s.add_argument("--answers-a"); s.add_argument("--checkpoint-b"); s.add_argument("--answers-b")
    s.add_argument("--json"); s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("recall"); s.add_argument("--a", required=True, help="gen_checkpoint.json of run A")
    s.add_argument("--b", required=True); s.add_argument("--questions"); s.add_argument("--ctx", type=int, default=12)
    s.add_argument("--ids"); s.set_defaults(fn=cmd_recall)

    s = sub.add_parser("verify"); s.add_argument("--run-dir", required=True)
    s.add_argument("--contexts", action="store_true", help="also rebuild every context from the corpus and compare hashes")
    s.add_argument("--log", help="write the context-equivalence result to this JSON file"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("stats"); s.add_argument("--results", required=True); s.set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
