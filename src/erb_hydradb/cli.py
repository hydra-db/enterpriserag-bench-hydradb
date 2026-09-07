"""``erb-hydradb`` command line.

    doctor     preflight: environment, keys by stage, pins, corpus, contexts (no cost)
    setup      clone EnterpriseRAG-Bench at the pinned commit, verify questions.jsonl
    download   build data/documents.sqlite (from the checkout, or from Hugging Face)
    ingest     load the corpus into a HydraDB collection as typed app sources
    generate   retrieve + answer (or regenerate from published contexts)
    judge      score answers with the benchmark's evaluator (strict | official)
    report     write RESULTS.md for a run directory
    compare    paired comparison of two results files
    recall     paired document-recall comparison of two retrieval runs
    verify     validate a run directory (files, schema, coverage, statistics, contexts)
    stats      recompute the evaluator's aggregates from a results file
    audit-corrections   check the official protocol's gold corrections against the published record
    inspect    everything about one question: retrieval, context, answer, judgments, correction

Exit codes: 0 ok · 1 usage / input problem · 2 the run completed with failures
or incomplete coverage (see the output; use --allow-partial to accept).
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

from . import analysis, manifest, paths, validate
from .config import RunConfig

EXIT_INCOMPLETE = 2


def _die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


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
    candidates = [Path(arg).expanduser().resolve()] if arg else [
        paths.questions_path(), paths.repo_root() / "tests" / "data" / "questions.jsonl"]
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


def _store(allow_unpinned: bool = False):
    from . import hydrate
    erb = paths.erb_repo()
    if erb and not paths.documents_db_path().exists():
        _erb_root(str(erb), allow_unpinned)
    return hydrate.DocumentStore(paths.documents_db_path(), erb)


# ----------------------------------------------------------------- doctor --

def cmd_doctor(a: argparse.Namespace) -> int:
    """Preflight without spending anything."""
    import platform
    ok = True

    def line(status: str, what: str, detail: str = ""):
        nonlocal ok
        if status == "FAIL":
            ok = False
        print(f"{status:5s} {what}" + (f": {detail}" if detail else ""))

    line("ok", "python", platform.python_version())
    for mod in ("httpx", "openai", "yaml", "pydantic", "tiktoken", "anthropic"):
        try:
            __import__(mod)
            line("ok", f"import {mod}")
        except ImportError:
            line("FAIL", f"import {mod}", "run: uv pip install -e '.[dev]'")
    for var, stage in (("OPENROUTER_API_KEY", "generate/judge via OpenRouter (levels 1-4)"),
                       ("OPENAI_API_KEY", "generate/judge via OpenAI (alternative)"),
                       ("HYDRADB_API_KEY", "generate against HydraDB / ingest (levels 3-4)")):
        line("ok" if os.environ.get(var) else "note", var, ("set" if os.environ.get(var) else "not set") + f" — {stage}")
    q = next((c for c in (paths.questions_path(), paths.repo_root() / "tests" / "data" / "questions.jsonl") if c.exists()), None)
    if q:
        line("ok" if _sha(q) == paths.QUESTIONS_SHA256 else "FAIL", "questions.jsonl", f"{q} sha256 {'matches' if _sha(q) == paths.QUESTIONS_SHA256 else 'DOES NOT MATCH'} the pin")
    else:
        line("FAIL", "questions.jsonl", "not found (run setup)")
    root = paths.erb_repo()
    if root and (root / "questions.jsonl").exists():
        head = _checkout_commit(root)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=root, capture_output=True, text=True).stdout.strip()
        line("ok" if head == paths.ERB_COMMIT else "FAIL", "benchmark checkout", f"{root} at {head[:12]}" + ("" if head == paths.ERB_COMMIT else f" (pinned {paths.ERB_COMMIT[:12]})"))
        line("ok" if not dirty else "note", "checkout src/ clean", "yes" if not dirty else f"modified: {dirty.splitlines()[:3]}")
    else:
        line("note", "benchmark checkout", "not present — needed for judge and level 3+ (run setup)")
    db = paths.documents_db_path()
    if db.exists():
        try:
            from . import corpus
            ident = corpus.corpus_identity(db)
            line("ok", "documents.sqlite", f"{ident}")
        except Exception as exc:  # noqa: BLE001
            line("note", "documents.sqlite", f"present, identity unreadable ({exc})")
    else:
        line("note", "documents.sqlite", "not built — hydration falls back to the checkout; run download")
    ctx = paths.artifacts_dir() / "run-2026-09-04" / "contexts.jsonl.gz"
    line("ok" if ctx.exists() else "FAIL", "published contexts", str(ctx))
    try:
        free = shutil.disk_usage(paths.repo_root()).free / 2**30
        line("ok" if free > 8 else "note", "disk free", f"{free:.1f} GB (the checkout needs ~5 GB)")
    except OSError:
        pass
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


# ------------------------------------------------------------------ setup --

def cmd_setup(a: argparse.Namespace) -> int:
    root = Path(a.erb_dir).expanduser().resolve() if a.erb_dir else paths.repo_root() / "EnterpriseRAG-Bench"
    if not root.exists():
        print(f"cloning {paths.ERB_REPO_URL} -> {root} (about 5 GB on disk)")
        subprocess.run(["git", "clone", "--filter=blob:none", paths.ERB_REPO_URL, str(root)], check=True)
    head = _checkout_commit(root)
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
    print(f"ok: {n} documents in {db}; identity {corpus.corpus_identity(db)}")
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
    incomplete = getattr(ingest, "IngestIncomplete", RuntimeError)
    try:
        ingest.run_ingest(cfg, paths.documents_db_path(), run_dir / "ingest_state.json", infer=not a.no_infer,
                          resume=a.resume, dry_run=a.dry_run, limit=a.limit, source_types=a.source_types,
                          batch_size=a.batch_size, batch_sleep=a.batch_sleep, provision=not a.no_provision,
                          wait=not a.no_wait, api_key=key, retry_failed=a.retry_failed,
                          accept_failures=a.accept_failures)
    except incomplete as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    return 0


# --------------------------------------------------------------- generate --

def cmd_generate(a: argparse.Namespace) -> int:
    from . import generate, llm
    cfg = RunConfig.load(a.config)
    run_dir = Path(a.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    qpath = _questions(a.questions, a.allow_unpinned)
    questions = generate.select_questions(generate.load_questions(qpath), a.targets, a.n)
    if not questions:
        _die("no questions selected")
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
        store = _store(a.allow_unpinned)
        print(f"document store: {store.backend}")
    # validate any existing checkpoint against this run's identity BEFORE writing anything
    store_identity = None
    if store is not None:
        store_identity = dict(store.identity)
        erb = paths.erb_repo()
        if erb and (erb / "questions.jsonl").exists():
            store_identity["checkout_commit"] = _checkout_commit(erb)
    identity = generate.run_identity(cfg, questions_path=qpath, from_contexts=from_contexts,
                                     store_identity=store_identity)
    rows, _ = generate.load_checkpoint(run_dir / "gen_checkpoint.json", identity, a.force_resume)
    generate.check_selection(run_dir, rows, [q["question_id"] for q in questions],
                             retrieval_only=a.retrieval_only, force=a.force_resume)
    cfg.save(run_dir / "config.yaml")
    _, incomplete = asyncio.run(generate.run_generate(
        cfg, run_dir, questions, api_key=os.environ.get("HYDRADB_API_KEY"), store=store, questions_path=qpath,
        retrieval_only=a.retrieval_only, from_contexts=from_contexts, force_resume=a.force_resume,
        store_identity=store_identity))
    if incomplete:
        print(f"INCOMPLETE: {incomplete}. answers.jsonl is partial; re-run to retry errors"
              + ("" if a.allow_partial else " (exit 2; --allow-partial to accept)"), file=sys.stderr)
        return 0 if a.allow_partial else EXIT_INCOMPLETE
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
    expected = {r["question_id"] for r in validate.read_jsonl_rows(answers)}
    if a.expect_n is not None and len(expected) != a.expect_n:
        _die(f"answers file has {len(expected)} unique question ids, --expect-n says {a.expect_n}")
    kw = {"allow_unpinned": a.allow_unpinned, "fresh": a.fresh}
    try:
        if a.protocol == "strict":
            out = judge.run_strict(cfg, run_dir, root, q, answers, results, question_id=a.question_id,
                                   expect_n=None if a.question_id else a.expect_n, **kw)
        else:
            if a.question_id:
                _die("--question-id is only supported with --protocol strict")
            out = judge.run_official(cfg, run_dir, root, q, answers, results, expect_n=a.expect_n, **kw)
    except judge.JudgeError as exc:
        print(f"JUDGE FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return EXIT_INCOMPLETE
    problems, data = validate.check_results_file(Path(out), None if a.question_id else expected)
    rc = analysis.recompute_stats(out)
    agg = rc["recomputed"]["aggregate_stats"]
    print(f"\n{a.protocol}: n={rc['n']} combined={agg['combined_correctness_completeness_score']} "
          f"correctness={agg['average_correctness_pct']} completeness={agg['average_completeness_pct']} "
          f"recall={agg['average_recall_pct']} invalid_extra={agg['average_invalid_extra_docs']}")
    print(f"results -> {out}")
    if problems:
        print("RESULTS FAILED VALIDATION:\n  " + "\n  ".join(problems[:10]), file=sys.stderr)
        return EXIT_INCOMPLETE
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
    qpath = None
    try:
        qpath = _questions(a.questions, True)
    except SystemExit:
        pass
    required = validate.REQUIRED_PUBLISHED if not a.minimal else ("SHA256SUMS",)
    if a.minimal:
        qpath = None   # a partial run (pilot): coverage is checked by cross-file equality only
    store = None
    cfg = RunConfig.load(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else RunConfig()
    if a.contexts:
        try:
            store = _store(True)
        except (FileNotFoundError, SystemExit):
            store = None
    report = validate.validate_run(run_dir, qpath, required=required, contexts=a.contexts, store=store,
                                   docs_in_context=cfg.generation.docs_in_context,
                                   max_context_chars=cfg.generation.max_context_chars)
    root = paths.erb_repo()
    if root and (root / "questions.jsonl").exists():
        head = _checkout_commit(root)
        report["checks"]["pinned_checkout"] = {
            "status": "ok" if head == paths.ERB_COMMIT else "fail",
            "problems": [] if head == paths.ERB_COMMIT else [f"checkout at {head[:12]}, pinned {paths.ERB_COMMIT[:12]}"],
            "commit": head[:12]}
    else:
        report["checks"]["pinned_checkout"] = {"status": "incomplete", "problems": ["no checkout present (run setup)"]}
    if report["checks"]["pinned_checkout"]["status"] == "fail":
        report["ok"] = False
    print(validate.format_report(report))
    if a.log:
        with open(a.log, "w", encoding="utf-8") as f:
            json.dump({**report, "run_dir": str(run_dir), "finished_at": manifest.now_iso()}, f, indent=1)
        print(f"wrote {a.log}")
    return 0 if report["ok"] else 1


def cmd_stats(a: argparse.Namespace) -> int:
    problems, _ = validate.check_results_file(Path(a.results), None)
    rc = analysis.recompute_stats(a.results)
    print(json.dumps(rc["recomputed"], indent=1))
    if problems:
        print("MISMATCH vs file:\n  " + "\n  ".join(problems[:10]))
        return 1
    print(f"ok: n={rc['n']}, every aggregate and per-category statistic in the file matches the recomputation")
    return 0


# ------------------------------------------------------- corrections audit --

def cmd_audit_corrections(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir).resolve()
    qpath = _questions(a.questions, True)
    res_path = run_dir / "official_results_protocol.json"
    corr_path = run_dir / "corrections.jsonl"
    if not res_path.exists() or not corr_path.exists():
        _die("needs official_results_protocol.json and corrections.jsonl in the run directory")
    originals = {r["question_id"]: r for r in validate.read_jsonl_rows(qpath)}
    results = {r["question_id"]: r for r in analysis.load_results(res_path)["questions"]}
    records = validate.read_jsonl_rows(corr_path)
    problems = validate.check_ids(records, None, "corrections")
    by_id = {r["question_id"]: r for r in records}
    flagged = {q for q, r in results.items() if r.get("corrected")}
    # the evaluator flags a row `corrected` only when the gold set or gold answer changed;
    # an update that only added valid_doc_ids is still a record (it changes invalid_extra_docs)
    changed = {q for q, r in by_id.items() if r.get("doc_set_changed") or r.get("gold_answer_changed")}
    for q in sorted(flagged - set(by_id)):
        problems.append(f"{q} is flagged corrected but has no correction record")
    for q in sorted(changed - flagged):
        problems.append(f"{q} has a gold-changing correction record but is not flagged corrected")
    for q in sorted(set(by_id) - changed):
        if not by_id[q].get("after", {}).get("valid_doc_ids"):
            problems.append(f"{q} has a correction record that changes nothing")
    answers = {r["question_id"]: r for r in validate.read_jsonl_rows(run_dir / "answers.jsonl")} \
        if (run_dir / "answers.jsonl").exists() else {}
    for q, rec in by_id.items():
        o = originals.get(q)
        if o is None:
            problems.append(f"{q}: not in the questions file")
            continue
        before, after = rec.get("before", {}), rec.get("after", {})
        # the declared change flags must follow from before/after
        if bool(rec.get("doc_set_changed")) != (set(before.get("expected_doc_ids") or []) != set(after.get("expected_doc_ids") or [])):
            problems.append(f"{q}: doc_set_changed flag does not match before/after")
        if bool(rec.get("gold_answer_changed")) != (before.get("gold_answer") != after.get("gold_answer")):
            problems.append(f"{q}: gold_answer_changed flag does not match before/after")
        # the corrected gold set must reproduce the results row's document recall
        ans_row = answers.get(q)
        if ans_row is not None:
            rec_pct, _ = validate.retrieval_metrics(ans_row["document_ids"], set(after.get("expected_doc_ids") or []),
                                                    set(after.get("valid_doc_ids") or []))
            got = results[q].get("document_recall_pct")
            if rec_pct is not None and got is not None and abs(rec_pct - float(got)) > 0.011:
                problems.append(f"{q}: corrected gold set gives recall {rec_pct} but the results row says {got}")
        if (before.get("expected_doc_ids") != o["expected_doc_ids"] or before.get("gold_answer") != o["gold_answer"]
                or before.get("answer_facts") != o["answer_facts"]):
            problems.append(f"{q}: 'before' does not equal the pinned original question")
        if not after.get("expected_doc_ids") or not after.get("gold_answer") or not isinstance(after.get("answer_facts"), list):
            problems.append(f"{q}: 'after' record incomplete")
        if not rec.get("update_reasons"):
            problems.append(f"{q}: no update_reasons")
    summary = {
        "flagged_corrected": len(flagged), "records": len(by_id),
        "doc_set_changed": sum(1 for r in by_id.values() if r.get("doc_set_changed")),
        "gold_answer_changed": sum(1 for r in by_id.values() if r.get("gold_answer_changed")),
        "uncorrected_questions": len(results) - len(flagged),
        "problems": problems, "ok": not problems, "finished_at": manifest.now_iso(),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1))
    for p in problems:
        print("  - " + p)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print(f"wrote {a.out}")
    return 0 if not problems else 1


# ---------------------------------------------------------------- inspect --

def cmd_inspect(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir).resolve()
    qid = a.question_id
    q = {r["question_id"]: r for r in validate.read_jsonl_rows(_questions(a.questions, True))}.get(qid)
    if q is None:
        _die(f"{qid} not in the questions file")
    out: dict = {"question": q}
    ck = run_dir / "gen_checkpoint.json"
    if ck.exists():
        _, rows = validate.read_checkpoint(ck)
        out["checkpoint"] = next((r for r in rows if r["question_id"] == qid), None)
    ans = run_dir / "answers.jsonl"
    if ans.exists():
        out["answer"] = next((r for r in validate.read_jsonl_rows(ans) if r["question_id"] == qid), None)
    for name in ("official_results_strict.json", "official_results_protocol.json"):
        p = run_dir / name
        if p.exists():
            out[name] = next((r for r in analysis.load_results(p)["questions"] if r["question_id"] == qid), None)
    corr = run_dir / "corrections.jsonl"
    if corr.exists():
        out["correction"] = next((r for r in validate.read_jsonl_rows(corr) if r["question_id"] == qid), None)
    ctx = run_dir / "contexts.jsonl.gz"
    if ctx.exists() and a.context:
        out["context"] = next((r for r in validate.read_jsonl_rows(ctx) if r["question_id"] == qid), None)
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 0
    print(f"# {qid} [{q['question_type']}] sources={q.get('source_types')}\n\nQ: {q['question']}\n")
    print(f"gold answer: {q.get('gold_answer')}\ngold docs: {q.get('expected_doc_ids')}\n")
    if out.get("checkpoint"):
        c = out["checkpoint"]
        print(f"retrieved (top {len(c.get('retrieved_doc_ids', []))}): {c.get('retrieved_doc_ids', [])[:10]} ...")
        print(f"context docs: {c.get('context_docs')}  chars={c.get('context_chars')} sha={str(c.get('context_sha256'))[:12]}\n")
    if out.get("answer"):
        print(f"answer ({len(out['answer']['answer'])} chars):\n{out['answer']['answer']}\n")
    for name in ("official_results_strict.json", "official_results_protocol.json"):
        r = out.get(name)
        if r:
            print(f"{name}: correct={r['answer_correct']} completeness={r['completeness_pct']} recall={r.get('document_recall_pct')} "
                  f"extra={r.get('invalid_extra_docs')} corrected={r.get('corrected')}\n  reasoning: {r.get('correctness_reasoning')}\n")
    if out.get("correction"):
        c = out["correction"]
        print(f"correction: doc set changed={c.get('doc_set_changed')} gold answer changed={c.get('gold_answer_changed')}")
        print(f"  before docs: {c['before']['expected_doc_ids']}\n  after docs:  {c['after']['expected_doc_ids']}")
    if out.get("context"):
        print("\n--- context ---\n" + out["context"]["context"])
    return 0


# ------------------------------------------------------------------- main --

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="erb-hydradb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor"); s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("setup"); s.add_argument("--erb-dir"); s.set_defaults(fn=cmd_setup)

    s = sub.add_parser("download"); s.add_argument("--from-checkout", action="store_true")
    s.add_argument("--erb"); s.add_argument("--limit", type=int); s.set_defaults(fn=cmd_download)

    s = sub.add_parser("ingest"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--no-infer", action="store_true"); s.add_argument("--resume", action="store_true")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--limit", type=int, help="documents to ingest; part of the resume identity (repeat it with --resume)")
    s.add_argument("--source-types", nargs="+"); s.add_argument("--batch-size", type=int, default=80)
    s.add_argument("--batch-sleep", type=float, default=1.0); s.add_argument("--no-provision", action="store_true")
    s.add_argument("--no-wait", action="store_true",
                   help="send without waiting for indexing status; the run ends INCOMPLETE (exit 2) until --resume reconciles it")
    s.add_argument("--retry-failed", action="store_true", help="re-send the items HydraDB reported as failed (use with --resume)")
    s.add_argument("--accept-failures", action="store_true",
                   help="end with exit 0 even if some items failed; they are recorded in the manifest and the corpus is NOT ready")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("generate"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--questions"); s.add_argument("--targets"); s.add_argument("--n", type=int)
    s.add_argument("--retrieval-only", action="store_true"); s.add_argument("--from-contexts")
    s.add_argument("--force-resume", action="store_true"); s.add_argument("--allow-unpinned", action="store_true")
    s.add_argument("--allow-partial", action="store_true", help="exit 0 even if some questions failed")
    s.set_defaults(fn=cmd_generate)

    s = sub.add_parser("judge"); s.add_argument("--config", required=True); s.add_argument("--run-dir", required=True)
    s.add_argument("--protocol", choices=["strict", "official"], default="strict")
    s.add_argument("--answers"); s.add_argument("--results"); s.add_argument("--erb"); s.add_argument("--questions")
    s.add_argument("--question-id"); s.add_argument("--expect-n", type=int, default=None)
    s.add_argument("--allow-unpinned", action="store_true")
    s.add_argument("--fresh", action="store_true", help="official: archive previous shards and judge everything again")
    s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("report"); s.add_argument("--run-dir", required=True); s.add_argument("--title")
    s.add_argument("--questions"); s.add_argument("--leaderboard"); s.add_argument("--out"); s.set_defaults(fn=cmd_report)

    s = sub.add_parser("compare"); s.add_argument("--a", required=True); s.add_argument("--b", required=True)
    s.add_argument("--questions"); s.add_argument("--ids"); s.add_argument("--expect-n", type=int)
    s.add_argument("--checkpoint-a"); s.add_argument("--answers-a"); s.add_argument("--checkpoint-b"); s.add_argument("--answers-b")
    s.add_argument("--json"); s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("recall"); s.add_argument("--a", required=True, help="gen_checkpoint.json of run A")
    s.add_argument("--b", required=True); s.add_argument("--questions"); s.add_argument("--ctx", type=int, default=12)
    s.add_argument("--ids"); s.set_defaults(fn=cmd_recall)

    s = sub.add_parser("verify"); s.add_argument("--run-dir", required=True); s.add_argument("--questions")
    s.add_argument("--contexts", action="store_true", help="also rebuild every context from the corpus and compare hashes")
    s.add_argument("--minimal", action="store_true",
                   help="partial run dir (pilot): do not require the published artifact set or full-set coverage")
    s.add_argument("--log", help="write the full validation report to this JSON file"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("stats"); s.add_argument("--results", required=True); s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("audit-corrections"); s.add_argument("--run-dir", required=True); s.add_argument("--questions")
    s.add_argument("--out"); s.set_defaults(fn=cmd_audit_corrections)

    s = sub.add_parser("inspect"); s.add_argument("--run-dir", required=True); s.add_argument("--question-id", required=True)
    s.add_argument("--questions"); s.add_argument("--context", action="store_true", help="print the full context text")
    s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_inspect)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
