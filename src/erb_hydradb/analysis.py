"""Scoring arithmetic and paired comparisons, all offline.

* ``recompute_stats`` re-derives the evaluator's aggregate and per-category
  statistics from per-question rows with the benchmark's own formula, so a
  published results file can be checked without any API key.
* ``compare`` is a paired, per-question comparison of two results files:
  bootstrap CI on the combined-score delta, flip table, per-category and
  Slack-only slices, and recall at several depths from the saved retrieval
  order.
* ``recall_compare`` compares two retrieval runs without any LLM involvement.
* ``report`` renders RESULTS.md from a run directory.
"""

from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

DEPTHS = (5, 10, 20, 50)


# ----------------------------------------------------------------- loading --

def load_results(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def results_by_id(path: str | Path) -> dict[str, dict]:
    return {q["question_id"]: q for q in load_results(path)["questions"]}


def load_jsonl(path: str | Path) -> dict[str, dict]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["question_id"]] = r
    return out


def load_retrieval(checkpoint: str | Path | None, answers: str | Path | None) -> tuple[dict[str, list[str]], str]:
    """Doc ids in rank order per question: the checkpoint's saved top-50 if
    available, else the submitted document_ids (and say which)."""
    from . import validate
    if checkpoint and Path(checkpoint).exists():
        with open(checkpoint, "r", encoding="utf-8") as f:
            rows = [r for r in json.load(f) if "question_id" in r]
        dup = validate.check_ids(rows, None, "checkpoint")
        if dup:
            raise SystemExit("retrieval file rejected: " + "; ".join(dup))
        if rows and all("retrieved_doc_ids" in r for r in rows):
            return {r["question_id"]: r["retrieved_doc_ids"] for r in rows}, "checkpoint retrieved_doc_ids"
        if rows and all("document_ids" in r for r in rows):
            # older checkpoints only kept the submitted ids
            return {r["question_id"]: r["document_ids"] for r in rows}, "checkpoint document_ids (submitted)"
    if answers and Path(answers).exists():
        return {q: r.get("document_ids") or [] for q, r in load_jsonl(answers).items()}, "submitted document_ids"
    return {}, "none"


# ------------------------------------------------------------- the formula --

def per_question_score(r: dict) -> float:
    """The benchmark's combined score for one question."""
    return float(r["completeness_pct"]) if r["answer_correct"] else 0.0


def stats_for_group(rows: list[dict]) -> dict:
    """Mirror of the evaluator's ``compute_stats_for_group``."""
    n = len(rows)
    if n == 0:
        return {"count": 0, "average_correctness_pct": 0.0, "average_completeness_pct": 0.0,
                "combined_correctness_completeness_score": 0.0, "average_recall_pct": 0.0,
                "average_invalid_extra_docs": 0.0}
    recall = [r["document_recall_pct"] for r in rows if r.get("document_recall_pct") is not None]
    extra = [r["invalid_extra_docs"] for r in rows if r.get("invalid_extra_docs") is not None]
    return {
        "count": n,
        "average_correctness_pct": round(sum(1 for r in rows if r["answer_correct"]) / n * 100, 2),
        "average_completeness_pct": round(sum(r["completeness_pct"] for r in rows) / n, 2),
        "combined_correctness_completeness_score": round(sum(per_question_score(r) for r in rows) / n, 2),
        "average_recall_pct": round(sum(recall) / len(recall), 2) if recall else 0.0,
        "average_invalid_extra_docs": round(sum(extra) / len(extra), 2) if extra else 0.0,
    }


def recompute_stats(path: str | Path) -> dict:
    """Recompute aggregates from the per-question rows and diff against the file."""
    data = load_results(path)
    rows = data["questions"]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r)
    recomputed = {"aggregate_stats": stats_for_group(rows),
                  "question_type_stats": {t: stats_for_group(v) for t, v in by_type.items()}}
    from . import validate
    diffs, _ = validate.check_results_file(Path(path), None)
    return {"n": len(rows), "recomputed": recomputed, "diffs": diffs}


# ------------------------------------------------------------ comparisons --

def summary(res: dict[str, dict], ids: list[str]) -> dict:
    return {
        "n": len(ids),
        "combined": round(st.mean(per_question_score(res[i]) for i in ids), 2),
        "correctness_pct": round(100 * st.mean(1 if res[i]["answer_correct"] else 0 for i in ids), 2),
        "completeness_pct": round(st.mean(res[i]["completeness_pct"] for i in ids), 2),
    }


def paired_bootstrap(deltas: list[float], n_boot: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"delta_mean": round(sum(deltas) / n, 2),
            "ci95": [round(means[int(0.025 * n_boot)], 2), round(means[int(0.975 * n_boot)], 2)],
            "n_boot": n_boot, "seed": seed}


def flips(a: dict, b: dict, ids: list[str]) -> dict:
    t = defaultdict(list)
    for i in ids:
        t[f"{'T' if a[i]['answer_correct'] else 'F'}->{'T' if b[i]['answer_correct'] else 'F'}"].append(i)
    return {k: {"n": len(v), "ids": v} for k, v in sorted(t.items())}


def recall_at(ranked: dict[str, list[str]], questions: dict[str, dict], ids: list[str], k: int) -> dict:
    """Set-based, exactly as the evaluator: |set(submitted[:k]) & set(gold)| / |set(gold)|.
    A gold list may repeat an id (qst_0413 does); the set is what counts."""
    rec, extra = [], []
    for i in ids:
        exp = set(questions[i].get("expected_doc_ids") or [])
        sub = set((ranked.get(i) or [])[:k])
        if exp:
            rec.append(100 * len(exp & sub) / len(exp))
            extra.append(len(sub - exp))
    return {"k": k, "recall_pct": round(st.mean(rec), 2) if rec else None,
            "invalid_extra": round(st.mean(extra), 2) if extra else None, "n": len(rec)}


def compare(a_path: str | Path, b_path: str | Path, questions_path: str | Path, *,
            ids_file: str | Path | None = None, expect_n: int | None = None,
            a_checkpoint=None, a_answers=None, b_checkpoint=None, b_answers=None) -> dict:
    from . import validate
    # validate RAW rows (schema, duplicates, every statistic) before collapsing by id
    for tag, path in (("A", a_path), ("B", b_path)):
        problems, _ = validate.check_results_file(Path(path), None)
        if problems:
            raise SystemExit(f"results file {tag} rejected: " + "; ".join(problems[:5]))
    a, b = results_by_id(a_path), results_by_id(b_path)
    qs = load_jsonl(questions_path)
    if expect_n is not None:
        for tag, res in (("A", a), ("B", b)):
            if len(res) != expect_n:
                raise SystemExit(f"COVERAGE FAILURE {tag}: {len(res)} judged rows (expected {expect_n})")
        if set(a) != set(b):
            raise SystemExit(f"COVERAGE FAILURE: question id sets differ ({len(set(a) ^ set(b))} ids)")
    ids = sorted(set(a) & set(b))
    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            keep = {line.strip() for line in f if line.strip()}
        ids = [i for i in ids if i in keep]
    if not ids:
        raise SystemExit("no overlapping question ids")

    deltas = [per_question_score(b[i]) - per_question_score(a[i]) for i in ids]
    report = {"n": len(ids), "a": summary(a, ids), "b": summary(b, ids),
              "paired_bootstrap": paired_bootstrap(deltas), "flips": flips(a, b, ids),
              "by_type": {}, "slack_only": None, "recall": {}}
    by_type = defaultdict(list)
    for i in ids:
        by_type[qs[i]["question_type"]].append(i)
    for t, tids in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        report["by_type"][t] = {"a": summary(a, tids), "b": summary(b, tids)}
    slack = [i for i in ids if qs[i].get("source_types") == ["slack"]]
    if slack:
        report["slack_only"] = {"a": summary(a, slack), "b": summary(b, slack),
                                "flips": {k: v["n"] for k, v in flips(a, b, slack).items()}}
    for tag, ck, ans in (("a", a_checkpoint, a_answers), ("b", b_checkpoint, b_answers)):
        ranked, source = load_retrieval(ck, ans)
        if ranked:
            report["recall"][tag] = {"source": source, "rows": [recall_at(ranked, qs, ids, k) for k in DEPTHS]}
    return report


def print_compare(report: dict) -> None:
    print(f"n={report['n']}")
    print(f"  A: {report['a']}")
    print(f"  B: {report['b']}")
    pb = report["paired_bootstrap"]
    print(f"  paired delta(combined) = {pb['delta_mean']}  95% CI {pb['ci95']}")
    print("  flips: " + ", ".join(f"{k}={v['n']}" for k, v in report["flips"].items()))
    print(f"  {'type':26s} {'n':>4}  {'A comb':>7} {'B comb':>7}  {'A corr':>6} {'B corr':>6}")
    for t, v in report["by_type"].items():
        print(f"  {t:26s} {v['a']['n']:>4}  {v['a']['combined']:>7} {v['b']['combined']:>7}  "
              f"{v['a']['correctness_pct']:>6} {v['b']['correctness_pct']:>6}")
    if report["slack_only"]:
        s = report["slack_only"]
        print(f"  slack-only n={s['a']['n']}: {s['a']['combined']} -> {s['b']['combined']} flips {s['flips']}")
    for tag, rr in report["recall"].items():
        print(f"  recall {tag} [{rr['source']}]: " + "; ".join(
            f"@{r['k']} {r['recall_pct']}% extra {r['invalid_extra']}" for r in rr["rows"]))


def recall_compare(a_ranked: dict[str, list[str]], b_ranked: dict[str, list[str]],
                   questions_path: str | Path, ctx: int = 12, ids_file: str | Path | None = None) -> dict:
    qs = load_jsonl(questions_path)
    ids = sorted(i for i in qs if qs[i].get("expected_doc_ids") and i in a_ranked and i in b_ranked)
    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            keep = {line.strip() for line in f if line.strip()}
        ids = [i for i in ids if i in keep]
    if not ids:
        raise SystemExit("no questions with gold documents are present in both retrieval runs")

    def rec(ranked, k):
        return [100 * len(set(qs[i]["expected_doc_ids"]) & set(ranked.get(i, [])[:k]))
                / len(set(qs[i]["expected_doc_ids"])) for i in ids]

    rows = []
    for k in sorted({5, 10, ctx, 20, 50}):
        ra, rb = rec(a_ranked, k), rec(b_ranked, k)
        d = [y - x for x, y in zip(ra, rb, strict=True)]
        rows.append({"k": k, "a": round(st.mean(ra), 2), "b": round(st.mean(rb), 2), **paired_bootstrap(d)})
    ra, rb = rec(a_ranked, ctx), rec(b_ranked, ctx)
    gained = [i for i, x, y in zip(ids, ra, rb, strict=True) if x < 100 <= y]
    lost = [i for i, x, y in zip(ids, ra, rb, strict=True) if y < 100 <= x]
    return {"n": len(ids), "ctx": ctx, "rows": rows, "full_coverage_gained": gained, "full_coverage_lost": lost}


# ---------------------------------------------------------------- report --

def render_results_md(run_dir: str | Path, questions_path: str | Path, title: str,
                      leaderboard: dict | None = None) -> str:
    """RESULTS.md for a run directory, generated from its files, never by hand."""
    run_dir = Path(run_dir)
    qs = load_jsonl(questions_path)
    out = [f"# {title}", ""]
    out.append("Generated by `erb-hydradb report` from the files in this directory. "
               "Every number below is recomputed from the per-question rows with the benchmark's formula.")
    out.append("")
    for name, label in (("official_results_protocol.json", "Official protocol (citation stripping + document correction)"),
                        ("official_results_strict.json", "Strict (no correction, no citation stripping)")):
        p = run_dir / name
        if not p.exists():
            continue
        rc = recompute_stats(p)
        a = rc["recomputed"]["aggregate_stats"]
        out += [f"## {label}", "",
                f"n = {rc['n']} questions" + (f"; **recomputation mismatches: {rc['diffs']}**" if rc["diffs"] else "; recomputation matches the file"),
                "",
                "| Combined score | Correctness | Completeness | Document recall | Invalid extra docs |",
                "|---|---|---|---|---|",
                f"| **{a['combined_correctness_completeness_score']}** | {a['average_correctness_pct']} % | "
                f"{a['average_completeness_pct']} % | {a['average_recall_pct']} % | {a['average_invalid_extra_docs']} |",
                "", "| Category | n | Combined | Correctness | Completeness | Recall |", "|---|---|---|---|---|---|"]
        by_t = defaultdict(list)
        for r in load_results(p)["questions"]:
            by_t[r["question_type"]].append(r)
        for t, s in sorted(rc["recomputed"]["question_type_stats"].items(), key=lambda kv: -kv[1]["count"]):
            has_gold = any(r.get("document_recall_pct") is not None for r in by_t[t])
            recall = f"{s['average_recall_pct']} %" if has_gold else "n/a (no gold documents)"
            out.append(f"| {t} | {s['count']} | {s['combined_correctness_completeness_score']} | "
                       f"{s['average_correctness_pct']} % | {s['average_completeness_pct']} % | {recall} |")
        out.append("")
    ranked, source = load_retrieval(run_dir / "gen_checkpoint.json", run_dir / "answers.jsonl")
    if ranked:
        ids = [i for i in qs if qs[i].get("expected_doc_ids") and i in ranked]
        out += [f"## Document recall at depth ({source})", "", "| k | Recall | Invalid extra docs |", "|---|---|---|"]
        for k in (1, 3, 5, 10, 20, 50):
            r = recall_at(ranked, qs, ids, k)
            out.append(f"| {k} | {r['recall_pct']} % | {r['invalid_extra']} |")
        full = sum(1 for i in ids if set(qs[i]["expected_doc_ids"]) <= set(ranked[i][:10]))  # set-based
        out += ["", f"Questions with every gold document inside the top 10: {full} of {len(ids)}.", ""]
    corr = run_dir / "corrections.jsonl"
    if corr.exists():
        recs = [json.loads(line) for line in open(corr, "r", encoding="utf-8") if line.strip()]
        out += ["## Gold corrections applied by the official protocol", "",
                f"{len(recs)} questions had their gold set changed by the evaluator's three-judge correction step "
                f"before scoring; each record in `corrections.jsonl` holds the pinned original and the corrected "
                f"gold documents, gold answer and answer facts, with the judges' reasons. "
                f"Document set changed: {sum(1 for r in recs if r.get('doc_set_changed'))}; gold answer changed: "
                f"{sum(1 for r in recs if r.get('gold_answer_changed'))}. Check with `erb-hydradb audit-corrections`.", "",
                "| Question | Type | Gold docs before | Gold docs after |", "|---|---|---|---|"]
        for r in recs:
            out.append(f"| {r['question_id']} | {r.get('question_type')} | {len(r['before']['expected_doc_ids'])} | "
                       f"{len(r['after']['expected_doc_ids'])} |")
        out.append("")
    if leaderboard:
        out += ["## Public leaderboard at the time of the run", "",
                f"Source: {leaderboard.get('source', '')} ({leaderboard.get('date', '')}).", "",
                "| Rank | System | Overall | Correctness | Completeness | Recall | Invalid extra |", "|---|---|---|---|---|---|---|"]
        for r in leaderboard.get("rows", []):
            out.append(f"| {r['rank']} | {r['system']} | {r['overall']} | {r['correctness']} | {r['completeness']} | {r['recall']} | {r['invalid_extra']} |")
        out.append("")
    return "\n".join(out)
