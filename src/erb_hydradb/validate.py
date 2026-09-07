"""One validator for run directories, results files and context files.

Every command that reads artifacts goes through these checks so that a
malformed, truncated, duplicated or tampered input fails loudly instead of
producing a plausible number. Each check returns a list of problem strings;
an empty list means the check passed. ``validate_run`` aggregates them into a
report that names every check performed and its status.

Design rules:
* validate RAW rows before any dict-by-id collapse (duplicates must be seen);
* compare against an EXPECTED question-id set, never just a count;
* hash the ACTUAL text, never trust a stored label;
* a requested check that cannot run is INCOMPLETE, never a silent pass.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from . import hydrate, manifest

RESULT_ROW_KEYS = {"question_id", "question_type", "answer_correct", "completeness_pct",
                   "document_recall_pct", "invalid_extra_docs"}
CONTEXT_ROW_KEYS = {"question_id", "context", "context_docs", "dropped_docs", "chunk_fallback_docs",
                    "context_chars", "context_sha256"}
STAT_KEYS = ("count", "average_correctness_pct", "average_completeness_pct",
             "combined_correctness_completeness_score", "average_recall_pct", "average_invalid_extra_docs")


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, "r", encoding="utf-8")


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with _open_text(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc.msg})") from exc
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- ids ----

def check_ids(rows: list[dict], expected: set[str] | None, label: str) -> list[str]:
    """Raw rows: every row has a question_id, no duplicates, exact coverage."""
    problems = []
    ids = [r.get("question_id") for r in rows]
    if any(not isinstance(i, str) or not i for i in ids):
        problems.append(f"{label}: rows without a question_id")
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(ids) != len(set(ids)) else []
    if dupes:
        problems.append(f"{label}: duplicate question ids {dupes[:5]}{' …' if len(dupes) > 5 else ''}")
    if expected is not None:
        got = set(i for i in ids if isinstance(i, str))
        missing, extra = sorted(expected - got), sorted(got - expected)
        if missing:
            problems.append(f"{label}: missing {len(missing)} expected ids {missing[:5]}{' …' if len(missing) > 5 else ''}")
        if extra:
            problems.append(f"{label}: {len(extra)} unexpected ids {extra[:5]}{' …' if len(extra) > 5 else ''}")
    return problems


def expected_ids_from_questions(questions_path: Path) -> set[str]:
    return {r["question_id"] for r in read_jsonl_rows(questions_path)}


# ------------------------------------------------------------ results ----

def check_results_file(path: Path, expected: set[str] | None) -> tuple[list[str], dict | None]:
    """Schema, ids, and every aggregate / per-category statistic recomputed."""
    from .analysis import stats_for_group
    problems = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path.name}: unreadable ({exc})"], None
    rows = data.get("questions")
    if not isinstance(rows, list) or not rows:
        return [f"{path.name}: no question rows"], None
    for r in rows:
        missing = RESULT_ROW_KEYS - set(r)
        if missing:
            problems.append(f"{path.name}: row {r.get('question_id')} missing {sorted(missing)}")
            break
        if not isinstance(r["answer_correct"], bool) or not isinstance(r["completeness_pct"], (int, float)):
            problems.append(f"{path.name}: row {r.get('question_id')} has invalid answer_correct/completeness_pct")
            break
    problems += check_ids(rows, expected, path.name)
    if problems:
        return problems, data
    agg = data.get("aggregate_stats")
    if not isinstance(agg, dict):
        problems.append(f"{path.name}: aggregate_stats missing")
    else:
        rec = stats_for_group(rows)
        # the evaluator's aggregate block reports total_questions/completed_questions instead of count
        for k in ("total_questions", "completed_questions"):
            if k in agg and int(agg[k]) != len(rows):
                problems.append(f"{path.name}: aggregate_stats.{k} = {agg[k]} but there are {len(rows)} rows")
        for k in STAT_KEYS:
            if k == "count":
                continue
            if k not in agg:
                problems.append(f"{path.name}: aggregate_stats.{k} missing")
            elif abs(float(agg[k]) - float(rec[k])) > 0.011:
                problems.append(f"{path.name}: aggregate_stats.{k} = {agg[k]} but recomputed {rec[k]}")
    qts = data.get("question_type_stats")
    if not isinstance(qts, dict):
        problems.append(f"{path.name}: question_type_stats missing")
    else:
        by_t: dict[str, list[dict]] = {}
        for r in rows:
            by_t.setdefault(r["question_type"], []).append(r)
        if set(qts) != set(by_t):
            problems.append(f"{path.name}: question_type_stats categories {sorted(set(qts) ^ set(by_t))} do not match rows")
        for t, group in by_t.items():
            if t not in qts:
                continue
            rec = stats_for_group(group)
            for k in STAT_KEYS:
                if k not in qts[t]:
                    problems.append(f"{path.name}: question_type_stats.{t}.{k} missing")
                elif abs(float(qts[t][k]) - float(rec[k])) > 0.011:
                    problems.append(f"{path.name}: question_type_stats.{t}.{k} = {qts[t][k]} but recomputed {rec[k]}")
    return problems, data


# ------------------------------------------------------------ contexts ----

def check_context_rows(rows: list[dict], expected: set[str] | None, label: str = "contexts") -> list[str]:
    """Schema, ids, actual sha256 of the text, declared length, forbidden markers."""
    problems = []
    for r in rows:
        missing = CONTEXT_ROW_KEYS - set(r)
        if missing:
            problems.append(f"{label}: row {r.get('question_id')} missing {sorted(missing)}")
            continue
        text = r["context"]
        if not isinstance(text, str):
            problems.append(f"{label}: row {r['question_id']} context is not a string")
            continue
        actual = sha256_text(text)
        if actual != r["context_sha256"]:
            problems.append(f"{label}: row {r['question_id']} context_sha256 label {str(r['context_sha256'])[:12]} "
                            f"does not match the actual text {actual[:12]}")
        if r["context_chars"] != len(text):
            problems.append(f"{label}: row {r['question_id']} context_chars {r['context_chars']} != len {len(text)}")
        for m in hydrate.FORBIDDEN_MARKERS:
            if m in text:
                problems.append(f"{label}: row {r['question_id']} contains forbidden marker {m!r}")
    problems += check_ids(rows, expected, label)
    if not rows:
        problems.append(f"{label}: no rows")
    return problems


def check_contexts_against_checkpoint(ctx_rows: list[dict], ck_rows: list[dict]) -> list[str]:
    problems = []
    ck = {r["question_id"]: r for r in ck_rows}
    for r in ctx_rows:
        c = ck.get(r["question_id"])
        if c is None:
            problems.append(f"contexts: {r['question_id']} has no checkpoint row")
        elif c.get("context_sha256") != r["context_sha256"]:
            problems.append(f"contexts: {r['question_id']} sha256 differs from the checkpoint")
        elif c.get("context_docs") != r["context_docs"]:
            problems.append(f"contexts: {r['question_id']} context_docs differ from the checkpoint")
    return problems


# ---------------------------------------------------------- checkpoint ----

def read_checkpoint(path: Path) -> tuple[dict, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    header = rows[0].get("_config", {}) if rows and "_config" in rows[0] else {}
    return header, [r for r in rows if "question_id" in r]


def check_checkpoint(rows: list[dict], expected: set[str] | None) -> list[str]:
    problems = check_ids(rows, expected, "checkpoint")
    for r in rows:
        if "retrieved_doc_ids" not in r:
            problems.append(f"checkpoint: {r.get('question_id')} has no retrieved_doc_ids")
            break
        if r.get("document_ids") != r["retrieved_doc_ids"][:len(r.get("document_ids", []))]:
            problems.append(f"checkpoint: {r['question_id']} document_ids are not a prefix of retrieved_doc_ids")
            break
    return problems


# ------------------------------------------------------------ run dir ----

REQUIRED_PUBLISHED = ("answers.jsonl", "gen_checkpoint.json", "contexts.jsonl.gz", "official_results_strict.json",
                      "config.yaml", "manifest.json", "SHA256SUMS")


def check_sums_and_manifest(run_dir: Path, required: tuple[str, ...]) -> list[str]:
    problems = []
    sums = run_dir / "SHA256SUMS"
    if not sums.exists() or not sums.read_text(encoding="utf-8").strip():
        problems.append("SHA256SUMS missing or empty")
    else:
        listed = {line.split("  ", 1)[1] for line in sums.read_text(encoding="utf-8").splitlines() if "  " in line}
        for name in required:
            if name != "SHA256SUMS" and name not in listed:
                problems.append(f"SHA256SUMS does not cover required file {name}")
        problems += manifest.verify_sums(run_dir)
    mpath = run_dir / "manifest.json"
    if mpath.exists():
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                m = json.load(f)
            for st in m.get("stages", []):
                for rel, digest in (st.get("files") or {}).items():
                    p = run_dir / rel
                    if not p.exists():
                        problems.append(f"manifest stage {st.get('stage')}: {rel} missing")
                    elif manifest.sha256_file(p) != digest:
                        problems.append(f"manifest stage {st.get('stage')}: {rel} hash differs from the file")
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"manifest.json unreadable ({exc})")
    for name in required:
        if not (run_dir / name).exists():
            problems.append(f"required file missing: {name}")
    return problems


def validate_run(run_dir: Path, questions_path: Path | None, *, required: tuple[str, ...] = REQUIRED_PUBLISHED,
                 contexts: bool = False, store=None, docs_in_context: int = 12,
                 max_context_chars: int = 240_000) -> dict:
    """Full validation of a run directory. Returns
    {"checks": {name: {"status": ok|fail|incomplete, "problems": [...], ...}}, "ok": bool}."""
    checks: dict[str, dict] = {}

    def record(name, problems, **extra):
        checks[name] = {"status": "ok" if not problems else "fail", "problems": problems, **extra}

    record("files_and_checksums", check_sums_and_manifest(run_dir, required))

    expected: set[str] | None = None
    if questions_path and questions_path.exists():
        expected = expected_ids_from_questions(questions_path)
        record("expected_ids", [], n=len(expected), source=str(questions_path))
    else:
        checks["expected_ids"] = {"status": "incomplete", "problems": ["no questions file; coverage checked by cross-file equality only"]}

    id_sets: dict[str, set[str]] = {}
    ans = run_dir / "answers.jsonl"
    if ans.exists():
        rows = read_jsonl_rows(ans)
        p = check_ids(rows, expected, "answers")
        for r in rows:
            if set(r) != {"question_id", "answer", "document_ids"}:
                p.append(f"answers: row {r.get('question_id')} has keys {sorted(r)} (expected question_id, answer, document_ids)")
                break
        record("answers", p, n=len(rows))
        id_sets["answers"] = {r.get("question_id") for r in rows}

    ck = run_dir / "gen_checkpoint.json"
    ck_rows: list[dict] = []
    if ck.exists():
        try:
            header, ck_rows = read_checkpoint(ck)
            record("checkpoint", check_checkpoint(ck_rows, expected), n=len(ck_rows), fingerprint=header.get("fingerprint"))
            id_sets["checkpoint"] = {r["question_id"] for r in ck_rows}
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            record("checkpoint", [f"unreadable ({exc})"])

    for name in ("official_results_strict.json", "official_results_protocol.json"):
        p = run_dir / name
        if p.exists():
            problems, data = check_results_file(p, expected)
            agg = (data or {}).get("aggregate_stats") or {}
            record(name, problems, n=len((data or {}).get("questions") or []),
                   combined=agg.get("combined_correctness_completeness_score"))
            if data:
                id_sets[name] = {r.get("question_id") for r in data["questions"]}
        elif name in required:
            record(name, ["missing"])

    ctx_path = run_dir / "contexts.jsonl.gz"
    ctx_rows: list[dict] = []
    if ctx_path.exists():
        try:
            ctx_rows = read_jsonl_rows(ctx_path)
            p = check_context_rows(ctx_rows, expected)
            if ck_rows:
                p += check_contexts_against_checkpoint(ctx_rows, ck_rows)
            record("contexts", p, n=len(ctx_rows))
            id_sets["contexts"] = {r.get("question_id") for r in ctx_rows}
        except ValueError as exc:
            record("contexts", [str(exc)])
    elif "contexts.jsonl.gz" in required:
        record("contexts", ["missing"])

    if len(id_sets) >= 2:
        ref_name, ref = next(iter(id_sets.items()))
        p = [f"{name} ids differ from {ref_name} ({len(s ^ ref)} ids)" for name, s in id_sets.items() if s != ref]
        record("cross_file_ids", p, files=sorted(id_sets))

    if contexts:
        if not ctx_rows or not ck_rows:
            checks["context_reconstruction"] = {"status": "incomplete", "problems": ["needs contexts.jsonl.gz and gen_checkpoint.json"]}
        elif store is None:
            checks["context_reconstruction"] = {"status": "incomplete",
                                                "problems": ["needs data/documents.sqlite or an ERB checkout (ERB_REPO)"]}
        else:
            published = {r["question_id"]: r["context_sha256"] for r in ctx_rows}
            mism, checked = [], 0
            for r in ck_rows:
                if r["question_id"] not in published:
                    continue
                rebuilt = hydrate.build_context(r["retrieved_doc_ids"], [], store, docs_in_context, max_context_chars)
                checked += 1
                if rebuilt["context_sha256"] != published[r["question_id"]]:
                    mism.append(r["question_id"])
            p = [f"{len(mism)} contexts do not rebuild from the corpus: {mism[:5]}"] if mism else []
            if checked == 0:
                p.append("no contexts were checked")
            record("context_reconstruction", p, checked=checked, reproduced=checked - len(mism), backend=store.backend)

    # A failed check fails the run. An INCOMPLETE check fails it only when that
    # check was explicitly requested (context reconstruction); otherwise it is
    # reported as such and the summary still says which checks did not run.
    ok = all(c["status"] != "fail" for c in checks.values())
    if contexts and checks.get("context_reconstruction", {}).get("status") == "incomplete":
        ok = False
    return {"ok": ok, "checks": checks}


def format_report(report: dict) -> str:
    lines = []
    for name, c in report["checks"].items():
        extra = {k: v for k, v in c.items() if k not in ("status", "problems")}
        tail = f"  {extra}" if extra else ""
        lines.append(f"{c['status'].upper():10s} {name}{tail}")
        for p in c["problems"]:
            lines.append(f"           - {p}")
    lines.append("VERIFIED" if report["ok"] else "PROBLEMS FOUND")
    return "\n".join(lines)
