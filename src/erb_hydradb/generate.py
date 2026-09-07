"""Answer generation: HydraDB retrieval -> canonical full-document context -> two-pass answer.

For each question:
  1. ``POST /query`` (hybrid, configured mode/alpha/max_results); chunks are
     collapsed to distinct ERB document ids in rank order;
  2. the top ``docs_in_context`` documents are hydrated with their canonical
     text (see ``hydrate.py``) under a total character budget;
  3. the answer model writes a draft, critiques it against the same documents,
     and rewrites (``prompts.py``, version two-pass-v1);
  4. the top ``submit_docs`` document ids are written to ``answers.jsonl`` in the
     benchmark's submission format.

Every checkpoint row carries an explicit ``stage``:

  retrieved   retrieval done, no answer requested (``--retrieval-only``)
  answered    retrieval done and an answer produced
  error       the row failed (``error`` says where); retried on resume

A run is COMPLETE only when every requested question has an ``answered`` row
(or ``retrieved`` in retrieval-only mode). ``run_generate`` returns the rows and
a completeness verdict; the CLI exits nonzero on an incomplete run unless
``--allow-partial`` is given. Resuming a retrieval-only checkpoint in generation
mode reuses the saved retrieval order and generates the missing answers; it
never counts a ``retrieved`` row as an answer.

Replaying published contexts (``--from-contexts``) validates the file before any
model call: schema, unique ids, coverage of the requested questions, the actual
sha256 of every context, its declared length, and the absence of benchmark
markers. The run identity (checkpoint fingerprint) binds the configuration,
the operation mode, the questions file and, for replay, the contexts file.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

from . import hydrate, manifest, validate
from .config import RunConfig
from .llm import complete
from .prompts import CRITIQUE_PROMPT, PASS1_PROMPT, PASS2_PROMPT, PROMPTS_VERSION


class IncompleteRun(RuntimeError):
    def __init__(self, missing: list[str], errors: list[str]):
        self.missing, self.errors = missing, errors
        super().__init__(f"{len(missing)} requested questions without a completed row; {len(errors)} error rows")


def load_questions(path: str | Path) -> list[dict]:
    rows = validate.read_jsonl_rows(Path(path))
    problems = validate.check_ids(rows, None, "questions")
    if problems:
        raise SystemExit("questions file invalid: " + "; ".join(problems))
    return rows


def select_questions(questions: list[dict], targets: str | Path | None, n: int | None) -> list[dict]:
    if targets:
        with open(targets, "r", encoding="utf-8") as f:
            keep = [line.strip() for line in f if line.strip()]
        by_id = {q["question_id"]: q for q in questions}
        unknown = [k for k in keep if k not in by_id]
        if unknown:
            raise SystemExit(f"--targets names {len(unknown)} ids not in the questions file: {unknown[:5]}")
        questions = [by_id[q] for q in keep]
    if n is not None:
        questions = questions[:n]
    return questions


def generate_answer(question: str, context: str, cfg: RunConfig) -> tuple[str, dict]:
    g = cfg.generation
    kw = dict(model=g.model, provider=g.provider, max_tokens=g.max_tokens, temperature=g.temperature)
    stages = {"pass1": False, "critique": False, "pass2": False}
    p1 = complete(PASS1_PROMPT.format(question=question, context=context), **kw)
    stages["pass1"] = bool(p1)
    if not g.correction_loop or not p1:
        return p1, stages
    try:
        critique = complete(CRITIQUE_PROMPT.format(question=question, context=context, answer=p1), **kw)
    except Exception:  # noqa: BLE001
        return p1, stages
    stages["critique"] = bool(critique)
    if not critique:
        return p1, stages
    try:
        p2 = complete(PASS2_PROMPT.format(question=question, context=context, answer=p1, critique=critique), **kw)
    except Exception:  # noqa: BLE001
        return p1, stages
    stages["pass2"] = bool(p2)
    return (p2 or p1), stages


# ---------------------------------------------------------- identity ----

def run_identity(cfg: RunConfig, *, questions_path: Path | None, from_contexts: Path | None) -> dict:
    """What a checkpoint is bound to. Operation mode is NOT part of it, so a
    retrieval-only checkpoint can be continued into generation; the stage
    field on each row records what was actually done."""
    ident = {
        "fingerprint": cfg.generation_fingerprint(),
        "prompts": PROMPTS_VERSION,
        "base_url": cfg.hydradb.base_url,
        "questions_sha256": manifest.sha256_file(questions_path) if questions_path else None,
        "contexts_sha256": manifest.sha256_file(from_contexts) if from_contexts else None,
    }
    ident["identity"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident


def load_checkpoint(path: Path, identity: dict, force: bool) -> tuple[list[dict], list[str]]:
    """Rows from a checkpoint bound to the same identity. Returns (rows, notes).
    Refuses a different identity unless ``force``. Error rows are dropped (they
    will be retried); ``retrieved`` rows are kept and finished later."""
    if not path.exists():
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    notes = []
    if rows and "_config" in rows[0]:
        stored = rows[0]["_config"]
        if stored.get("identity") != identity["identity"]:
            diff = {k: (stored.get(k), identity.get(k)) for k in identity if k != "identity" and stored.get(k) != identity.get(k)}
            if not force:
                raise SystemExit(f"Refusing to resume {path}: it belongs to a different run identity {diff}. "
                                 f"Use a new run directory, or --force-resume to override (recorded in the manifest).")
            notes.append(f"forced resume across identities: {diff}")
        rows = rows[1:]
    keep = [r for r in rows if r.get("stage") in ("retrieved", "answered")]
    if len(keep) != len(rows):
        notes.append(f"retrying {len(rows) - len(keep)} error rows")
    return keep, notes


def write_answers(rows: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "answer": r.get("answer", ""),
                                "document_ids": r.get("document_ids", [])}) + "\n")
    return len(rows)


def load_published_contexts(path: Path, requested: list[dict]) -> dict[str, dict]:
    """Validate a contexts file against the requested questions BEFORE any
    model call: schema, unique ids, coverage, actual hashes, lengths, markers."""
    rows = validate.read_jsonl_rows(path)
    expected = {q["question_id"] for q in requested}
    problems = validate.check_context_rows(rows, None)
    have = {r.get("question_id") for r in rows}
    missing = sorted(expected - have)
    if missing:
        problems.append(f"contexts: {len(missing)} requested questions have no context row {missing[:5]}")
    if problems:
        raise SystemExit("published contexts rejected (nothing was sent to the model):\n  " + "\n  ".join(problems))
    return {r["question_id"]: r for r in rows if r["question_id"] in expected}


async def run_generate(cfg: RunConfig, run_dir: Path, questions: list[dict], *, api_key: str | None,
                       store: hydrate.DocumentStore | None, questions_path: Path | None = None,
                       retrieval_only: bool = False, from_contexts: Path | None = None,
                       force_resume: bool = False) -> tuple[list[dict], IncompleteRun | None]:
    from .hydradb_client import HydraDBQueryClient

    run_dir.mkdir(parents=True, exist_ok=True)
    g, r = cfg.generation, cfg.retrieval
    identity = run_identity(cfg, questions_path=questions_path, from_contexts=from_contexts)
    ckpt = run_dir / "gen_checkpoint.json"
    ctx_path = run_dir / "contexts.jsonl"
    rows, notes = load_checkpoint(ckpt, identity, force_resume)
    for n in notes:
        print(f"  resume: {n}", flush=True)
    by_id = {x["question_id"]: x for x in rows}
    requested_ids = [q["question_id"] for q in questions]
    # rows that satisfy this run: answered always; retrieved only in retrieval-only mode
    done = {qid for qid, x in by_id.items() if x["stage"] == "answered" or (retrieval_only and x["stage"] == "retrieved")}
    to_finish = [q for q in questions if q["question_id"] in by_id and q["question_id"] not in done]   # retrieved -> answer
    to_query = [q for q in questions if q["question_id"] not in by_id]
    total = len(requested_ids)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(g.workers)
    counter = {"n": sum(1 for q in questions if q["question_id"] in done)}

    published: dict[str, dict] = {}
    if from_contexts:
        published = load_published_contexts(from_contexts, to_query + to_finish)
        if store is None:
            print(f"  replaying {len(published)} validated contexts from {from_contexts}", flush=True)

    def save():
        ordered = [by_id[q] for q in requested_ids if q in by_id] + [x for q, x in by_id.items() if q not in set(requested_ids)]
        tmp = ckpt.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([{"_config": {**identity, "config": cfg.to_dict()}}] + ordered, f, ensure_ascii=False)
        os.replace(tmp, ckpt)

    def dump_context(qid: str, ctx: dict):
        with open(ctx_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": qid, **ctx}, ensure_ascii=False) + "\n")

    async def query(client: HydraDBQueryClient, question: str) -> list[dict]:
        import httpx
        for attempt in range(5):
            try:
                res = await client.query(cfg.hydradb.collection, question, max_results=r.max_results,
                                         mode=r.mode, alpha=r.alpha, query_by=r.query_by, query_apps=r.query_apps)
                if not res and attempt < 4:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                return res
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt < 4:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise
        return []

    def context_for(qid: str, doc_ids: list[str], chunks: list[dict]) -> dict:
        if published:
            pc = published[qid]
            ctx = {k: pc[k] for k in validate.CONTEXT_ROW_KEYS if k != "question_id"}
            ctx["context_sha256"] = validate.sha256_text(ctx["context"])  # hash of what is actually sent
            return ctx
        return hydrate.build_context(doc_ids, chunks, store, g.docs_in_context, g.max_context_chars)

    async def answer(q: dict, ctx: dict) -> tuple[str, dict, str | None]:
        try:
            a, stages = await asyncio.to_thread(generate_answer, q["question"], ctx["context"], cfg)
            if not a:
                return "", stages, "gen: empty answer"
            return a, stages, None
        except Exception as exc:  # noqa: BLE001
            print(f"    GEN ERROR {q['question_id']}: {str(exc)[:120]}", flush=True)
            return "", {}, f"gen: {str(exc)[:150]}"

    async def commit(qid: str, row: dict, ctx: dict | None, q: dict):
        async with lock:
            by_id[qid] = row
            counter["n"] += 1
            save()
            if ctx is not None and not published:
                dump_context(qid, {**ctx, "retrieved_doc_ids": row["retrieved_doc_ids"]})
            print(f"[{counter['n']}/{total}] {qid} [{q.get('question_type', '?')}] stage={row['stage']} "
                  f"docs={len(row['context_docs'])} ctx={row['context_chars']} ans_len={len(row['answer'])} "
                  f"{row['secs']:.0f}s" + (f" ERROR {row['error']}" if row.get("error") else ""), flush=True)

    async def finish_retrieved(q: dict):
        """A retrieved row from an earlier retrieval-only run: answer it now."""
        qid = q["question_id"]
        async with sem:
            t0 = time.time()
            prev = by_id[qid]
            ctx = context_for(qid, prev["retrieved_doc_ids"], [])
            a, stages, err = await answer(q, ctx)
            row = {**prev, "answer": a, "stages": stages, "secs": round(time.time() - t0, 1),
                   "context_docs": ctx["context_docs"], "context_chars": ctx["context_chars"],
                   "context_sha256": ctx["context_sha256"], "stage": "error" if err else "answered"}
            row.pop("error", None)
            if err:
                row["error"] = err
            await commit(qid, row, None, q)

    async def one(client: HydraDBQueryClient | None, q: dict):
        qid = q["question_id"]
        async with sem:
            t0 = time.time()
            if published:
                doc_ids = list(published[qid].get("retrieved_doc_ids") or published[qid]["context_docs"])
                chunks: list[dict] = []
            else:
                try:
                    chunks = await query(client, q["question"])
                except Exception as exc:  # noqa: BLE001
                    row = {"question_id": qid, "stage": "error", "error": f"query: {str(exc)[:150]}", "answer": "",
                           "document_ids": [], "retrieved_doc_ids": [], "context_docs": [], "context_chars": 0,
                           "secs": round(time.time() - t0, 1)}
                    await commit(qid, row, None, q)
                    return
                doc_ids = hydrate.distinct_docs(chunks)
            ctx = context_for(qid, doc_ids, chunks)
            a, stages, err = "", {}, None
            if not retrieval_only:
                if not doc_ids:
                    err = "query: no documents retrieved"
                else:
                    a, stages, err = await answer(q, ctx)
            row = {
                "question_id": qid,
                "stage": "error" if err else ("retrieved" if retrieval_only else "answered"),
                "answer": a, "document_ids": doc_ids[:g.submit_docs], "retrieved_doc_ids": doc_ids[:50],
                "context_docs": ctx["context_docs"], "dropped_docs": ctx["dropped_docs"],
                "chunk_fallback_docs": ctx["chunk_fallback_docs"], "context_chars": ctx["context_chars"],
                "context_sha256": ctx["context_sha256"], "stages": stages, "secs": round(time.time() - t0, 1),
            }
            if err:
                row["error"] = err
            await commit(qid, row, ctx, q)

    if published or retrieval_only is False and not to_query and to_finish:
        pass
    if to_query and not published:
        if not api_key:
            raise RuntimeError("HYDRADB_API_KEY is required unless --from-contexts is used")
        async with HydraDBQueryClient(api_key, cfg.hydradb.database, cfg.hydradb.base_url) as client:
            await asyncio.gather(*(one(client, q) for q in to_query))
    elif to_query:
        await asyncio.gather(*(one(None, q) for q in to_query))
    if to_finish:
        if store is None and not published:
            raise RuntimeError("finishing retrieved rows needs a document store (or --from-contexts)")
        await asyncio.gather(*(finish_retrieved(q) for q in to_finish))

    final = [by_id[q] for q in requested_ids if q in by_id]
    want = "retrieved" if retrieval_only else "answered"
    errors = [x["question_id"] for x in final if x["stage"] == "error"]
    missing = [q for q in requested_ids if q not in by_id or by_id[q]["stage"] not in ("answered", want)]
    complete = not missing and not errors
    n = write_answers([x for x in final if x["stage"] != "error"], run_dir / "answers.jsonl")
    manifest.write_manifest(run_dir, "generate", cfg.to_dict(), [run_dir / "answers.jsonl", ckpt, ctx_path],
                            extra={"requested": total, "written": n, "errors": len(errors), "missing": len(missing),
                                   "complete": complete, "retrieval_only": retrieval_only,
                                   "from_contexts": str(from_contexts) if from_contexts else None,
                                   "document_store": store.identity if store else None, "identity": identity,
                                   "resume_notes": notes})
    print(f"\nWrote {n} rows -> {run_dir / 'answers.jsonl'}  requested={total} errors={len(errors)} "
          f"missing={len(missing)} complete={complete}")
    return final, (None if complete else IncompleteRun(missing, errors))
