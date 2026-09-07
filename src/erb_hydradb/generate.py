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

Run directory layout and evidence rules:

  gen_checkpoint.json      one row per question with an explicit ``stage``
                           (retrieved | answered | error); header binds the run
                           identity (config fingerprint, prompts, endpoint,
                           questions file hash, contexts file hash for replay,
                           document-store identity)
  contexts.attempts.jsonl  append-only attempt history: every context ever sent
                           to the model, with attempt number and timestamp,
                           written BEFORE the checkpoint row that depends on it
  contexts.jsonl.gz        the authoritative contexts, exactly one per completed
                           question, materialised from the attempt history at
                           the end of every run (the row whose sha256 the
                           checkpoint records)
  answers.jsonl            submission format, one row per non-error question
  manifest.json            one record per stage; file hashes are as of that stage

A run is COMPLETE only when every requested question has an ``answered`` row
(or ``retrieved`` in retrieval-only mode). ``run_generate`` returns the rows and
a completeness verdict; the CLI exits nonzero on an incomplete run unless
``--allow-partial`` is given. Resuming a retrieval-only checkpoint in generation
mode reuses the saved retrieval order and the SAVED context (not a rebuild) and
generates the missing answers; it never counts a ``retrieved`` row as an answer.

Replaying published contexts (``--from-contexts``) validates the file before any
model call: schema, unique ids, coverage of the requested questions, the actual
sha256 of every context, its declared length, and the absence of benchmark
markers.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

from . import hydrate, manifest, validate
from .config import RunConfig
from .llm import complete
from .prompts import CRITIQUE_PROMPT, PASS1_PROMPT, PASS2_PROMPT, PROMPTS_VERSION

ATTEMPTS = "contexts.attempts.jsonl"
CONTEXTS = "contexts.jsonl.gz"


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

def run_identity(cfg: RunConfig, *, questions_path: Path | None, from_contexts: Path | None,
                 store_identity: dict | None = None) -> dict:
    """What a checkpoint is bound to: configuration, prompts, endpoint, the
    questions file, the contexts file (replay) and the document store. The
    operation mode is NOT part of it, so a retrieval-only checkpoint can be
    continued into generation; each row's ``stage`` records what was done."""
    ident = {
        "fingerprint": cfg.generation_fingerprint(),
        "prompts": PROMPTS_VERSION,
        "base_url": cfg.hydradb.base_url,
        "questions_sha256": manifest.sha256_file(questions_path) if questions_path else None,
        "contexts_sha256": manifest.sha256_file(from_contexts) if from_contexts else None,
        "document_store": store_identity or None,
    }
    ident["identity"] = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()
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
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "answer": r.get("answer", ""),
                                "document_ids": r.get("document_ids", [])}) + "\n")
    os.replace(tmp, out)
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


# ------------------------------------------------------- context files ----

def saved_context(run_dir: Path, qid: str, sha256: str | None) -> dict | None:
    """The attempt-history row for ``qid`` whose actual text hashes to ``sha256``
    (or the latest row when no hash is given)."""
    path = run_dir / ATTEMPTS
    if not path.exists():
        return None
    found = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("question_id") != qid:
                continue
            if sha256 is None or validate.sha256_text(r.get("context", "")) == sha256:
                found = r
    return found


def materialise_contexts(run_dir: Path, rows: list[dict]) -> Path:
    """Write ``contexts.jsonl.gz``: one row per completed checkpoint row, the
    attempt whose actual sha256 equals the checkpoint's. Atomic."""
    wanted = {r["question_id"]: r["context_sha256"] for r in rows
              if r.get("stage") in ("retrieved", "answered") and r.get("context_sha256")}
    chosen: dict[str, dict] = {}
    path = run_dir / ATTEMPTS
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                qid = r.get("question_id")
                if qid in wanted and validate.sha256_text(r.get("context", "")) == wanted[qid]:
                    chosen[qid] = {k: r[k] for k in validate.CONTEXT_ROW_KEYS} | {"retrieved_doc_ids": r.get("retrieved_doc_ids", [])}
    out = run_dir / CONTEXTS
    tmp = run_dir / (CONTEXTS + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for qid in wanted:
            if qid in chosen:
                f.write(json.dumps(chosen[qid], ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    missing = sorted(set(wanted) - set(chosen))
    if missing:
        raise RuntimeError(f"{len(missing)} completed rows have no saved context matching the checkpoint: {missing[:5]}")
    return out


# ------------------------------------------------------------- the run ----

async def run_generate(cfg: RunConfig, run_dir: Path, questions: list[dict], *, api_key: str | None,
                       store: hydrate.DocumentStore | None, questions_path: Path | None = None,
                       retrieval_only: bool = False, from_contexts: Path | None = None,
                       force_resume: bool = False, store_identity: dict | None = None) -> tuple[list[dict], IncompleteRun | None]:
    from .hydradb_client import HydraDBQueryClient

    run_dir.mkdir(parents=True, exist_ok=True)
    g, r = cfg.generation, cfg.retrieval
    if store_identity is None and store is not None:
        store_identity = store.identity
    identity = run_identity(cfg, questions_path=questions_path, from_contexts=from_contexts,
                            store_identity=store_identity)
    ckpt = run_dir / "gen_checkpoint.json"
    attempts_path = run_dir / ATTEMPTS
    rows, notes = load_checkpoint(ckpt, identity, force_resume)
    for n in notes:
        print(f"  resume: {n}", flush=True)
    by_id = {x["question_id"]: x for x in rows}
    requested_ids = [q["question_id"] for q in questions]
    done = {qid for qid, x in by_id.items() if x["stage"] == "answered" or (retrieval_only and x["stage"] == "retrieved")}
    to_finish = [q for q in questions if q["question_id"] in by_id and q["question_id"] not in done]
    to_query = [q for q in questions if q["question_id"] not in by_id]
    total = len(requested_ids)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(g.workers)
    counter = {"n": sum(1 for q in questions if q["question_id"] in done)}
    attempt_no = {"n": 0}
    if attempts_path.exists():
        with open(attempts_path, "r", encoding="utf-8") as f:
            attempt_no["n"] = sum(1 for line in f if line.strip())

    published: dict[str, dict] = {}
    if from_contexts:
        published = load_published_contexts(from_contexts, to_query + to_finish)
        print(f"  replaying {len(published)} validated contexts from {from_contexts}", flush=True)

    def save():
        ordered = [by_id[q] for q in requested_ids if q in by_id] + [x for q, x in by_id.items() if q not in set(requested_ids)]
        tmp = ckpt.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([{"_config": {**identity, "config": cfg.to_dict()}}] + ordered, f, ensure_ascii=False)
        os.replace(tmp, ckpt)

    def record_attempt(qid: str, ctx: dict, retrieved: list[str]):
        """Durable evidence first: the context is on disk before the checkpoint
        row that depends on it is written."""
        attempt_no["n"] += 1
        with open(attempts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": qid, "attempt": attempt_no["n"], "recorded_at": manifest.now_iso(),
                                **{k: ctx[k] for k in validate.CONTEXT_ROW_KEYS if k != "question_id"},
                                "retrieved_doc_ids": retrieved}, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

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

    async def commit(qid: str, row: dict, q: dict):
        async with lock:
            by_id[qid] = row
            counter["n"] += 1
            save()
            print(f"[{counter['n']}/{total}] {qid} [{q.get('question_type', '?')}] stage={row['stage']} "
                  f"docs={len(row['context_docs'])} ctx={row['context_chars']} ans_len={len(row['answer'])} "
                  f"{row['secs']:.0f}s" + (f" ERROR {row['error']}" if row.get("error") else ""), flush=True)

    async def finish_retrieved(q: dict):
        """A retrieved row from an earlier retrieval-only run: answer it from
        the SAVED context (the evidence the run recorded), not a rebuild."""
        qid = q["question_id"]
        async with sem:
            t0 = time.time()
            prev = by_id[qid]
            note = None
            if published:
                ctx = context_for(qid, prev["retrieved_doc_ids"], [])
            else:
                saved = saved_context(run_dir, qid, prev.get("context_sha256"))
                if saved is not None:
                    ctx = {k: saved[k] for k in validate.CONTEXT_ROW_KEYS if k != "question_id"}
                else:
                    ctx = hydrate.build_context(prev["retrieved_doc_ids"], [], store, g.docs_in_context, g.max_context_chars)
                    note = "saved context not found; rebuilt from the retrieval order"
            async with lock:
                record_attempt(qid, ctx, prev["retrieved_doc_ids"])
            a, stages, err = await answer(q, ctx)
            row = {**prev, "answer": a, "stages": stages, "secs": round(time.time() - t0, 1),
                   "context_docs": ctx["context_docs"], "context_chars": ctx["context_chars"],
                   "context_sha256": ctx["context_sha256"], "stage": "error" if err else "answered"}
            row.pop("error", None)
            if err:
                row["error"] = err
            if note:
                row["note"] = note
            await commit(qid, row, q)

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
                    await commit(qid, row, q)
                    return
                doc_ids = hydrate.distinct_docs(chunks)
            ctx = context_for(qid, doc_ids, chunks)
            async with lock:
                record_attempt(qid, ctx, doc_ids[:50])
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
            await commit(qid, row, q)

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
    ctx_out = materialise_contexts(run_dir, final)
    manifest.write_manifest(run_dir, "generate", cfg.to_dict(), [run_dir / "answers.jsonl", ckpt, ctx_out],
                            extra={"requested": total, "written": n, "errors": len(errors), "missing": len(missing),
                                   "complete": complete, "retrieval_only": retrieval_only,
                                   "from_contexts": str(from_contexts) if from_contexts else None,
                                   "document_store": store_identity, "identity": identity, "resume_notes": notes,
                                   "attempts_recorded": attempt_no["n"]})
    print(f"\nWrote {n} rows -> {run_dir / 'answers.jsonl'}  requested={total} errors={len(errors)} "
          f"missing={len(missing)} complete={complete}")
    return final, (None if complete else IncompleteRun(missing, errors))
