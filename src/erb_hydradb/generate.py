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

Everything the answer model saw is written to ``contexts.jsonl`` with a sha256,
and every question's full retrieval order (top 50) is kept in the checkpoint,
so any judgment can be traced back to its input and recall can be recomputed
at any depth.

Two variants exist for cheaper reproduction:
  * ``retrieval_only=True`` skips the answer model (retrieval ablations);
  * ``from_contexts=<path>`` skips HydraDB and regenerates answers from a
    published ``contexts.jsonl`` (tests the generator with an LLM key only).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from . import hydrate, manifest
from .config import RunConfig
from .llm import complete
from .prompts import CRITIQUE_PROMPT, PASS1_PROMPT, PASS2_PROMPT, PROMPTS_VERSION


def load_questions(path: str | Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def select_questions(questions: list[dict], targets: str | Path | None, n: int | None) -> list[dict]:
    if targets:
        with open(targets, "r", encoding="utf-8") as f:
            keep = [line.strip() for line in f if line.strip()]
        by_id = {q["question_id"]: q for q in questions}
        questions = [by_id[q] for q in keep if q in by_id]
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


def _load_checkpoint(path: Path, fingerprint: str, force: bool) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if rows and "_config" in rows[0]:
        stored = rows[0]["_config"].get("fingerprint")
        if stored != fingerprint and not force:
            raise SystemExit(
                f"Refusing to resume {path}: it was produced under a different configuration "
                f"(fingerprint {stored} != {fingerprint}). Use a new run directory or --force-resume."
            )
        rows = rows[1:]
    keep = [r for r in rows if not r.get("error") and (r.get("answer") or r.get("_retrieval_only"))]
    if len(keep) != len(rows):
        print(f"  resume: retrying {len(rows) - len(keep)} errored/empty rows", flush=True)
    return keep


def write_answers(rows: list[dict], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"question_id": r["question_id"], "answer": r.get("answer", ""),
                                "document_ids": r.get("document_ids", [])}) + "\n")
    return len(rows)


async def run_generate(cfg: RunConfig, run_dir: Path, questions: list[dict], *,
                       api_key: str | None, store: hydrate.DocumentStore | None,
                       retrieval_only: bool = False, from_contexts: Path | None = None,
                       force_resume: bool = False) -> list[dict]:
    from .hydradb_client import HydraDBQueryClient

    run_dir.mkdir(parents=True, exist_ok=True)
    g, r = cfg.generation, cfg.retrieval
    fingerprint = cfg.generation_fingerprint()
    ckpt = run_dir / "gen_checkpoint.json"
    ctx_path = run_dir / "contexts.jsonl"
    rows = _load_checkpoint(ckpt, fingerprint, force_resume)
    done = {x["question_id"] for x in rows}
    remaining = [q for q in questions if q["question_id"] not in done]
    total = len(done) + len(remaining)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(g.workers)
    counter = {"n": len(done)}

    published_contexts: dict[str, dict] = {}
    if from_contexts:
        with open(from_contexts, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    published_contexts[c["question_id"]] = c

    def save():
        tmp = ckpt.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([{"_config": {"fingerprint": fingerprint, "prompts": PROMPTS_VERSION,
                                    "config": cfg.to_dict()}}] + rows, f, ensure_ascii=False)
        os.replace(tmp, ckpt)

    def dump_context(qid: str, ctx: dict):
        with open(ctx_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": qid, **ctx}, ensure_ascii=False) + "\n")

    async def query(client: HydraDBQueryClient, question: str) -> list[dict]:
        import httpx
        for attempt in range(5):
            try:
                res = await client.query(cfg.hydradb.collection, question, max_results=r.max_results,
                                         mode=r.mode, alpha=r.alpha, query_by=r.query_by,
                                         query_apps=r.query_apps)
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

    async def one(client: HydraDBQueryClient | None, q: dict):
        qid = q["question_id"]
        async with sem:
            t0 = time.time()
            if from_contexts:
                pc = published_contexts.get(qid)
                if pc is None:
                    return
                ctx = {k: pc[k] for k in ("context", "context_docs", "dropped_docs", "chunk_fallback_docs",
                                          "context_chars", "context_sha256")}
                doc_ids = list(pc.get("retrieved_doc_ids") or pc["context_docs"])
                chunks = [{"id": d} for d in doc_ids]
            else:
                try:
                    chunks = await query(client, q["question"])
                except Exception as exc:  # noqa: BLE001
                    async with lock:
                        rows.append({"question_id": qid, "error": f"query: {str(exc)[:150]}",
                                     "answer": "", "document_ids": [], "retrieved_doc_ids": []})
                        counter["n"] += 1
                        save()
                        print(f"[{counter['n']}/{total}] {qid} QUERY ERROR {str(exc)[:100]}", flush=True)
                    return
                doc_ids = hydrate.distinct_docs(chunks)
                ctx = hydrate.build_context(doc_ids, chunks, store, g.docs_in_context, g.max_context_chars)

            answer, stages, gen_error = "", {}, None
            if chunks and not retrieval_only:
                try:
                    answer, stages = await asyncio.to_thread(generate_answer, q["question"], ctx["context"], cfg)
                except Exception as exc:  # noqa: BLE001
                    gen_error = f"gen: {str(exc)[:150]}"
                    print(f"    GEN ERROR {qid}: {str(exc)[:120]}", flush=True)
            row = {
                "question_id": qid,
                "answer": answer,
                "document_ids": doc_ids[:g.submit_docs],
                "retrieved_doc_ids": doc_ids[:50],
                "context_docs": ctx["context_docs"],
                "dropped_docs": ctx["dropped_docs"],
                "chunk_fallback_docs": ctx["chunk_fallback_docs"],
                "context_chars": ctx["context_chars"],
                "context_sha256": ctx["context_sha256"],
                "stages": stages,
                "secs": round(time.time() - t0, 1),
            }
            if retrieval_only:
                row["_retrieval_only"] = True
            if gen_error:
                row["error"] = gen_error
            async with lock:
                rows.append(row)
                counter["n"] += 1
                save()
                if not from_contexts:
                    dump_context(qid, {**ctx, "retrieved_doc_ids": doc_ids[:50]})
                print(f"[{counter['n']}/{total}] {qid} [{q.get('question_type', '?')}] "
                      f"docs={len(ctx['context_docs'])} dropped={len(ctx['dropped_docs'])} "
                      f"ctx={ctx['context_chars']} ans_len={len(answer)} {row['secs']:.0f}s", flush=True)

    if from_contexts:
        await asyncio.gather(*(one(None, q) for q in remaining))
    else:
        if not api_key:
            raise RuntimeError("HYDRADB_API_KEY is required unless --from-contexts is used")
        async with HydraDBQueryClient(api_key, cfg.hydradb.database, cfg.hydradb.base_url) as client:
            await asyncio.gather(*(one(client, q) for q in remaining))

    order = {q["question_id"]: i for i, q in enumerate(questions)}
    rows.sort(key=lambda x: order.get(x["question_id"], 1 << 30))
    n = write_answers(rows, run_dir / "answers.jsonl")
    errors = [x for x in rows if x.get("error")]
    manifest.write_manifest(run_dir, "generate", cfg.to_dict(),
                            [run_dir / "answers.jsonl", ckpt, ctx_path],
                            extra={"questions": n, "errors": len(errors),
                                   "retrieval_only": retrieval_only,
                                   "from_contexts": str(from_contexts) if from_contexts else None,
                                   "document_store": store.backend if store else None,
                                   "generation_fingerprint": fingerprint})
    print(f"\nWrote {n} rows -> {run_dir / 'answers.jsonl'}  (errors: {len(errors)}; "
          f"dropped-doc rows: {sum(1 for x in rows if x.get('dropped_docs'))})")
    return rows
