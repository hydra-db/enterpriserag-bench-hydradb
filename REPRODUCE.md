# Reproducing the result

Five levels, each independent of the ones above it. Pick the one that matches
the keys and time you have. All commands assume the quick-start install from
the README and a `.env` copied from `.env.example`.

| Level | What you reproduce | Needs | Time | Cost |
|---|---|---|---|---|
| 0 | The published statistics from the published per-question rows | nothing | 1 min | 0 |
| 1 | The judge: re-score our published answers with the benchmark evaluator | an LLM key | 10–20 min | ~$8 strict, ~$60 official |
| 2 | The generator: regenerate answers from the exact contexts our model saw, then judge | an LLM key | ~1 h | ~$150 + level 1 |
| 3 | Retrieval: query HydraDB, hydrate, generate, judge | HydraDB key + LLM key + corpus | ~1.5 h | ~$200 |
| 4 | Ingestion: load the corpus into a fresh HydraDB collection, then level 3 | as level 3 | hours | ingest cost + level 3 |

Costs are for GPT-5.4 via OpenRouter at September 2026 prices and are approximate.

## Level 0 — statistics from the published rows (no keys)

```bash
.venv/bin/python -m erb_hydradb verify --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb report --run-dir artifacts/run-2026-09-04 \
    --questions tests/data/questions.jsonl --leaderboard scripts/leaderboard-2026-08-28.json --out RESULTS.md
.venv/bin/python -m erb_hydradb compare \
    --a artifacts/baseline-2026-09-04/official_results_strict.json --answers-a artifacts/baseline-2026-09-04/answers.jsonl \
    --b artifacts/run-2026-09-04/official_results_strict.json --checkpoint-b artifacts/run-2026-09-04/gen_checkpoint.json \
    --questions tests/data/questions.jsonl --expect-n 500
```

Expected: `VERIFIED`; combined 88.73 (official) / 88.30 (strict); paired delta vs
the baseline +24.07 with 95 % CI [20.27, 28.01]; flips F→T 124, T→F 10.

## Setup for levels 1 to 4

```bash
.venv/bin/python -m erb_hydradb setup          # clones EnterpriseRAG-Bench at the pinned commit, verifies questions.jsonl
echo "ERB_REPO=$PWD/EnterpriseRAG-Bench" >> .env
```

The clone uses `--filter=blob:none`; the first evaluator run with the official
protocol will fetch document blobs on demand. If you prefer a full clone, drop
the filter.

Judge provider: set `judge.provider` in the config (or copy the config and edit).

- `openrouter` (what we used): needs `OPENROUTER_API_KEY`; two files in the checkout
  are replaced by `src/erb_hydradb/erb_patches/` so the judge is called through
  OpenRouter chat completions. `erb-hydradb judge` applies and records this.
- `openai`: needs `OPENAI_API_KEY`; the evaluator runs unmodified (Responses API,
  reasoning effort medium). Set `judge.model: gpt-5.4`. This is the maintainers'
  exact path.

## Level 1 — re-judge our published answers

```bash
R=data/runs/rejudge
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R \
    --protocol strict   --answers artifacts/run-2026-09-04/answers.jsonl
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R \
    --protocol official --answers artifacts/run-2026-09-04/answers.jsonl --expect-n 500
.venv/bin/python -m erb_hydradb compare --a artifacts/run-2026-09-04/official_results_strict.json \
    --b $R/official_results_strict.json --questions data/questions.jsonl --expect-n 500
```

Expected tolerance: combined within about ±1.0 of 88.30 (strict) and 88.73
(official); correctness flips in the low single digits; the official protocol's
corrected-question count near 14. A row with `answer_correct: true` and
`completeness_pct: 0` is the evaluator's known failure mode (a single
fact-validation call failed); re-judge that question alone with
`--protocol strict --question-id qst_XXXX --results <separate file>`.

## Level 2 — regenerate answers from our saved contexts

This exercises the answer-generation stage with an LLM key only: the exact
documents our model saw are in `contexts.jsonl.gz`.

```bash
R=data/runs/regen
.venv/bin/python -m erb_hydradb generate --config config/run-2026-09-04.yaml --run-dir $R \
    --from-contexts artifacts/run-2026-09-04/contexts.jsonl.gz
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R --protocol strict
```

Expected: combined within about ±2 of 88.30 (two sources of variance now: the
generator and the judge). Document ids are taken from the saved retrieval order,
so recall is identical to the published run by construction.

## Level 3 — retrieval against HydraDB

Needs `HYDRADB_API_KEY` with access to the database named in the config, and a
document store for hydration: either `erb-hydradb download --from-checkout` (builds
`data/documents.sqlite` from the clone in a few minutes, no Hugging Face needed) or
the checkout itself (used automatically as a fallback).

For verification by the benchmark authors we provide a read-only key to the
published collection on request; anyone else needs level 4 first.

```bash
.venv/bin/python -m erb_hydradb download --from-checkout
R=data/runs/full
.venv/bin/python -m erb_hydradb generate --config config/run-2026-09-04.yaml --run-dir $R
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R --protocol strict
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R --protocol official --expect-n 500
.venv/bin/python -m erb_hydradb report --run-dir $R --questions data/questions.jsonl
.venv/bin/python -m erb_hydradb recall --a artifacts/run-2026-09-04/gen_checkpoint.json --b $R/gen_checkpoint.json --questions data/questions.jsonl
```

Useful variants: `--targets ids.txt` or `--n 50` for a pilot; `--retrieval-only`
to measure recall without any LLM cost (see `recall`); a copy of the config with
`retrieval.mode: fast` for the fast/thinking ablation (checkpoints refuse to
resume under a changed config, so use a new run directory).

Expected: retrieval is deterministic in fast mode (identical top-50 on re-query);
thinking mode varies slightly (recall@10 within about ±1 point). Combined score
within about ±2 of the published numbers.

## Level 4 — ingest the corpus

```bash
.venv/bin/python -m erb_hydradb download --from-checkout            # or: pip install '.[corpus]' and omit the flag for Hugging Face
.venv/bin/python -m erb_hydradb ingest --config config/my-run.yaml --run-dir data/runs/ingest \
    --batch-size 80 --batch-sleep 1.0            # --resume to continue after an interruption; --dry-run to test conversion
```

`config/my-run.yaml` should name a database you own and a fresh collection. The
published run ingested with inference on (`--no-infer` turns it off). The command
converts documents one at a time, sends batches, waits for indexing status per
batch, and writes `ingest_manifest.json` with settled / errored / pending counts.
Query only after the manifest shows no pending ids. Then run level 3 against the
new collection.

## What can and cannot vary

- **Retrieval order** is what HydraDB returned; it is saved per question (top 50)
  so recall at any depth can be recomputed and two runs compared with `recall`.
- **Contexts** are a pure function of the retrieval order and the corpus; the
  sha256 in the checkpoint lets you check that a regenerated context is
  byte-identical to the published one.
- **Answers** vary with the model even at temperature 0.
- **Judgments** vary with the judge; the official protocol can also change the gold
  set for a few questions (14 in our run), which is why both protocols are
  reported.
