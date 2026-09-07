# Reproducing the result

Five levels, each independent of the ones above it. Pick the one that matches
the keys and time you have. All commands assume the README install and, from
level 1 up, a `.env` copied from `.env.example`. Run `erb-hydradb doctor` first:
it checks the environment, which keys are present for which stage, the pinned
benchmark files, the corpus, and the published contexts, without spending
anything.

| Level | What you reproduce | Needs | Time | Cost (planning figure) |
|---|---|---|---|---|
| 0 | The published statistics and evidence, from the published files | nothing | 5 min | 0 |
| 1 | The judge: re-score our published answers with the benchmark evaluator | LLM key + benchmark checkout | 10–20 min | ~$8 strict, ~$60 official |
| 2 | The generator: regenerate answers from the exact contexts our model saw | LLM key | ~1 h | ~$150, plus level 1 to judge |
| 3 | Retrieval: query HydraDB, hydrate, generate, judge | HydraDB key + LLM key + checkout | ~1.5 h | ~$200 |
| 4 | Ingestion: load the corpus into your own HydraDB database, then level 3 | as level 3 | hours | ingest cost + level 3 |

Costs are for GPT-5.4 via OpenRouter at September 2026 prices, approximate.

Exit codes everywhere: `0` ok; `1` usage or input problem (nothing was spent);
`2` the command ran but the run is incomplete or failed validation (the output
says what; partial files are kept for inspection).

## Level 0 — the published evidence (no keys)

```bash
.venv/bin/python -m erb_hydradb verify --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb audit-corrections --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb report --run-dir artifacts/run-2026-09-04 \
    --leaderboard scripts/leaderboard-2026-08-28.json \
    --title "HydraDB on EnterpriseRAG-Bench: published run 2026-09-04" --out /tmp/RESULTS.md   # identical to the committed RESULTS.md
.venv/bin/python -m erb_hydradb compare \
    --a artifacts/baseline-2026-09-04/official_results_strict.json --answers-a artifacts/baseline-2026-09-04/answers.jsonl \
    --b artifacts/run-2026-09-04/official_results_strict.json --checkpoint-b artifacts/run-2026-09-04/gen_checkpoint.json \
    --expect-n 500
.venv/bin/python -m erb_hydradb inspect --run-dir artifacts/run-2026-09-04 --question-id qst_0224 --context
```

`verify` enumerates what it checked: file integrity and manifest hashes, schema
and unique ids of every artifact, exact coverage against the 500 questions,
cross-file id equality, every aggregate and per-category statistic recomputed
with the benchmark's formula, and the actual sha256 of every context. Expected:
all `OK`, `VERIFIED`; combined 88.73 (official) / 88.30 (strict); audit 14/14
`ok`; paired delta vs the baseline +24.07 with interval [20.27, 28.01]; flips
F→T 124, T→F 10.

With the benchmark checkout present (after `setup`), `verify --contexts` also
rebuilds all 500 contexts from the saved retrieval order and the corpus and
compares their sha256 against the published ones. Expected: 500 of 500; the
published result of that check is `artifacts/run-2026-09-04/context_equivalence.json`.

## Setup for levels 1, 3 and 4

```bash
.venv/bin/python -m erb_hydradb setup          # clones EnterpriseRAG-Bench at the pinned commit, verifies questions.jsonl
echo "ERB_REPO=$PWD/EnterpriseRAG-Bench" >> .env
.venv/bin/python -m erb_hydradb doctor
```

The benchmark repository contains the corpus as 511,962 files: the clone is
about 5 GB on disk and takes a few minutes. Level 2 does not need it.

Judge provider, set in the config:

- `openrouter` (what we used): needs `OPENROUTER_API_KEY`. The evaluator is run
  from a harness-owned copy of its `src/` tree with two files replaced
  (`src/erb_hydradb/erb_patches/`) so the judge is called through OpenRouter chat
  completions. Your checkout is never modified.
- `openai`: needs `OPENAI_API_KEY`; the evaluator runs unmodified (Responses API,
  reasoning effort medium). Set `judge.model: gpt-5.4` and
  `judge.cheap_model: gpt-5-mini`. This is the maintainers' exact path.

## Level 1 — re-judge our published answers

```bash
R=data/runs/rejudge
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R \
    --protocol strict   --answers artifacts/run-2026-09-04/answers.jsonl --expect-n 500
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R \
    --protocol official --answers artifacts/run-2026-09-04/answers.jsonl --expect-n 500
.venv/bin/python -m erb_hydradb compare --a artifacts/run-2026-09-04/official_results_strict.json \
    --b $R/official_results_strict.json --expect-n 500
```

Planning expectation, not a bound: combined within about ±1 of 88.30 (strict) and
88.73 (official); a handful of correctness flips; the official protocol's
corrected-question count near 14. Our own 24-question repeat-judge panel is in
`artifacts/run-2026-09-04/judge_audit/` (ids in `panel_ids.txt`; selected with
`random.seed(2026)` over the baseline answers): 0 correctness flips, one
completeness disagreement.

A row with `answer_correct: true` and `completeness_pct: 0` is an anomaly worth
re-checking: re-judge that question alone with
`--protocol strict --question-id qst_XXXX --results <separate file>` and report
both outcomes. Do not overwrite the original.

Retries: the official protocol runs as 20 shards. If some fail, re-run the same
command; completed shards are kept and only the missing ones are judged. Use
`--fresh` only to deliberately judge everything again (the previous shard
directory is archived, not deleted). After a successful merge the command writes
`corrections.jsonl` (one record per question whose gold set the protocol
changed, taken from the shard that judged it) and `questions_effective.jsonl`
(the questions with those corrections applied); `verify`, `audit-corrections`
and `report` read the former, so retrieval metrics are checked against the gold
set the scores were computed with. The corpus (checkout HEAD plus any local edit
under `generated_data/`) is recorded at the start of every judge run and checked
again at the end; a change in between fails the run.

## Level 2 — regenerate answers from our saved contexts

The exact documents our model saw are in `contexts.jsonl.gz`. No checkout, no
HydraDB key. The file is validated before any model call (schema, unique ids,
coverage of the requested questions, the actual sha256 of every context, its
declared length, absence of benchmark markers); a bad file fails with exit 1 and
costs nothing.

```bash
R=data/runs/regen
.venv/bin/python -m erb_hydradb generate --config config/run-2026-09-04.yaml --run-dir $R \
    --from-contexts artifacts/run-2026-09-04/contexts.jsonl.gz
```

Then level 1 on `$R/answers.jsonl` to judge. Document ids come from the saved
retrieval order, so recall is identical to the published run by construction.
Planning expectation: combined within about ±2 of 88.30 (generator and judge
variance now both apply).

If some questions fail (provider errors), the command exits 2 and writes the
partial `answers.jsonl`; re-running the same command retries only the failed
rows. `--allow-partial` accepts an incomplete run with exit 0 (recorded in the
manifest as `complete: false`).

## Level 3 — retrieval against HydraDB

Needs `HYDRADB_API_KEY` with access to the database named in the config, the
checkout (for hydration and judging), and optionally `data/documents.sqlite`
(`download --from-checkout`, a few minutes, no Hugging Face needed; the checkout
is used directly otherwise).

For verification by the benchmark authors we provide a read-only key to the
published collection on request. Anyone else needs level 4 first, with their
own database in a copy of `config/my-run.example.yaml`.

```bash
.venv/bin/python -m erb_hydradb download --from-checkout
R=data/runs/full
.venv/bin/python -m erb_hydradb generate --config config/run-2026-09-04.yaml --run-dir $R
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R --protocol strict   --expect-n 500
.venv/bin/python -m erb_hydradb judge --config config/run-2026-09-04.yaml --run-dir $R --protocol official --expect-n 500
.venv/bin/python -m erb_hydradb verify --run-dir $R --contexts
.venv/bin/python -m erb_hydradb report --run-dir $R
.venv/bin/python -m erb_hydradb recall --a artifacts/run-2026-09-04/gen_checkpoint.json --b $R/gen_checkpoint.json
```

Variants: `--targets ids.txt` or `--n 50` for a pilot; `--retrieval-only` to
measure recall with no LLM cost (then `recall`); a copy of the config with
`retrieval.mode: fast` for the fast/thinking ablation. A run directory is bound
to its configuration, questions file, (for replay) contexts file and the content
of the document store; resuming under a different one is refused, so use a new
directory per variant. A run directory can be continued or extended (a larger
selection) but not narrowed: resuming with fewer questions than the checkpoint
holds is refused before anything is written, because `answers.jsonl` and
`contexts.jsonl.gz` would then export fewer rows than the checkpoint. A
retrieval-only run can be continued into generation in the same directory: the
saved retrieval is reused and only the answers are generated.

Planning expectation: fast-mode retrieval re-queries identically; thinking mode
varies slightly (recall@10 within about ±1 point); combined within about ±2.

## Level 4 — ingest the corpus

```bash
cp config/my-run.example.yaml config/my-run.yaml     # edit hydradb.database / collection
.venv/bin/python -m erb_hydradb download --from-checkout
.venv/bin/python -m erb_hydradb ingest --config config/my-run.yaml --run-dir data/runs/ingest \
    --batch-size 80 --batch-sleep 1.0            # --dry-run first to exercise conversion without API calls
```

The command converts documents one at a time, journals every batch before it is
sent, waits for HydraDB's indexing status per batch, and writes
`ingest_manifest.json` with sent / settled / errored / pending counts, the list
of failed item ids, and two flags: `resolved` (every sent item has a terminal
status) and `ready` (every sent item settled successfully, nothing failed).
It exits 2 unless `ready` is true. `--resume` reconciles outstanding items
(re-sends a batch that was interrupted before acknowledgement; ids are
deterministic and ingestion is upsert, so this is safe); `--resume
--retry-failed` re-sends exactly the items HydraDB reported as failed;
`--accept-failures` ends with exit 0 while recording the failed ids and leaving
`ready` false, for a deliberately incomplete corpus. A resume is refused if the
corpus revision, the converter, the source scope or the target changed. Query
only when `ready` is true. Then level 3 against the new collection with the same
config file.

## Troubleshooting

| Symptom | Meaning | What to do |
|---|---|---|
| `doctor` says a key is not set | that stage will refuse to start | add it to `.env`; keys are never printed |
| `checkout ... is at X, not the pinned ...` | wrong benchmark revision | `erb-hydradb setup`, or `--allow-unpinned` (recorded; results not comparable) |
| `questions.jsonl sha256 ... does not match` | wrong questions file | same as above |
| `published contexts rejected` | the contexts file is malformed or incomplete | nothing was spent; check the listed problems |
| `Refusing to resume ...: different run identity` | config, questions, contexts or corpus content changed | new `--run-dir` (or `--force-resume`, recorded) |
| `Refusing to resume ... narrower selection` | the checkpoint has questions this selection drops | select a superset, or a new `--run-dir` |
| `journal: quarantined an incomplete trailing record` | a previous run was interrupted mid-write | nothing to do; the fragment is kept in a `.torn-*` file |
| `corpus changed during judging` | a checkout file under `generated_data/` changed while the evaluator ran | restore the checkout (`git status` in it) and re-run with `--fresh` |
| `is held by pid ...` | another process is working in this run directory | wait, or remove `*.lock` only if that pid is really gone |
| HTTP 401 / 403 | bad key or no access to that database | check `.env` and the database name |
| HTTP 429 | rate limit | the client backs off and retries; slow batches with `--batch-sleep` |
| timeouts / connection errors | transient | safe to re-run every command; generate/ingest/judge resume |
| `INCOMPLETE` (exit 2) | some questions or items failed | re-run to retry failures; inspect `manifest.json` |
| `JUDGE FAILED: shards failed` | some evaluator shards died | re-run the same command; completed shards are kept |
| `PROBLEMS FOUND` from `verify` | an artifact is inconsistent | the report names the check and the ids |

Where things are: every run directory has `manifest.json` (one record per
stage, with hashes of what it wrote at that time), `SHA256SUMS`,
`contexts.attempts.jsonl` (every context ever sent to the model, in order) next
to the authoritative `contexts.jsonl.gz`, and for judging
`protocol_shards/log_*.attempt*.txt` plus `shards.json` (the plan and its
fingerprint). To report a problem, attach the manifest
and the output of `verify --run-dir <dir> --minimal`.

## What can and cannot vary

- **Retrieval order** is what HydraDB returned; it is saved per question (top 50).
- **Contexts** are a pure function of the retrieval order and the corpus; the
  sha256 in the checkpoint lets you check that a regenerated context is
  byte-identical to the published one.
- **Answers** vary with the model even at temperature 0.
- **Judgments** vary with the judge; the official protocol can also change the gold
  set for a few questions (14 in our run, published in `corrections.jsonl`), which
  is why both protocols are reported.
