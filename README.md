# HydraDB on EnterpriseRAG-Bench

Reproducible harness and published artifacts for HydraDB's result on
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), Onyx's
benchmark of retrieval-augmented generation over company-internal data
(511,958 exported documents across nine enterprise systems, 500 questions).

| Run of 2026-09-04 (all 500 questions) | Combined score | Correctness | Completeness | Document recall @10 | Invalid extra docs @10 |
|---|---|---|---|---|---|
| Official protocol (citation stripping + document correction) | **88.73** | 92.0 % | 92.53 % | 96.65 % | 8.63 |
| Strict (no correction, no citation stripping) | 88.30 | 91.2 % | 92.93 % | 96.63 % | 8.61 |

The public leaderboard's highest listed score at the time was 80.34 (22 systems,
28 August 2026; snapshot in `scripts/leaderboard-2026-08-28.json`). **This result
is self-reported**: scored with the benchmark's own evaluator, by us, not by the
benchmark authors. Not every category improved over our own earlier pipeline;
`completeness`-type questions score 37.5 (baseline 42.3) and `project_related`
54.1. See [RESULTS.md](RESULTS.md) and "Status and caveats".

Every number above is recomputed from the per-question rows in
[`artifacts/run-2026-09-04/`](artifacts/run-2026-09-04/) by the test suite and by
`erb-hydradb verify`. The artifacts were produced by the scripts this package was
ported from; the port is checked against them (converter identical on all 511,962
source files; all 500 published contexts rebuilt hash-for-hash by
`verify --contexts`; the 14 gold corrections applied by the official protocol are
published and audited). METHODOLOGY.md section 7 has the details and the full run
history.

## Three ways in

**1. Verify the released evidence. No keys, five minutes.**

```bash
git clone https://github.com/hydra-db/enterpriserag-bench-hydradb.git && cd enterpriserag-bench-hydradb
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m erb_hydradb doctor
.venv/bin/python -m pytest
.venv/bin/python -m erb_hydradb verify --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb audit-corrections --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb inspect --run-dir artifacts/run-2026-09-04 --question-id qst_0224
```

Expected: `doctor` ends `READY` (keys are notes, not failures); 64 tests pass;
`verify` lists every check as `OK` and ends `VERIFIED`; the audit reports 14
flagged, 14 records, `ok: true`.

**2. Try two questions. One LLM key, about $1.**

```bash
cp .env.example .env            # add OPENROUTER_API_KEY (or OPENAI_API_KEY and set provider: openai)
printf 'qst_0001\nqst_0224\n' > /tmp/two.txt
.venv/bin/python -m erb_hydradb generate --config config/run-2026-09-04.yaml --run-dir data/runs/two \
    --targets /tmp/two.txt --from-contexts artifacts/run-2026-09-04/contexts.jsonl.gz
```

This regenerates two answers from the exact documents our model saw. Judging
them needs the benchmark checkout (`erb-hydradb setup`, ~5 GB); see
[REPRODUCE.md](REPRODUCE.md) level 1.

**3. Run the benchmark.** [REPRODUCE.md](REPRODUCE.md) has five levels, from
re-judging our answers through re-ingesting the corpus into your own HydraDB
database, with expected outputs, costs, and what is safe to retry.

## What is in this repository

| Path | What |
|---|---|
| `artifacts/run-2026-09-04/` | The published run: `answers.jsonl` (submission format), both evaluator results files, the exact context the answer model saw for every question (`contexts.jsonl.gz`, sha256-linked to the checkpoint), the top-50 retrieval order per question (`gen_checkpoint.json`), the 14 gold corrections the official protocol applied (`corrections.jsonl`), the judge-stability panel (`judge_audit/`), the historical `config.yaml`, `manifest.json`, `SHA256SUMS`. |
| `artifacts/baseline-2026-09-04/` | The earlier chunk-based pipeline (64.23) the run is compared against. Its generator is not part of this package: a record, not a reproducible run. |
| `config/run-2026-09-04.yaml` | Reproduction defaults for the published run. `config/my-run.example.yaml` for your own database. |
| `src/erb_hydradb/` | The harness: corpus build, typed app-source conversion and ingestion, retrieval + generation, evaluator wrapper, validation, paired analysis. |
| `METHODOLOGY.md` | What the pipeline does at each stage, what it does not do, provenance, run history. |
| `REPRODUCE.md` | Reproduction levels, expected outputs, retry rules, troubleshooting. |
| `tests/` | 64 offline tests, including a regression test for every false-success and data-loss case found in review. |
| `THIRD_PARTY_NOTICES.md` | What is redistributed from the benchmark and under which license. |

## The pipeline in one paragraph

The corpus is loaded into one HydraDB collection as typed app sources (Slack
threads as messages, Gmail as email, Jira/Linear/GitHub as tickets with comments,
Confluence and Drive as documents, HubSpot as CRM records) with HydraDB's
ingest-time inference on. For each question, `POST /query` runs a hybrid search in
thinking mode and returns chunks; chunks are collapsed to distinct documents in
rank order; the top 12 documents are hydrated with their canonical text (the
benchmark's own title/content extraction rule, nothing else) and given in full to
GPT-5.4, which drafts, critiques and rewrites the answer; the top 10 document ids
are submitted. Scoring is the unmodified benchmark evaluator. What is HydraDB's and
what is the answer model's is spelled out in [METHODOLOGY.md](METHODOLOGY.md).

## Status and caveats

- **Self-reported.** Not verified or listed by the benchmark authors. Their policy
  is to verify before listing; this repository is the reproduction package for
  that. The wording here will change only when they have done so.
- **A pipeline comparison, not a causal claim.** The +24.07 over our earlier
  pipeline (paired bootstrap 95 % interval [20.27, 28.01], conditional on these two
  runs) reflects several simultaneous changes: whole-document hydration, thinking-mode
  retrieval, and a different submitted depth. It is not an estimate of any one of
  them, of model drift, or of superiority over any leaderboard system.
- **Judge routing.** The published run called the evaluator's GPT-5.4 judge through
  OpenRouter (chat completions). The evaluator's own path is the OpenAI Responses
  API with reasoning effort "medium". Both are supported (`judge.provider`); the
  `openai` provider runs the evaluator unmodified. We have not re-judged with it.
- **Answer model = judge model.** Both GPT-5.4, the evaluator's default. Correctness
  and completeness therefore measure HydraDB's retrieval plus GPT-5.4's writing from
  it; document recall is HydraDB's alone.
- **Temperature.** Generation requests temperature 0. The evaluator's judge calls do
  not set a temperature; whatever the provider's default is applies.
- **Submitted depth.** The benchmark does not fix how many documents a system may
  submit. We submit 10. Recall at 1/3/5/10/20/50 from the same retrieval is in
  `RESULTS.md`.
- **Rerun variation is not well characterised.** On a 24-question repeat-judge panel
  (`artifacts/run-2026-09-04/judge_audit/`) we saw 0 correctness flips and one
  completeness disagreement. Treat the tolerances in REPRODUCE.md as planning
  expectations, not bounds.
- **Pilot overlap.** The configuration was chosen on a 76-question stratified
  subset that is part of the 500. The final run is not a held-out estimate.

## Citing

See [CITATION.cff](CITATION.cff). Please cite the benchmark paper
([arXiv:2605.05253](https://arxiv.org/abs/2605.05253)) alongside this repository.

## License

Code in this repository: MIT, © 2026 HydraDB (`LICENSE`). This repository
**redistributes a subset of the benchmark**: the 500 questions, five sample
documents, the canonical text of the 2,667 documents that appear in our published
contexts, and the gold data for 14 corrected questions, all © 2026 DanswerAI, Inc.
under MIT; see `THIRD_PARTY_NOTICES.md`. The full corpus is not included.
