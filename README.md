# HydraDB on EnterpriseRAG-Bench

Reproducible harness and published artifacts for HydraDB's result on
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), Onyx's
benchmark of retrieval-augmented generation over company-internal data
(511,958 documents across nine enterprise systems, 500 questions).

| Run of 2026-09-04 (all 500 questions) | Combined score | Correctness | Completeness | Document recall @10 | Invalid extra docs @10 |
|---|---|---|---|---|---|
| Official protocol (citation stripping + document correction) | **88.73** | 92.0 % | 92.53 % | 96.65 % | 8.63 |
| Strict (no correction, no citation stripping) | 88.30 | 91.2 % | 92.93 % | 96.63 % | 8.61 |

The public leaderboard's highest listed score at the time was 80.34 (22 systems,
28 August 2026; snapshot in `scripts/leaderboard-2026-08-28.json`). **This result
is self-reported.** It was scored with the benchmark's own evaluator, but by us, not
by the benchmark authors; see "Status and caveats" below.

Every number above is recomputed from the per-question rows in
[`artifacts/run-2026-09-04/`](artifacts/run-2026-09-04/) by the test suite and by
`erb-hydradb verify`. [`RESULTS.md`](RESULTS.md) is generated from those files.

## What is in this repository

| Path | What |
|---|---|
| `artifacts/run-2026-09-04/` | The published run: `answers.jsonl` (submission format), both evaluator results files, the exact context the answer model saw for every question (`contexts.jsonl.gz`, sha256-linked to the checkpoint), the top-50 retrieval order per question (`gen_checkpoint.json`), `manifest.json`, `SHA256SUMS`. |
| `artifacts/baseline-2026-09-04/` | The earlier chunk-based pipeline (64.23) the run is compared against in `RESULTS.md`. |
| `config/run-2026-09-04.yaml` | Every parameter of the published run. |
| `src/erb_hydradb/` | The harness: corpus build, typed app-source conversion and ingestion into HydraDB, retrieval + generation, evaluator wrapper, paired analysis. |
| `METHODOLOGY.md` | Exactly what the pipeline does at each stage, and what it does not do. |
| `REPRODUCE.md` | Five reproduction levels, from "no keys, five minutes" to "re-ingest the corpus", with expected outputs and tolerances. |
| `tests/` | Offline tests: canonical extraction, leak assertion, conversion, the scoring formula against the published files. |

## Quick start (no API keys)

```bash
git clone https://github.com/hydra-db/enterpriserag-bench-hydradb.git
cd enterpriserag-bench-hydradb
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest                                     # offline tests
.venv/bin/python -m erb_hydradb verify --run-dir artifacts/run-2026-09-04
.venv/bin/python -m erb_hydradb stats  --results artifacts/run-2026-09-04/official_results_protocol.json
```

`verify` checks every artifact's checksum and recomputes the evaluator's
aggregate and per-category statistics from the per-question rows with the
benchmark's own formula. That is reproduction level 0. Levels 1 to 4 (re-judge
our answers, regenerate answers from our saved contexts, re-run retrieval against
HydraDB, re-ingest the corpus) are in [REPRODUCE.md](REPRODUCE.md).

## The pipeline in one paragraph

The corpus is loaded into one HydraDB collection as typed app sources (Slack
threads as messages, Gmail as email, Jira/Linear/GitHub as tickets with comments,
Confluence and Drive as documents, HubSpot as CRM records) with HydraDB's
ingest-time inference on. For each question, `POST /query` runs a hybrid search in
thinking mode and returns chunks; chunks are collapsed to distinct documents in
rank order; the top 12 documents are hydrated with their canonical text (the
benchmark's own title/content extraction rule, nothing else) and given in full to
GPT-5.4, which drafts, critiques and rewrites the answer; the top 10 document ids
are submitted. Scoring is the unmodified benchmark evaluator. Details, including
what is HydraDB's and what is the answer model's, are in
[METHODOLOGY.md](METHODOLOGY.md).

## Status and caveats

- **Self-reported.** Not yet verified or listed by the benchmark authors. Their
  policy is to verify before listing; this repository is the reproduction package
  for that.
- **Judge routing.** The published run called the evaluator's GPT-5.4 judge through
  OpenRouter (chat completions). The evaluator's own path is the OpenAI Responses
  API with reasoning effort "medium". Both are supported here (`judge.provider`);
  the `openai` provider runs the evaluator unmodified and is the exact official
  path. We have not yet re-judged with it.
- **Answer model = judge model.** Both are GPT-5.4. That is the evaluator's default,
  not a choice of ours. Correctness and completeness therefore measure HydraDB's
  retrieval plus GPT-5.4's writing from it; document recall is HydraDB's alone.
- **Submitted depth.** The benchmark does not fix how many documents a system may
  submit. We submit 10; recall figures across systems depend partly on that choice.
  Recall at 1/3/5/10/20/50 from the same retrieval is in `RESULTS.md`.
- **LLM nondeterminism.** Temperature is 0 everywhere but outputs still vary. On a
  24-question repeat-judge panel we observed 0 correctness flips and one
  completeness disagreement; expect re-runs to land within about ±1 point on the
  combined score. See REPRODUCE.md for tolerances.

## Citing

See [CITATION.cff](CITATION.cff). Please cite the benchmark paper
([arXiv:2605.05253](https://arxiv.org/abs/2605.05253)) alongside this repository.

## License

MIT for the code in this repository. The benchmark, its questions and corpus are
Onyx's, released under their MIT license; this repository does not redistribute the
corpus. `artifacts/` contains our answers and the evaluator's outputs on them.
