# Methodology

What the harness does at each stage of the published run, stated so that a
reader can tell exactly which parts are HydraDB, which are the answer model, and
which are the benchmark's own evaluator.

## 1. Benchmark

EnterpriseRAG-Bench (Onyx, 2026; [paper](https://arxiv.org/abs/2605.05253),
[repository](https://github.com/onyx-dot-app/EnterpriseRAG-Bench),
[dataset](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)).
Pinned in this harness:

- repository commit `d36685e273713975ee20299bbf1ab64165575b3c`;
- `questions.jsonl` sha256 `f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905`
  (500 core questions; the 100 metadata-dependent extra questions are excluded
  from the leaderboard and were not run).

470 of the 500 questions carry gold document ids; `high_level` (10) and
`info_not_found` (20) do not, by design.

## 2. Corpus and ingestion (HydraDB)

The 511,958 documents were loaded into **one** HydraDB database/collection
(`erb_appsources` / `entire`) as **typed app sources** via `POST /context/ingest`
(`type=knowledge`, `app_knowledge` array), not as generic documents. The
conversion is deterministic (`src/erb_hydradb/convert.py`):

| Benchmark source | HydraDB kind | provider | Items per document |
|---|---|---|---|
| slack | `message` | slack | one per parsed message, ids `<doc_id>_m0000…`; `thread_id` = first message, `parent_id` = previous message |
| gmail | `email` | gmail | one per parsed message, ids `<doc_id>_e00…` when the thread has more than one message; reply pointer in `in_reply_to` |
| jira, linear, github | `ticket` + `comment` | jira / linear / github | `<doc_id>` for the ticket, `<doc_id>_c00…` for each comment with `parent_id` = the ticket |
| confluence | `knowledge_base` | notion | one; revision history parsed into `metadata.revisions`, timestamp = latest revision |
| google_drive, fireflies | `knowledge_base` | google_drive / fireflies | one |
| hubspot | `custom` | hubspot | one |
| (empty content) | | | no items |

Slack transcripts carry no per-message time, so message *i* gets a synthetic,
deterministic timestamp (2025-01-01T00:00:00Z + 45 s × *i*); only the ordering
matters. Full detail is in the `convert.py` module docstring, and
`tests/test_convert.py` pins the behaviour on sample documents of every type.

Choices a reader should know:

- Each ERB document keeps its benchmark id (`dsid_…`) as the HydraDB id; typed
  sub-items get a suffix (`_m0001` for a Slack message, `_eNN` for an email in a
  thread, `_cNN` for a comment). The evaluator sees only the stripped `dsid_…`.
- Ingestion ran with HydraDB's inference step on (`infer=true`), which extracts
  people, thread structure and cross-references from each item at ingest time.
  This harness does not measure what inference contributes; an on/off comparison
  would need two ingests of the same corpus.
- Titles are capped at 1000 characters (vector-store field limit).
- The corpus content given to HydraDB is the benchmark's export (title + content),
  the same text used for hydration in step 4.

## 3. Retrieval (HydraDB)

For each question, one `POST /query` with:

| Parameter | Value |
|---|---|
| `type` | `knowledge` |
| `query_by` | `hybrid` (dense + keyword) |
| `mode` | `thinking` (adds a reranking pass over the candidate set; `fast` is the other mode) |
| `alpha` | 0.5 |
| `max_results` | 100 chunks |
| `graph_context` | false |
| `query_apps` | not sent (server default, false) |

Returned chunks are collapsed to **distinct documents in rank order** by
stripping the typed-item suffix from each chunk id. The first 50 distinct ids are
saved per question (`gen_checkpoint.json` → `retrieved_doc_ids`); the first 10 are
submitted for scoring (`answers.jsonl` → `document_ids`); the first 12 go to the
answer model.

The benchmark does not fix the submitted depth. Recall at other depths is
recomputed from the saved order in `RESULTS.md`.

## 4. Hydration (harness)

The answer model receives the **full canonical text** of the top 12 retrieved
documents, in rank order, formatted as

```
[document 1 | source=<source_type> | title=<title> | id=<dsid_…>]
<content>

=====

[document 2 | …]
```

"Canonical" means exactly what the benchmark exports for a document: the value
of its `title_field_name` field and its `content_field_names` fields joined by
the benchmark's own rule (`src/utils/document_content.py` in the ERB repository;
mirrored in `hydrate.extract_document_content`). Nothing else from the raw
corpus files is rendered. The raw files carry benchmark-internal annotations
(`dataset_doc_uuid`, `dataset_noise_document`, the field-name labels); the harness
raises `ContextLeakError` if any of them appears in a context, and the test suite
checks every published context for them.

A total budget of 240,000 characters applies; whole documents are dropped from
the tail of the ranking, never cut mid-document, and drops are recorded
(none occurred in the published run; median context 84k characters, max 188k).

## 5. Answer generation (GPT-5.4)

Three calls per question with `openai/gpt-5.4` via OpenRouter, temperature 0,
`max_tokens` 4000, prompts verbatim in `src/erb_hydradb/prompts.py`
(version `two-pass-v1`):

1. **Draft** from the documents, instructed to state every specific fact relevant
   to the question and to say when the documents do not contain part of the answer.
2. **Critique** of the draft against the same documents (missing / wrong /
   unsupported items).
3. **Rewrite** incorporating the critique.

If the critique or rewrite call fails, the draft is used. No per-question or
per-category logic exists anywhere in the pipeline.

## 6. Scoring (benchmark evaluator)

`src/scripts/answer_evaluation/metrics_based_eval.py` from the pinned checkout,
invoked as a subprocess from the checkout root. Two settings are reported:

- **Strict**: `--no-correction --skip-citation-stripping`. The gold set is never
  changed, so runs are directly comparable to each other.
- **Official protocol**: citation stripping and the three-judge document
  correction flow, which may add or remove gold documents and regenerate the gold
  answer and facts for a question (14 of 500 in the published run). Run as 20
  concurrent shards of 25 questions (the evaluator writes results only at the end
  of a process) and merged with the evaluator's own `compute_stats_for_group`.

Judge model: GPT-5.4, the evaluator's default. The published run called it
through OpenRouter chat completions (two files in the checkout replaced by
`src/erb_hydradb/erb_patches/`); the evaluator's own path is the OpenAI Responses
API with reasoning effort "medium" and is available here with
`judge.provider: openai`.

Per-question outputs: `answer_correct`, `completeness_pct`, `document_recall_pct`,
`invalid_extra_docs`, `correctness_reasoning`. Leaderboard score =
mean over questions of (`completeness_pct` if `answer_correct` else 0).

Known evaluator behaviour: if any single fact-validation call raises, the
question's completeness is recorded as 0 with no marker. The published run has
one such row (`qst_0242`, correct, 0 %); re-judging it gave 100 %. Published
numbers are the unadjusted run.

## 7. Provenance

Every stage writes to `manifest.json` in the run directory: package version,
repository commit, Python version, the full config, the pinned upstream
identifiers, and sha256 of every file it produced; `SHA256SUMS` covers the whole
directory. The generation checkpoint carries a configuration fingerprint and
refuses to resume under a different configuration.

## 8. What this harness does not do

- It does not tune anything per question or per category.
- It does not consult gold document ids or gold answers at any point before
  scoring; hydration resolves only the ids HydraDB returned.
- It does not redistribute the corpus; `download` rebuilds `documents.sqlite`
  from the benchmark's own release.
- It does not attribute the score to ingestion-time inference, to the reranking
  mode, or to hydration individually. The pilot ablation that led to this
  configuration (chunks/fast 58.66 → full documents/fast 80.0 → full
  documents/thinking 88.42 on a 76-question stratified subset) is described in
  the accompanying report; the harness supports re-running it with
  `--targets`, a copied config, and `compare`.
