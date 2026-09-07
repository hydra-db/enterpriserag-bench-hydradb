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

The corpus (511,962 source files in the checkout; 511,958 documents in the benchmark's export, which skips files lacking the required fields) was loaded into **one** HydraDB database/collection
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

The header of each block carries the document's benchmark id (`dsid_…`). The
answer model therefore sees document ids, and 24 of the 500 published answers
mention one. These are the ids HydraDB returned, never gold ids (the pipeline
reads only the `question` field of each question until scoring), but note the
interaction with the two judge settings: the official protocol strips citations
before judging, the strict setting does not.

## 5. Answer generation (GPT-5.4)

Three calls per question with `openai/gpt-5.4` via OpenRouter, temperature 0
(requested explicitly for generation; the evaluator's judge calls set none),
`max_tokens` 4000, prompts verbatim in `src/erb_hydradb/prompts.py`
(version `two-pass-v1`):

1. **Draft** from the documents, instructed to state every specific fact relevant
   to the question and to say when the documents do not contain part of the answer.
2. **Critique** of the draft against the same documents (missing / wrong /
   unsupported items).
3. **Rewrite** incorporating the critique.

If the critique or rewrite call fails, the draft is used. No per-question or
per-category logic exists anywhere in the pipeline.

The prompts are written for this evaluator: the draft prompt tells the model
that the answer "will be graded fact-by-fact against a gold reference" and asks
for every specific value, id and step. That is a legitimate but deliberate
choice, and it is why completeness is high; the prompts are published verbatim
so it can be judged.

## 6. Scoring (benchmark evaluator)

`src/scripts/answer_evaluation/metrics_based_eval.py` from the pinned checkout,
invoked as a subprocess. The harness runs it from its own copy of the
checkout's `src/` tree (`data/evaluator/<provider>/`, with the corpus and
questions linked in), so the user's checkout is never modified and the exact
evaluator bytes used are hashed into the manifest. Two settings are reported:

- **Strict**: `--no-correction --skip-citation-stripping`. The gold set is never
  changed, so runs are directly comparable to each other.
- **Official protocol**: citation stripping and the three-judge document
  correction flow, which may add or remove gold documents and regenerate the gold
  answer and facts for a question (14 of 500 in the published run; every
  correction, with the before/after gold documents, gold answer, answer facts and
  the judges' reasons, is published in `corrections.jsonl` and checked by
  `erb-hydradb audit-corrections`). Run as 20 concurrent shards of 25 questions
  for throughput and fault isolation (the evaluator supports `--resume`, which
  retries use), and merged with the evaluator's own `compute_stats_for_group`.
  The citation-stripped answer texts the judge actually scored were not retained
  by the originating run.

Judge model: GPT-5.4, the evaluator's default. The published run called it
through OpenRouter chat completions (two files in the checkout replaced by
`src/erb_hydradb/erb_patches/`); the evaluator's own path is the OpenAI Responses
API with reasoning effort "medium" and is available here with
`judge.provider: openai`.

Per-question outputs: `answer_correct`, `completeness_pct`, `document_recall_pct`,
`invalid_extra_docs`, `correctness_reasoning`. Leaderboard score =
mean over questions of (`completeness_pct` if `answer_correct` else 0).

Known evaluator behaviour: if any single fact-validation call raises, the
question's completeness is recorded as 0 with no marker, and the same row is
produced when every fact is legitimately judged unsupported. The published run
has one row judged correct with 0 % completeness (`qst_0242`); a separate
re-judge of the same answer gave 100 %
(`artifacts/run-2026-09-04/judge_audit/v3_rejudge_anomaly.json`). That shows
instability on this question; it does not establish the cause. Published
numbers are the unadjusted run.

The evaluator defines a second, "cheap" model (`CHEAP_LLM_MODEL_NAME`,
upstream default `gpt-5-mini`) used by its JSON-recovery helper. The published
run did not set it. Offline inspection of the pinned evaluator shows that the
scoring paths used here (holistic correctness, fact validation, document
evaluation, fact regeneration) do not call that helper, so this setting is not
a plausible cause of the anomaly above. The harness still normalises it per
provider (`judge.cheap_model`) so that any future evaluator path that does use
it has a valid model id; the historical config records it as unset.

Statistics outside the evaluator (`verify`, `stats`, `report`, `compare`) are
computed by `analysis.py`, a reimplementation of the evaluator's
`compute_stats_for_group` (same arithmetic and rounding; the test suite checks
that it reproduces both published results files to the cent). Merging
official-protocol shards uses the evaluator's own function.

## 7. Provenance

**The published artifacts were produced by the scripts this package was ported
from, not by this package.** The originating scripts lived in an internal
evaluation repository; this package is a self-contained port of them, made so
that the run can be reproduced without that repository. Visible consequences in
the artifacts: the checkpoint's `_config` header has no fingerprint (the
originating script did not write one), the context rows carry no
`retrieved_doc_ids` (they are in the checkpoint), the official-protocol results
file's `merged_from` paths are relative to the originating layout, and the
answers are in the originating script's question order (a difficulty ordering)
rather than file order. Runs made with this package differ in exactly those
respects and in nothing that affects scores.

Equivalence between the port and the originating code was checked and is
checkable:

- `convert.py` produces output identical to the original converter for every
  one of the 511,962 source files in the pinned checkout (6,038,192 typed items,
  compared as sorted JSON and on per-item key order).
- `erb-hydradb verify --contexts` rebuilds every published context from the
  saved retrieval order and the corpus and compares sha256 with both the
  checkpoint and `contexts.jsonl.gz`. The result for the published run is in
  `artifacts/run-2026-09-04/context_equivalence.json`: 500 of 500.
- The prompts are the originating prompts verbatim; the retrieval request body
  is the same field for field (`hydradb_client.py`).

For runs made with this package, every stage appends to `manifest.json` in the
run directory (package version, repository commit, Python version, the full
config, the pinned and the actual upstream identifiers, sha256 of every file it
produced as of that stage; the file on disk must match the latest record that
names it), and `SHA256SUMS` covers the directory. Generation keeps an
append-only attempt history (`contexts.attempts.jsonl`, written before the
checkpoint row that depends on it) and materialises `contexts.jsonl.gz` with
exactly one authoritative context per completed question at the end of every
run. The checkpoint carries a run identity (configuration fingerprint, prompts,
endpoint, questions file hash, contexts file hash for replay, document-store
identity) and refuses to resume under a different one. The published run's
manifest records are marked `retrospective`.

### Run history

Everything that was run against this collection before the published result,
so that "88.73" can be read as what it is: one full run, not a best-of.

| Date | Scope | Pipeline | Combined (strict) | Kept as |
|---|---|---|---|---|
| 2026-09-04 | 500 | top-12 chunks, fast retrieval, up to 50 ids | 64.23 | `artifacts/baseline-2026-09-04` |
| 2026-09-04 | 76-question pilot | full documents (raw JSON, later found to render benchmark-internal fields), fast | 80.00 | not published |
| 2026-09-04 | 76-question pilot | full documents (raw JSON), thinking | 88.49 | not published |
| 2026-09-04 | 76-question pilot | full documents (canonical), thinking | 88.42 | not published |
| 2026-09-04 | 500 (stopped at 22) | full documents (raw JSON), thinking | never judged | discarded |
| 2026-09-04 | 500 | full documents (canonical), thinking | 88.30 strict / 88.73 official | **`artifacts/run-2026-09-04`** |

The pilot subset (76 questions, stratified by category) was used only to choose
the configuration; the 500-question run was then made once with it. The pilot
questions are not held out from the 500, so the final run is not a held-out
estimate. The +24.07 paired improvement over the baseline is a comparison of two
pipelines that differ in hydration, retrieval mode and submitted depth at once;
its bootstrap interval is conditional on these two runs and says nothing about
model drift, another rerun, or other systems.

### Judge stability panel

`artifacts/run-2026-09-04/judge_audit/`: 24 baseline answers (ids in
`panel_ids.txt`, drawn with `random.seed(2026)`) judged a second time under the
strict setting: 0 correctness flips, one completeness disagreement of 33 points
on one question. This is a small panel; it characterises the judge only roughly.

### What HydraDB received

HydraDB's ingestion received only the converted items (`convert.py`): the
benchmark's exported title and content of each document, split into typed
items with the fields listed in section 2, and a flat metadata dictionary. It
received no questions, no gold answers, no gold document ids, and none of the
benchmark-internal annotation fields.

## 8. What this harness does not do

- It does not tune anything per question or per category.
- It does not consult gold document ids or gold answers at any point before
  scoring; hydration resolves only the ids HydraDB returned.
- It does not redistribute the full corpus; `download` rebuilds `documents.sqlite`
  from the benchmark's own release. It does redistribute a subset (the canonical
  text of the 2,667 documents in the published contexts, the questions, five
  sample documents, and the gold data of 14 corrected questions); see
  `THIRD_PARTY_NOTICES.md`.
- It does not attribute the score to ingestion-time inference, to the reranking
  mode, or to hydration individually. The pilot ablation that led to this
  configuration (chunks/fast 58.66 → full documents/fast 80.0 → full
  documents/thinking 88.42 on a 76-question stratified subset) is described in
  the accompanying report; the harness supports re-running it with
  `--targets`, a copied config, and `compare`.
