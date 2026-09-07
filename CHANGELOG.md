# Changelog

## Unreleased (release candidate for 1.0.0)

Two rounds of external review before publication. Nothing in the published
artifacts changed; the harness around them did.

- Verification cannot falsely succeed: one validator checks files, manifest
  hashes (latest writer wins), schema with finite bounded metrics, unique ids,
  exact coverage, cross-file id equality, answers against the checkpoint,
  retrieval metrics recomputed from the submitted ids and gold sets, every
  statistic, and the actual sha256 of every context; impossible checks report
  INCOMPLETE.
- Generation keeps an attempt history and materialises one authoritative
  `contexts.jsonl.gz`; rows carry stages; incomplete runs exit 2; replayed
  contexts are validated before any model call; continuation reuses the saved
  context; the run identity binds config, prompts, endpoint, questions,
  contexts and document store.
- Ingestion journals every batch before sending, reconciles on resume, keeps
  failed ids, and is ready only when every sent item settled successfully.
- The evaluator runs from immutable, hash-named overlays; the user's checkout is
  never written; official-protocol plans are fingerprinted by inputs, evaluator
  and corpus identity; `--fresh` archives and reruns; runs are locked.
- Recall is set-based everywhere, matching the evaluator (a gold list can repeat
  an id).
- The 14 gold corrections and the judge-stability panel are published and
  audited; provenance separates historical config from reproduction defaults;
  `THIRD_PARTY_NOTICES.md` ships in the wheel; `uv.lock` committed.

## 1.0.0 — 2026-09-07 (internal, not released)

- Initial public release.
- Published run `run-2026-09-04`: canonical full-document hydration (top 12) with
  thinking-mode retrieval, GPT-5.4 two-pass generation, scored with the pinned
  EnterpriseRAG-Bench evaluator under both the strict and the official protocol.
- Baseline `baseline-2026-09-04` (chunk-based generator, fast retrieval) kept for
  the paired comparison.
- Harness: `setup`, `download`, `ingest`, `generate` (with `--retrieval-only` and
  `--from-contexts`), `judge` (strict / official, sharded), `report`, `compare`,
  `recall`, `verify`, `stats`.
