# Changelog

## 1.0.0 — 2026-09-07

- Initial public release.
- Published run `run-2026-09-04`: canonical full-document hydration (top 12) with
  thinking-mode retrieval, GPT-5.4 two-pass generation, scored with the pinned
  EnterpriseRAG-Bench evaluator under both the strict and the official protocol.
- Baseline `baseline-2026-09-04` (chunk-based generator, fast retrieval) kept for
  the paired comparison.
- Harness: `setup`, `download`, `ingest`, `generate` (with `--retrieval-only` and
  `--from-contexts`), `judge` (strict / official, sharded), `report`, `compare`,
  `recall`, `verify`, `stats`.
