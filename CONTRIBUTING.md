# Contributing

This repository exists so the published result can be reproduced and challenged.
Changes that make that easier are welcome; changes that alter what the published
run did are not, unless they are recorded as a new run.

- **Do not edit `artifacts/`.** Those files are the record. `erb-hydradb verify`
  and the test suite fail if their checksums or recomputed statistics change.
- **Do not edit `src/erb_hydradb/prompts.py` in place.** Add a new version constant
  and reference it from a new run config; the generation fingerprint depends on it.
- **A new run** goes in `artifacts/<run-name>/` with its `config.yaml`,
  `answers.jsonl`, evaluator results, `gen_checkpoint.json`, `contexts.jsonl.gz`,
  and the `manifest.json` / `SHA256SUMS` that `erb-hydradb` writes. Regenerate
  `RESULTS.md` with `erb-hydradb report`.
- **Tests must stay offline.** Anything that needs a key belongs in REPRODUCE.md,
  not in `tests/`.
- **Never commit `.env`**, keys, or a HydraDB tenant's credentials. CI greps for
  common key prefixes and fails on a match.
- Run `make test lint verify` before opening a pull request.

Found a problem with the run itself (a leaked field, a scoring mistake, a
mismatch with the benchmark's protocol)? Open an issue with the question id and
the file it is in. That is the most useful contribution this repository can get.
