# Third-party notices

## EnterpriseRAG-Bench (Onyx / DanswerAI, Inc.) — MIT

This repository redistributes material from
[onyx-dot-app/EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
at commit `d36685e273713975ee20299bbf1ab64165575b3c`, and contains code derived
from it. Scope:

- `tests/data/questions.jsonl` — the 500 official questions, verbatim.
- `tests/data/erb_repo/` — five sample corpus documents, verbatim, as test fixtures.
- `artifacts/run-2026-09-04/contexts.jsonl.gz` — the canonical text (title and
  content) of 2,667 corpus documents, i.e. the exact context our answer model
  saw. This is a **subset of the benchmark corpus, redistributed so the run can
  be audited**; the full corpus is not included and is obtained from Onyx's
  release by `erb-hydradb setup` / `download`.
- `artifacts/run-2026-09-04/corrections.jsonl` — gold answers and answer facts
  for 14 questions, before and after the evaluator's correction step.
- `src/erb_hydradb/corpus.py` (`extract_document_content`) mirrors
  `src/utils/document_content.py`; `src/erb_hydradb/erb_patches/` are modified
  copies of `src/llm/openai_llm.py` and `src/llm/factory.py`; `analysis.py`
  reimplements `compute_stats_for_group`.

The upstream license, reproduced as required:

```
MIT License

Copyright (c) 2026 DanswerAI, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Everything else in this repository is © 2026 HydraDB, MIT (see `LICENSE`).
