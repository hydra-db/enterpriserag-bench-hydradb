# Convenience targets. Every target is a thin wrapper over `erb-hydradb`; see REPRODUCE.md.
PY ?= .venv/bin/python
CFG ?= config/run-2026-09-04.yaml
RUN ?= data/runs/$(shell date +%Y-%m-%d)

.PHONY: venv test lint verify setup download regenerate judge-strict judge-official report

venv:            ## create .venv with pinned dependencies
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -e ".[dev]"

test:            ## offline tests (no keys, no network)
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

verify:          ## level 0: checksums + recompute published stats
	$(PY) -m erb_hydradb verify --run-dir artifacts/run-2026-09-04

setup:           ## clone the benchmark at the pinned commit, verify questions, apply judge patches if needed
	$(PY) -m erb_hydradb setup

download:        ## build data/documents.sqlite from the local checkout (no HF needed)
	$(PY) -m erb_hydradb download --from-checkout

regenerate:      ## level 2: regenerate answers from the published contexts (LLM key only)
	$(PY) -m erb_hydradb generate --config $(CFG) --run-dir $(RUN) --from-contexts artifacts/run-2026-09-04/contexts.jsonl.gz

judge-strict:    ## level 1: re-judge answers in $(RUN) (or ANSWERS=...) with the strict protocol
	$(PY) -m erb_hydradb judge --config $(CFG) --run-dir $(RUN) --protocol strict $(if $(ANSWERS),--answers $(ANSWERS),)

judge-official:  ## official protocol (sharded)
	$(PY) -m erb_hydradb judge --config $(CFG) --run-dir $(RUN) --protocol official $(if $(ANSWERS),--answers $(ANSWERS),)

report:
	$(PY) -m erb_hydradb report --run-dir $(RUN)
