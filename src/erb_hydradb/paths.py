"""Filesystem layout and pinned upstream identifiers.

Everything the harness reads or writes lives under one data directory
(``ERB_DATA_DIR``, default ``./data``) plus one checkout of the benchmark
repository (``ERB_REPO``). The upstream commit and question-file hash below are
the ones the published result was produced with; ``erb-hydradb verify``
checks a checkout against them.
"""

from __future__ import annotations

import os
from pathlib import Path

# onyx-dot-app/EnterpriseRAG-Bench commit the published run used.
ERB_REPO_URL = "https://github.com/onyx-dot-app/EnterpriseRAG-Bench.git"
ERB_COMMIT = "d36685e273713975ee20299bbf1ab64165575b3c"
# sha256 of questions.jsonl at that commit (500 core questions).
QUESTIONS_SHA256 = "f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905"

HF_DATASET = "onyx-dot-app/EnterpriseRAG-Bench"
HF_QUESTIONS_CONFIG = "questions"
HF_DOCUMENTS_CONFIG = "documents"
HF_SPLIT = "test"


def repo_root() -> Path:
    """The checkout this package was installed from (editable install). Falls
    back to the current directory when installed as a wheel."""
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


def data_dir() -> Path:
    env = os.getenv("ERB_DATA_DIR")
    p = Path(env).expanduser().resolve() if env else repo_root() / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def erb_repo() -> Path | None:
    env = os.getenv("ERB_REPO")
    if env:
        return Path(env).expanduser().resolve()
    candidate = repo_root() / "EnterpriseRAG-Bench"
    return candidate if candidate.exists() else None


def questions_path() -> Path:
    """The official 500-question file. Copied from the ERB checkout by ``setup``."""
    return data_dir() / "questions.jsonl"


def documents_db_path() -> Path:
    return data_dir() / "documents.sqlite"


def runs_dir() -> Path:
    p = data_dir() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"
