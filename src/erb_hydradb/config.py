"""Run configuration: one YAML file captures every parameter that can change a result.

The published run's config is ``config/run-2026-09-04.yaml``. A run directory
stores the config it was produced with, and ``fingerprint()`` (a sha256 over
the generation-relevant fields) is embedded in checkpoints so a resume under a
different configuration is refused rather than silently mixed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class HydraDBConfig:
    base_url: str = "https://api.hydradb.com"
    database: str = "erb_appsources"
    collection: str = "entire"


@dataclass
class RetrievalConfig:
    mode: str = "thinking"          # fast | thinking
    alpha: float = 0.5              # hybrid weight (dense vs keyword)
    max_results: int = 100          # chunks requested per query
    query_by: str = "hybrid"
    query_apps: bool = False        # additive app-search lane; off in the published run


@dataclass
class GenerationConfig:
    model: str = "openai/gpt-5.4"   # OpenRouter model id; "gpt-5.4" for direct OpenAI
    provider: str = "openrouter"    # openrouter | openai
    max_tokens: int = 4000
    temperature: float = 0.0
    docs_in_context: int = 12       # full documents given to the answer model
    submit_docs: int = 10           # document ids written to answers.jsonl
    max_context_chars: int = 240_000
    correction_loop: bool = True    # draft -> critique -> rewrite
    prompts_version: str = "two-pass-v1"
    hydration: str = "canonical-v2" # benchmark's own title/content extraction rule
    workers: int = 6


@dataclass
class JudgeConfig:
    model: str = "openai/gpt-5.4"   # the evaluator's default judge
    provider: str = "openrouter"    # openrouter | openai
    # The evaluator's JSON-recovery path calls a second, "cheap" model
    # (upstream default gpt-5-mini). It must be a valid id for the provider.
    cheap_model: str | None = "openai/gpt-5-mini"   # None = leave CHEAP_LLM_MODEL_NAME unset (as the published run did)
    strict_parallelism: int = 6     # threads in the single strict process
    shards: int = 20                # official protocol: concurrent processes
    shard_parallelism: int = 5      # threads per shard


@dataclass
class RunConfig:
    name: str = "run"
    hydradb: HydraDBConfig = field(default_factory=HydraDBConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            name=raw.get("name", Path(path).stem),
            hydradb=HydraDBConfig(**raw.get("hydradb", {})),
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            judge=JudgeConfig(**raw.get("judge", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def generation_fingerprint(self) -> str:
        """sha256 over every field that changes what the generator produces."""
        relevant = {
            "hydradb": {"base_url": self.hydradb.base_url, "database": self.hydradb.database,
                        "collection": self.hydradb.collection},
            "retrieval": asdict(self.retrieval),
            "generation": {k: v for k, v in asdict(self.generation).items() if k != "workers"},
        }
        blob = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
