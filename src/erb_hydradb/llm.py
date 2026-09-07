"""Answer-model calls: one function, two providers, deterministic settings.

The published run used ``openai/gpt-5.4`` through OpenRouter's OpenAI-compatible
endpoint. A direct OpenAI key works the same way with ``provider: openai`` and
model ``gpt-5.4``. Temperature is 0; retries cover transient transport and
rate-limit failures only.
"""

from __future__ import annotations

import os
import time

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _client(provider: str):
    from openai import OpenAI

    if provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAI(api_key=key)
    raise ValueError(f"unknown provider {provider!r}")


def complete(prompt: str, *, model: str, provider: str = "openrouter", max_tokens: int = 4000,
             temperature: float = 0.0, attempts: int = 5) -> str:
    """Single-turn chat completion; returns the text (stripped, possibly empty)."""
    from openai import Timeout

    client = _client(provider)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=Timeout(timeout=120.0, connect=30.0),
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status_code", None)
            transient = (
                isinstance(exc, (ConnectionError, TimeoutError, OSError))
                or status in (429, 500, 502, 503, 504)
                or "Connection error" in str(exc)
                or "timeout" in str(exc).lower()
            )
            if not transient or attempt == attempts - 1:
                raise
            last = exc
            time.sleep(min(5 * (2 ** attempt), 60))
    raise last or RuntimeError("LLM call failed")


def provider_available(provider: str) -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"))
