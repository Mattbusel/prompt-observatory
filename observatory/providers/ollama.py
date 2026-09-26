"""Ollama provider: stream from a model running locally, no API key needed.

Uses Ollama's OpenAI-compatible endpoint (http://localhost:11434/v1 by default,
or OLLAMA_HOST). Model names in the dashboard look like ``ollama/qwen2.5:0.5b``.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

PREFIX = "ollama/"


def base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip() or "http://localhost:11434"
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    if host.startswith("http://0.0.0.0"):
        host = host.replace("0.0.0.0", "127.0.0.1", 1)
    return host.rstrip("/")


def list_models(timeout: float = 0.6) -> list[str]:
    """Names of the models installed in a running Ollama, or [] if it is not running."""
    try:
        import httpx

        r = httpx.get(base_url() + "/api/tags", timeout=timeout)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


async def stream_tokens(
    prompt: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """Yield raw text chunks from a local Ollama model."""
    import openai  # type: ignore[import]

    name = model[len(PREFIX):] if model.startswith(PREFIX) else model
    client = openai.AsyncOpenAI(base_url=base_url() + "/v1", api_key="ollama", max_retries=0)
    stream = await client.chat.completions.create(
        model=name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
