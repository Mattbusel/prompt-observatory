"""Anthropic provider — async token stream adapter."""

from __future__ import annotations

import os
from typing import AsyncIterator


async def stream_tokens(
    prompt: str,
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """
    Yield raw text chunks from Anthropic's streaming API.

    Raises:
        ImportError: if `anthropic` package is not installed.
        anthropic.AuthenticationError: if the API key is invalid.
    """
    try:
        import anthropic  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Run: pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=key)

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
