"""OpenAI provider — async token stream adapter."""

from __future__ import annotations

import os
from typing import AsyncIterator


async def stream_tokens(
    prompt: str,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> AsyncIterator[str]:
    """
    Yield raw text chunks from OpenAI's streaming API.

    Raises:
        ImportError: if `openai` package is not installed.
        openai.AuthenticationError: if the API key is invalid.
    """
    try:
        import openai  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Run: pip install openai") from exc

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    client = openai.AsyncOpenAI(api_key=key)

    stream = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
