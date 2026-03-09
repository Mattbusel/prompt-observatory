"""
Token stream interceptor — captures token-level data from LLM streams.

Inspired by Every-Other-Token (github.com/Mattbusel/Every-Other-Token).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class TokenEvent:
    """A single token captured from a live LLM stream."""

    index: int
    text: str
    timestamp_ms: float
    # Perplexity proxy: inter-token latency normalized to [0,1].
    # High latency = model is "thinking harder" = higher perplexity signal.
    latency_signal: float = 0.0
    cumulative_text: str = ""


@dataclass
class StreamSession:
    """Accumulated state for one complete LLM response."""

    prompt: str
    model: str
    provider: str
    tokens: list[TokenEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def total_tokens(self) -> int:
        return len(self.tokens)

    @property
    def full_text(self) -> str:
        return "".join(t.text for t in self.tokens)

    @property
    def avg_latency_signal(self) -> float:
        if not self.tokens:
            return 0.0
        return sum(t.latency_signal for t in self.tokens) / len(self.tokens)

    def finish(self) -> None:
        self.finished_at = time.time()


class TokenStreamInterceptor:
    """
    Wraps an async token stream and emits TokenEvent objects.

    Usage::

        async for event in interceptor.intercept(raw_stream):
            yield event  # send to UI
    """

    def __init__(self, smoothing_window: int = 5) -> None:
        self._smoothing_window = smoothing_window
        self._recent_latencies: list[float] = []

    def _latency_signal(self, raw_ms: float) -> float:
        """Normalize latency to [0,1] using a rolling window."""
        self._recent_latencies.append(raw_ms)
        if len(self._recent_latencies) > self._smoothing_window:
            self._recent_latencies.pop(0)
        window_max = max(self._recent_latencies) or 1.0
        return min(raw_ms / window_max, 1.0)

    async def intercept(
        self,
        raw_stream: AsyncIterator[str],
        prompt: str,
        model: str,
        provider: str,
    ) -> AsyncIterator[tuple[TokenEvent, StreamSession]]:
        """
        Yield (TokenEvent, StreamSession) tuples as tokens arrive.

        The StreamSession accumulates; callers can read its state at any point.
        """
        session = StreamSession(prompt=prompt, model=model, provider=provider)
        last_ts = time.perf_counter() * 1000
        index = 0
        cumulative = ""

        async for chunk in raw_stream:
            now = time.perf_counter() * 1000
            latency_ms = now - last_ts
            last_ts = now
            cumulative += chunk

            event = TokenEvent(
                index=index,
                text=chunk,
                timestamp_ms=now,
                latency_signal=self._latency_signal(latency_ms),
                cumulative_text=cumulative,
            )
            session.tokens.append(event)
            index += 1
            yield event, session

        session.finish()
