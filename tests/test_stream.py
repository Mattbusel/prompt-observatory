"""Tests for the token stream interceptor."""

from __future__ import annotations

import asyncio
import pytest
from observatory.core.stream import TokenStreamInterceptor, StreamSession


async def _fake_stream(chunks: list[str]):
    for chunk in chunks:
        await asyncio.sleep(0.001)
        yield chunk


@pytest.mark.asyncio
async def test_interceptor_collects_all_tokens():
    chunks = ["Hello", ", ", "world", "!"]
    interceptor = TokenStreamInterceptor()
    events = []
    async for event, session in interceptor.intercept(
        _fake_stream(chunks), "test prompt", "test-model", "test-provider"
    ):
        events.append(event)

    assert len(events) == 4
    assert events[0].text == "Hello"
    assert events[3].text == "!"


@pytest.mark.asyncio
async def test_interceptor_assigns_sequential_indices():
    chunks = ["a", "b", "c"]
    interceptor = TokenStreamInterceptor()
    indices = []
    async for event, _ in interceptor.intercept(
        _fake_stream(chunks), "p", "m", "p"
    ):
        indices.append(event.index)
    assert indices == [0, 1, 2]


@pytest.mark.asyncio
async def test_interceptor_latency_signal_bounded():
    chunks = ["x"] * 20
    interceptor = TokenStreamInterceptor()
    async for event, _ in interceptor.intercept(
        _fake_stream(chunks), "p", "m", "p"
    ):
        assert 0.0 <= event.latency_signal <= 1.0


@pytest.mark.asyncio
async def test_interceptor_cumulative_text_grows():
    chunks = ["foo", "bar"]
    interceptor = TokenStreamInterceptor()
    seen = []
    async for event, _ in interceptor.intercept(
        _fake_stream(chunks), "p", "m", "p"
    ):
        seen.append(event.cumulative_text)
    assert seen[0] == "foo"
    assert seen[1] == "foobar"


@pytest.mark.asyncio
async def test_session_full_text():
    chunks = ["The", " quick", " brown", " fox"]
    interceptor = TokenStreamInterceptor()
    session = None
    async for _, sess in interceptor.intercept(
        _fake_stream(chunks), "p", "m", "p"
    ):
        session = sess
    assert session is not None
    assert session.full_text == "The quick brown fox"
    assert session.total_tokens == 4


@pytest.mark.asyncio
async def test_session_avg_latency_computed():
    chunks = ["a", "b"]
    interceptor = TokenStreamInterceptor()
    session = None
    async for _, sess in interceptor.intercept(
        _fake_stream(chunks), "p", "m", "p"
    ):
        session = sess
    assert session is not None
    assert 0.0 <= session.avg_latency_signal <= 1.0


@pytest.mark.asyncio
async def test_empty_stream_produces_no_events():
    async def empty():
        return
        yield  # make it an async generator

    interceptor = TokenStreamInterceptor()
    events = []
    async for event, _ in interceptor.intercept(empty(), "p", "m", "p"):
        events.append(event)
    assert events == []
