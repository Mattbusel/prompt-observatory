"""Tests for session export (JSON + HTML)."""

from __future__ import annotations

import json
import pytest
from observatory.core.stream import StreamSession, TokenEvent
from observatory.core.hallucination import HallucinationScorer
from observatory.core.cost import PromptCostAnalyzer
from observatory.ui.export import to_json, to_html


def _build_fixtures():
    session = StreamSession(prompt="What is AI?", model="claude-sonnet-4-6", provider="anthropic")
    for i, text in enumerate(["AI", " is", " transformative", "."]):
        session.tokens.append(TokenEvent(
            index=i, text=text, timestamp_ms=float(i),
            latency_signal=0.2, cumulative_text="AI is transformative."[:i+len(text)],
        ))
    session.finish()

    scorer = HallucinationScorer()
    hallucination = scorer.score(session)

    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    cost = analyzer.analyze("What is AI?")
    return session, hallucination, cost


def test_json_export_is_valid_json():
    session, hallucination, cost = _build_fixtures()
    result = to_json(session, hallucination, cost)
    parsed = json.loads(result)
    assert "session" in parsed
    assert "hallucination" in parsed
    assert "cost" in parsed


def test_json_export_contains_prompt():
    session, hallucination, cost = _build_fixtures()
    result = to_json(session, hallucination, cost)
    parsed = json.loads(result)
    assert parsed["session"]["prompt"] == "What is AI?"


def test_json_export_contains_token_count():
    session, hallucination, cost = _build_fixtures()
    result = to_json(session, hallucination, cost)
    parsed = json.loads(result)
    assert parsed["session"]["total_tokens"] == 4


def test_html_export_is_string():
    session, hallucination, cost = _build_fixtures()
    result = to_html(session, hallucination, cost)
    assert isinstance(result, str)
    assert len(result) > 100


def test_html_export_contains_model_info():
    session, hallucination, cost = _build_fixtures()
    result = to_html(session, hallucination, cost)
    assert "claude-sonnet-4-6" in result


def test_html_export_contains_risk_level():
    session, hallucination, cost = _build_fixtures()
    result = to_html(session, hallucination, cost)
    assert any(level in result.upper() for level in ["LOW", "MEDIUM", "HIGH"])


def test_json_export_has_exported_at():
    session, hallucination, cost = _build_fixtures()
    result = to_json(session, hallucination, cost)
    parsed = json.loads(result)
    assert "exported_at" in parsed
