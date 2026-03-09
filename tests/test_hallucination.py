"""Tests for the hallucination scorer."""

from __future__ import annotations

import pytest
from observatory.core.stream import StreamSession, TokenEvent
from observatory.core.hallucination import HallucinationScorer, FlagType


def _make_session(tokens: list[str], latencies: list[float] | None = None) -> StreamSession:
    session = StreamSession(prompt="test", model="test-model", provider="test")
    for i, text in enumerate(tokens):
        lat = latencies[i] if latencies else 0.1
        event = TokenEvent(
            index=i,
            text=text,
            timestamp_ms=float(i),
            latency_signal=lat,
            cumulative_text="".join(tokens[: i + 1]),
        )
        session.tokens.append(event)
    return session


def test_clean_session_has_low_score():
    session = _make_session(["Hello", " world", "."])
    scorer = HallucinationScorer()
    report = scorer.score(session)
    assert report.is_clean
    assert report.risk_level == "low"


def test_hedge_phrase_detected():
    tokens = ["I", " think", " it", " is", " correct"]
    session = _make_session(tokens)
    scorer = HallucinationScorer()
    report = scorer.score(session)
    hedge_flags = [f for f in report.flags if f.flag_type == FlagType.HEDGE_PHRASE]
    assert len(hedge_flags) >= 1


def test_numeric_claim_flagged():
    tokens = ["The", " year", " 1984", " was", " notable"]
    session = _make_session(tokens)
    scorer = HallucinationScorer()
    report = scorer.score(session)
    numeric_flags = [f for f in report.flags if f.flag_type == FlagType.NUMERIC_CLAIM]
    assert len(numeric_flags) >= 1


def test_high_latency_token_flagged():
    tokens = ["slow", " response", " token"]
    latencies = [0.9, 0.1, 0.1]
    session = _make_session(tokens, latencies)
    scorer = HallucinationScorer(latency_threshold=0.7)
    report = scorer.score(session)
    latency_flags = [f for f in report.flags if f.flag_type == FlagType.HIGH_LATENCY]
    assert len(latency_flags) >= 1
    assert latency_flags[0].token_index == 0


def test_entity_claim_flagged():
    tokens = ["Barack", " Obama", " was", " president"]
    session = _make_session(tokens)
    scorer = HallucinationScorer()
    report = scorer.score(session)
    entity_flags = [f for f in report.flags if f.flag_type == FlagType.ENTITY_CLAIM]
    assert len(entity_flags) >= 1


def test_overall_score_bounded():
    tokens = ["I", " think", " probably", " 1984", " Barack", " Obama"]
    session = _make_session(tokens, [0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    scorer = HallucinationScorer()
    report = scorer.score(session)
    assert 0.0 <= report.overall_score <= 1.0


def test_report_session_id_contains_model():
    session = _make_session(["hi"])
    scorer = HallucinationScorer()
    report = scorer.score(session)
    assert "test-model" in report.session_id


def test_flagged_indices_subset_of_all_indices():
    tokens = ["I", " believe", " 2024", " was", " great"]
    session = _make_session(tokens)
    scorer = HallucinationScorer()
    report = scorer.score(session)
    all_indices = {t.index for t in session.tokens}
    assert report.flagged_token_indices.issubset(all_indices)
