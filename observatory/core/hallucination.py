"""
Hallucination scorer — flags tokens and spans with confidence signals.

Inspired by LLM-Hallucination-Detection-Script
(github.com/Mattbusel/LLM-Hallucination-Detection-Script).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .stream import StreamSession, TokenEvent


class FlagType(str, Enum):
    HIGH_LATENCY = "high_latency"       # Model hesitated — uncertainty signal
    HEDGE_PHRASE = "hedge_phrase"        # "I think", "probably", "might be"
    NUMERIC_CLAIM = "numeric_claim"      # Numbers/dates that could be hallucinated
    ENTITY_CLAIM = "entity_claim"        # Proper nouns — high hallucination risk
    CONTRADICTION = "contradiction"      # Repeats then contradicts earlier text


_HEDGE_PATTERNS = re.compile(
    r"\b(i think|i believe|probably|likely|might|may|perhaps|"
    r"approximately|around|roughly|i'm not sure|i'm not certain|"
    r"it seems|it appears|supposedly|allegedly)\b",
    re.IGNORECASE,
)

_NUMERIC_PATTERN = re.compile(r"\b\d{4}\b|\b\d+[%$]\b|\b\d+\.\d+\b")
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


@dataclass
class TokenFlag:
    token_index: int
    flag_type: FlagType
    confidence: float  # 0.0 (clean) → 1.0 (almost certainly hallucinated)
    explanation: str


@dataclass
class HallucinationReport:
    session_id: str
    overall_score: float  # weighted average confidence across all flags
    flags: list[TokenFlag] = field(default_factory=list)
    flagged_token_indices: set[int] = field(default_factory=set)

    @property
    def is_clean(self) -> bool:
        return self.overall_score < 0.25

    @property
    def risk_level(self) -> str:
        if self.overall_score < 0.25:
            return "low"
        if self.overall_score < 0.55:
            return "medium"
        return "high"


class HallucinationScorer:
    """
    Analyzes a completed StreamSession and produces a HallucinationReport.

    Scoring is heuristic-based (latency signal + linguistic patterns).
    For production use, augment with a dedicated fact-checking LLM call.
    """

    def __init__(
        self,
        latency_threshold: float = 0.7,
        numeric_confidence: float = 0.45,
        entity_confidence: float = 0.35,
        hedge_confidence: float = 0.30,
    ) -> None:
        self._latency_threshold = latency_threshold
        self._numeric_confidence = numeric_confidence
        self._entity_confidence = entity_confidence
        self._hedge_confidence = hedge_confidence

    def score(self, session: StreamSession) -> HallucinationReport:
        flags: list[TokenFlag] = []

        # Pass 1: per-token latency flags
        for token in session.tokens:
            if token.latency_signal >= self._latency_threshold:
                flags.append(TokenFlag(
                    token_index=token.index,
                    flag_type=FlagType.HIGH_LATENCY,
                    confidence=token.latency_signal,
                    explanation=f"High generation latency (signal={token.latency_signal:.2f})",
                ))

        # Pass 2: full-text linguistic flags — map back to token spans
        full_text = session.full_text
        token_starts = self._build_token_offsets(session.tokens)

        for match in _HEDGE_PATTERNS.finditer(full_text):
            idx = self._offset_to_token(match.start(), token_starts)
            flags.append(TokenFlag(
                token_index=idx,
                flag_type=FlagType.HEDGE_PHRASE,
                confidence=self._hedge_confidence,
                explanation=f"Hedge phrase: '{match.group()}'",
            ))

        for match in _NUMERIC_PATTERN.finditer(full_text):
            idx = self._offset_to_token(match.start(), token_starts)
            flags.append(TokenFlag(
                token_index=idx,
                flag_type=FlagType.NUMERIC_CLAIM,
                confidence=self._numeric_confidence,
                explanation=f"Numeric claim: '{match.group()}'",
            ))

        for match in _ENTITY_PATTERN.finditer(full_text):
            idx = self._offset_to_token(match.start(), token_starts)
            flags.append(TokenFlag(
                token_index=idx,
                flag_type=FlagType.ENTITY_CLAIM,
                confidence=self._entity_confidence,
                explanation=f"Entity claim: '{match.group()}'",
            ))

        flagged_indices = {f.token_index for f in flags}
        overall = (
            sum(f.confidence for f in flags) / max(len(session.tokens), 1)
            if flags else 0.0
        )

        return HallucinationReport(
            session_id=f"{session.provider}/{session.model}@{session.started_at:.0f}",
            overall_score=min(overall, 1.0),
            flags=flags,
            flagged_token_indices=flagged_indices,
        )

    @staticmethod
    def _build_token_offsets(tokens: list[TokenEvent]) -> list[int]:
        offsets: list[int] = []
        pos = 0
        for t in tokens:
            offsets.append(pos)
            pos += len(t.text)
        return offsets

    @staticmethod
    def _offset_to_token(char_offset: int, token_starts: list[int]) -> int:
        for i in range(len(token_starts) - 1, -1, -1):
            if token_starts[i] <= char_offset:
                return i
        return 0
