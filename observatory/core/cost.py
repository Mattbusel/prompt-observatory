"""
Prompt cost analyzer — token counts, API cost estimates, compression suggestions.

Inspired by Token-Visualizer (github.com/Mattbusel/Token-Visualizer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Approximate costs per 1M tokens (input / output) as of early 2026.
# Update these as pricing changes.
_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input_per_1m, output_per_1m)
    "claude-opus-4-6":    (15.00, 75.00),
    "claude-sonnet-4-6":  (3.00,  15.00),
    "claude-haiku-4-5":   (0.25,   1.25),
    "gpt-4o":             (5.00,  15.00),
    "gpt-4o-mini":        (0.15,   0.60),
    "gpt-4-turbo":        (10.00, 30.00),
}

_VERBOSE_PHRASES = [
    (r"\bin order to\b", "to"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bit is important to note that\b", "note:"),
    (r"\bplease be advised that\b", ""),
    (r"\bin the event that\b", "if"),
    (r"\bwith regard to\b", "regarding"),
]


@dataclass
class LineStats:
    text: str
    token_count: int
    is_high_density: bool  # more tokens than average


@dataclass
class CompressionSuggestion:
    pattern: str
    replacement: str
    estimated_savings: int  # tokens


@dataclass
class CostReport:
    prompt: str
    model: str
    input_tokens: int
    estimated_output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    efficiency_ratio: float  # chars per token (higher = denser)
    line_stats: list[LineStats] = field(default_factory=list)
    suggestions: list[CompressionSuggestion] = field(default_factory=list)

    @property
    def potential_savings_tokens(self) -> int:
        return sum(s.estimated_savings for s in self.suggestions)

    @property
    def potential_savings_usd(self) -> float:
        model_pricing = _PRICING.get(self.model, (5.0, 15.0))
        return self.potential_savings_tokens * model_pricing[0] / 1_000_000


class PromptCostAnalyzer:
    """
    Analyzes a prompt string and returns cost estimates and compression hints.

    Uses tiktoken when available, falls back to word-split approximation.
    """

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        self._tokenizer = self._load_tokenizer(model)

    @staticmethod
    def _load_tokenizer(model: str):  # type: ignore[return]
        try:
            import tiktoken  # type: ignore[import]
            # Map Claude models to cl100k (closest approximation)
            enc_name = "cl100k_base" if model.startswith("claude") else model
            try:
                return tiktoken.encoding_for_model(enc_name)
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))
        # Fallback: ~4 chars per token heuristic
        return max(1, len(text) // 4)

    def analyze(
        self,
        prompt: str,
        estimated_output_tokens: int = 512,
    ) -> CostReport:
        input_tokens = self.count_tokens(prompt)
        pricing = _PRICING.get(self.model, (5.0, 15.0))
        input_cost = input_tokens * pricing[0] / 1_000_000
        output_cost = estimated_output_tokens * pricing[1] / 1_000_000

        lines = prompt.split("\n")
        token_per_line = [self.count_tokens(ln) for ln in lines]
        avg = sum(token_per_line) / max(len(token_per_line), 1)
        line_stats = [
            LineStats(text=ln, token_count=tc, is_high_density=tc > avg * 1.5)
            for ln, tc in zip(lines, token_per_line)
        ]

        suggestions = self._suggest_compressions(prompt)
        efficiency = len(prompt) / max(input_tokens, 1)

        return CostReport(
            prompt=prompt,
            model=self.model,
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            efficiency_ratio=efficiency,
            line_stats=line_stats,
            suggestions=suggestions,
        )

    def _suggest_compressions(self, text: str) -> list[CompressionSuggestion]:
        suggestions: list[CompressionSuggestion] = []
        for pattern, replacement in _VERBOSE_PHRASES:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                original_tokens = sum(self.count_tokens(m) for m in matches)
                replacement_tokens = self.count_tokens(replacement) * len(matches)
                savings = max(0, original_tokens - replacement_tokens)
                suggestions.append(CompressionSuggestion(
                    pattern=pattern,
                    replacement=replacement,
                    estimated_savings=savings,
                ))

        # Flag repeated content
        words = re.findall(r"\b\w{5,}\b", text.lower())
        from collections import Counter
        repeats = {w: c for w, c in Counter(words).items() if c >= 3}
        for word, count in sorted(repeats.items(), key=lambda x: -x[1])[:3]:
            suggestions.append(CompressionSuggestion(
                pattern=f"'{word}' repeated {count}x",
                replacement="consider a pronoun or restructure",
                estimated_savings=max(0, (count - 1) * self.count_tokens(word)),
            ))

        return suggestions
