"""Tests for the prompt cost analyzer."""

from __future__ import annotations

import pytest
from observatory.core.cost import PromptCostAnalyzer, CostReport


def test_token_count_nonzero_for_nonempty_prompt():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    count = analyzer.count_tokens("Hello world")
    assert count > 0


def test_token_count_scales_with_length():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    short = analyzer.count_tokens("Hi")
    long = analyzer.count_tokens("Hi " * 100)
    assert long > short


def test_cost_report_has_positive_input_cost():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    report = analyzer.analyze("What is the capital of France?")
    assert report.input_cost_usd > 0


def test_cost_report_total_includes_output():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    report = analyzer.analyze("short", estimated_output_tokens=1000)
    assert report.total_cost_usd > report.input_cost_usd


def test_efficiency_ratio_positive():
    analyzer = PromptCostAnalyzer(model="gpt-4o")
    report = analyzer.analyze("The quick brown fox")
    assert report.efficiency_ratio > 0


def test_verbose_phrase_generates_suggestion():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    report = analyzer.analyze("in order to achieve this goal, in order to proceed")
    suggestions = [s for s in report.suggestions if "in order to" in s.pattern]
    assert len(suggestions) >= 1


def test_line_stats_count_matches_lines():
    prompt = "line one\nline two\nline three"
    analyzer = PromptCostAnalyzer(model="claude-haiku-4-5")
    report = analyzer.analyze(prompt)
    assert len(report.line_stats) == 3


def test_potential_savings_non_negative():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    report = analyzer.analyze("probably due to the fact that I think it might be correct")
    assert report.potential_savings_tokens >= 0
    assert report.potential_savings_usd >= 0


def test_unknown_model_uses_fallback_pricing():
    analyzer = PromptCostAnalyzer(model="unknown-model-xyz")
    report = analyzer.analyze("test prompt")
    assert report.total_cost_usd > 0


def test_repeated_word_suggestion():
    analyzer = PromptCostAnalyzer(model="claude-sonnet-4-6")
    prompt = "important important important important content important important"
    report = analyzer.analyze(prompt)
    repeat_suggestions = [s for s in report.suggestions if "repeated" in s.pattern]
    assert len(repeat_suggestions) >= 1
