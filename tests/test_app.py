"""Dashboard rendering, the no-key paths and the pause heuristic (no network needed)."""

from __future__ import annotations

from observatory import app
from observatory.core.cost import PromptCostAnalyzer, price_for
from observatory.core.hallucination import FlagType, HallucinationScorer
from observatory.core.stream import StreamSession, TokenEvent
from observatory.providers import ollama


def _session(gaps_ms: list[float], texts: list[str] | None = None) -> StreamSession:
    texts = texts or [f"w{i} " for i in range(len(gaps_ms))]
    s = StreamSession(prompt="p", model="ollama/test", provider="ollama")
    t = 0.0
    for i, (g, txt) in enumerate(zip(gaps_ms, texts)):
        t += g
        s.tokens.append(TokenEvent(index=i, text=txt, timestamp_ms=t, latency_signal=1.0,
                                   latency_ms=g))
    return s


def test_even_fast_stream_has_no_pause_flags():
    s = _session([300] + [3] * 40)
    report = HallucinationScorer().score(s)
    assert not [f for f in report.flags if f.flag_type == FlagType.HIGH_LATENCY]


def test_real_pause_is_flagged_with_its_duration():
    s = _session([300] + [20] * 20 + [600] + [20] * 5)
    report = HallucinationScorer().score(s)
    pauses = [f for f in report.flags if f.flag_type == FlagType.HIGH_LATENCY]
    assert [f.token_index for f in pauses] == [21]
    assert "600 ms" in pauses[0].explanation


def test_pause_heat_ignores_first_chunk_and_scales_to_real_pauses():
    heat = app.pause_heat(_session([900, 10, 10, 240]))
    assert heat[0] == 0.0
    assert heat[1] < 0.1
    assert heat[3] == 1.0


def test_stream_html_escapes_and_reports_stats():
    s = _session([250, 5, 5], ["<b>", "Apollo 11 ", "1969"])
    report = HallucinationScorer().score(s)
    out = app.stream_html(s, report, done=True)
    assert "&lt;b&gt;" in out and "<b>\"" not in out
    assert "first chunk" in out and "250 ms" in out
    assert "<svg" in out


def test_risk_html_lists_numbers_and_entities():
    s = _session([250, 5, 5, 5], ["Neil ", "Armstrong ", "landed in ", "1969."])
    report = HallucinationScorer().score(s)
    out = app.risk_html(report, s)
    assert "1969" in out and "Neil Armstrong" in out
    assert "number or date" in out


def test_cost_panel_empty_and_local():
    assert "Start typing" in app.live_cost("", "claude-opus-5", 512)
    local = app.live_cost("in order to test", "ollama/qwen2.5:0.5b", 512)
    assert "$0 (local)" in local
    assert "Shorter phrasings" in local
    assert price_for("ollama/anything") == (0.0, 0.0)


def test_cost_prices_are_current():
    assert price_for("claude-opus-4-6") == (5.0, 25.0)
    assert price_for("claude-haiku-4-5") == (1.0, 5.0)
    r = PromptCostAnalyzer(model="claude-sonnet-4-6").analyze("x" * 40, estimated_output_tokens=1000)
    assert abs(r.output_cost_usd - 0.015) < 1e-9


async def _collect(*args):  # type: ignore[no-untyped-def]
    return [out async for out in app.observe(*args)]


async def test_observe_empty_prompt_explains():
    outs = await _collect("   ", "claude-opus-5", "", 64)
    assert "Write a prompt first" in outs[-1][0]


async def test_observe_missing_key_suggests_ollama(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    outs = await _collect("hi", "claude-opus-5", "", 64)
    assert "No Anthropic API key" in outs[-1][0]
    assert "ollama pull" in outs[-1][0]


async def test_observe_ollama_down_is_friendly(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    outs = await _collect("hi", "ollama/none", "", 64)
    assert "Could not" in outs[-1][0]


def test_ollama_list_models_when_not_running(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    assert ollama.list_models(timeout=0.3) == []
    assert ollama.base_url() == "http://127.0.0.1:9"


def test_provider_detection():
    assert app._detect_provider("ollama/llama3.2") == "ollama"
    assert app._detect_provider("gpt-4o") == "openai"
    assert app._detect_provider("claude-opus-5") == "anthropic"
