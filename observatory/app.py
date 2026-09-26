"""
Prompt Observatory: Gradio UI entry point.

A local web page that streams a response from Claude, GPT or a local Ollama
model, shades every chunk by how long it took to arrive, flags spans worth a
second look, and prices the prompt before you send it.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import statistics
import sys
from collections import Counter
from typing import Any, AsyncIterator

from . import __version__
from .core.cost import CostReport, PromptCostAnalyzer, price_for
from .core.hallucination import FlagType, HallucinationReport, HallucinationScorer
from .core.stream import StreamSession, TokenStreamInterceptor
from .providers import ollama as _ollama
from .ui.export import to_json

CLOUD_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5",
    "gpt-4o",
    "gpt-4o-mini",
]

EXAMPLE_PROMPT = (
    "When was the Hubble Space Telescope launched, and what are three things it discovered? "
    "Please be advised that I need this in order to write a short summary, "
    "due to the fact that my notes are incomplete."
)

FLAG_LABELS = {
    FlagType.HIGH_LATENCY: "slow chunk",
    FlagType.HEDGE_PHRASE: "hedge",
    FlagType.NUMERIC_CLAIM: "number or date",
    FlagType.ENTITY_CLAIM: "named entity",
    FlagType.CONTRADICTION: "contradiction",
}


# --------------------------------------------------------------------------- providers

def _detect_provider(model: str) -> str:
    if model.startswith(_ollama.PREFIX):
        return "ollama"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "anthropic"


def _stream_fn(provider: str):  # type: ignore[no-untyped-def]
    if provider == "ollama":
        return _ollama.stream_tokens
    if provider == "openai":
        from .providers.openai import stream_tokens
        return stream_tokens
    from .providers.anthropic import stream_tokens as anthropic_stream
    return anthropic_stream


def _missing_key_message(provider: str, api_key: str) -> str | None:
    if provider == "ollama" or api_key.strip():
        return None
    env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if os.environ.get(env):
        return None
    who = "Anthropic" if provider == "anthropic" else "OpenAI"
    return (
        f"No {who} API key. Paste one into the API key field, or set {env} before starting "
        "the app. No key? Pick a local model instead: install Ollama, run "
        "<code>ollama pull qwen2.5:0.5b</code>, restart the dashboard and choose "
        "<code>ollama/qwen2.5:0.5b</code>."
    )


def _friendly_error(provider: str, model: str, exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc)
    if provider == "ollama" and ("Connect" in name or "connect" in text.lower()):
        return (f"Could not reach Ollama at {html.escape(_ollama.base_url())}. Start it "
                "(<code>ollama serve</code>, or open the Ollama app) and try again.")
    if provider == "ollama" and ("not found" in text.lower() or "404" in text):
        short = html.escape(model[len(_ollama.PREFIX):])
        return f"Ollama does not have {short}. Run <code>ollama pull {short}</code> first."
    if "Authentication" in name or "401" in text:
        return "The API key was rejected. Check that it is correct and belongs to this provider."
    if "NotFound" in name or "404" in text:
        return f"The provider does not know the model {html.escape(model)}."
    if "RateLimit" in name or "429" in text:
        return "Rate limited by the provider. Wait a moment and run it again."
    if "Connect" in name or "Timeout" in name:
        return "Could not reach the provider. Check your internet connection."
    return f"{html.escape(name)}: {html.escape(text[:400])}"


def available_models() -> tuple[list[str], str]:
    """The dropdown choices and a sensible default for this machine."""
    local = [_ollama.PREFIX + m for m in _ollama.list_models()]
    choices = CLOUD_MODELS + local
    if os.environ.get("ANTHROPIC_API_KEY"):
        default = "claude-opus-5"
    elif os.environ.get("OPENAI_API_KEY"):
        default = "gpt-4o-mini"
    elif local:
        default = local[0]
    else:
        default = "claude-opus-5"
    return choices, default


# --------------------------------------------------------------------------- rendering

def _e(s: str) -> str:
    return html.escape(s, quote=True)


def _panel(title: str, body: str, meta: str = "", extra_class: str = "") -> str:
    meta_html = f'<span class="po-meta">{meta}</span>' if meta else ""
    return (f'<section class="po po-panel {extra_class}"><header class="po-head">'
            f'<h3>{title}</h3>{meta_html}</header>{body}</section>')


def empty_stream_html() -> str:
    body = ('<div class="po-empty"><div class="po-empty-mark" aria-hidden="true">'
            + "".join(f'<i style="height:{h}px"></i>' for h in (6, 14, 9, 22, 7, 12, 30, 8, 11, 6))
            + '</div><p><b>Nothing observed yet.</b> Write a prompt and press <b>Observe</b>. '
            'Each chunk of the reply lands here the moment it arrives, shaded by how long '
            'the model paused before sending it.</p></div>')
    return _panel("Stream", body)


def empty_risk_html() -> str:
    return _panel("Second look", '<p class="po-muted">After a run, the spans worth checking '
                  "(numbers, dates, names, hedges, slow chunks) are listed here.</p>")


def waiting_html(model: str) -> str:
    body = (f'<div class="po-wait"><span class="po-pulse"></span>Waiting for the first chunk '
            f'from <code>{_e(model)}</code> ...</div>')
    return _panel("Stream", body, meta="connecting")


def error_html(message: str) -> str:
    return _panel("Stream", f'<div class="po-error" role="alert"><b>Could not run.</b> '
                  f"{message}</div>")


def stream_html(session: StreamSession, report: HallucinationReport | None, done: bool) -> str:
    flagged = report.flagged_token_indices if report else set()
    heat = pause_heat(session)
    spans = []
    for t, h in zip(session.tokens, heat):
        cls = "tk fl" if t.index in flagged else "tk"
        title = f"chunk {t.index} &#183; {t.latency_ms:.0f} ms"
        spans.append(f'<span class="{cls}" style="--h:{h:.2f}" '
                     f'title="{title}">{_e(t.text)}</span>')
    cursor = "" if done else '<span class="po-caret" aria-hidden="true"></span>'
    plate = f'<div class="po-plate" aria-live="polite">{"".join(spans)}{cursor}</div>'

    lat = [t.latency_ms for t in session.tokens]
    stats = ""
    chart = ""
    if lat:
        ttft = lat[0]
        gaps = lat[1:] or [0.0]
        total_s = (session.tokens[-1].timestamp_ms - session.tokens[0].timestamp_ms + ttft) / 1000
        items = [
            ("chunks", f"{len(lat)}"),
            ("first chunk", f"{ttft:,.0f} ms"),
            ("median gap", f"{statistics.median(gaps):,.0f} ms"),
            ("slowest gap", f"{max(gaps):,.0f} ms"),
            ("elapsed", f"{total_s:,.1f} s"),
        ]
        stats = '<dl class="po-stats">' + "".join(
            f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in items) + "</dl>"
        chart = _strip_chart(session, flagged)
    legend = ('<div class="po-legend"><span><i class="sw sw-h0"></i>quick</span>'
              '<span><i class="sw sw-h1"></i>model paused</span>'
              '<span><i class="sw sw-fl"></i>flagged for a second look</span></div>')
    meta = f"{_e(session.model)} &#183; {'done' if done else 'streaming'}"
    return _panel("Stream", plate + chart + stats + legend, meta=meta,
                  extra_class="" if done else "po-live")


def pause_heat(session: StreamSession) -> list[float]:
    """0..1 shade per chunk: the gap before it relative to a real pause.

    A pause means at least 120 ms and 4x the run's median gap. The first chunk
    (time to first token) is shown in the stats instead, so it is not shaded.
    """
    gaps = [t.latency_ms for t in session.tokens[1:]]
    if not gaps:
        return [0.0] * len(session.tokens)
    median = sorted(gaps)[len(gaps) // 2]
    scale = max(120.0, 4 * median)
    return [0.0] + [min(g / scale, 1.0) for g in gaps]


def _strip_chart(session: StreamSession, flagged: set[int]) -> str:
    """Per-chunk gap as a strip chart: one bar per chunk, height = ms since the last one."""
    gaps = [t.latency_ms for t in session.tokens[1:]]
    if not gaps:
        return ""
    n = len(gaps)
    width, height = 1000, 64
    ordered = sorted(gaps)
    cap = max(ordered[min(n - 1, int(n * 0.98))], 1.0)
    bw = width / n
    bars = []
    for i, (t, g) in enumerate(zip(session.tokens[1:], gaps)):
        h = max(1.5, min(g / cap, 1.0) * (height - 4))
        cls = "fl" if t.index in flagged else ""
        bars.append(f'<rect class="{cls}" x="{i * bw:.2f}" y="{height - h:.2f}" '
                    f'width="{max(bw * 0.72, 0.8):.2f}" height="{h:.2f}"/>')
    return ('<figure class="po-chart"><figcaption>Gap before each chunk (ms), '
            f'scale 0 to {cap:,.0f}</figcaption>'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
            f'aria-label="Gap before each of {n} chunks">{"".join(bars)}</svg></figure>')


def risk_html(report: HallucinationReport, session: StreamSession) -> str:
    level = report.risk_level
    counts = Counter(f.flag_type for f in report.flags)
    chips = "".join(f'<span class="po-chip">{FLAG_LABELS.get(k, k.value)} <b>{v}</b></span>'
                    for k, v in counts.most_common())
    head = (f'<div class="po-risk po-risk-{level}"><div class="po-risk-level">{level}</div>'
            f'<div><div class="po-risk-score">score {report.overall_score:.2f}</div>'
            f'<div class="po-muted">{len(report.flags)} flags across {session.total_tokens} '
            'chunks. A flag means "check this", not "this is wrong".</div></div></div>')
    rows = []
    seen: set[tuple[str, str]] = set()
    for f in report.flags:
        key = (f.flag_type.value, f.explanation)
        if key in seen:
            continue
        seen.add(key)
        if f.flag_type == FlagType.HIGH_LATENCY:
            detail = f.explanation
        else:
            detail = f.explanation.split(": ", 1)[-1] if ": " in f.explanation else f.explanation
        rows.append(f'<li><span class="po-tag po-tag-{f.flag_type.value}">'
                    f'{FLAG_LABELS.get(f.flag_type, f.flag_type.value)}</span>'
                    f'<span class="po-detail">{_e(detail)}</span>'
                    f'<span class="po-idx">#{f.token_index}</span></li>')
    more = ""
    if len(rows) > 14:
        more = f'<p class="po-muted">and {len(rows) - 14} more in the session JSON.</p>'
        rows = rows[:14]
    body = head + (f'<div class="po-chips">{chips}</div><ul class="po-flags">{"".join(rows)}</ul>'
                   if rows else '<p class="po-muted">Nothing flagged in this reply.</p>') + more
    return _panel("Second look", body)


def cost_html(report: CostReport | None, model: str) -> str:
    if report is None or not report.prompt.strip():
        return _panel("Prompt cost", '<p class="po-muted">Start typing. The token count and '
                      "price update as you write, before anything is sent.</p>",
                      meta="no API call")
    local = model.startswith(_ollama.PREFIX)
    pin, pout = price_for(model)

    def usd(x: float) -> str:
        return "$0 (local)" if local else (f"${x:.4f}" if x >= 0.0001 else f"${x:.6f}")

    approx = model.startswith("claude") or local
    figs = [
        ("input tokens", f"{report.input_tokens:,}" + ("*" if approx else "")),
        ("input cost", usd(report.input_cost_usd)),
        (f"worst case, {report.estimated_output_tokens:,} out", usd(report.total_cost_usd)),
        ("chars per token", f"{report.efficiency_ratio:.1f}"),
    ]
    grid = '<dl class="po-figs">' + "".join(
        f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in figs) + "</dl>"
    sugg = ""
    if report.suggestions:
        items = []
        for s in report.suggestions:
            pat = s.pattern.replace(r"\b", "")
            rep = s.replacement or "(delete)"
            items.append(f'<li><s>{_e(pat)}</s><span class="po-arrow">to</span>'
                         f'<b>{_e(rep)}</b><span class="po-save">-{s.estimated_savings} tok</span></li>')
        sugg = ('<h4>Shorter phrasings</h4><ul class="po-sugg">' + "".join(items) + "</ul>")
    else:
        sugg = '<p class="po-muted">No wordy phrases found.</p>'
    note = ""
    if approx:
        note = ('<p class="po-foot">* counted with the GPT tokenizer (cl100k); this model\'s '
                "own tokenizer may differ a little.</p>")
    price = "free, runs on your machine" if local else f"${pin:g} in / ${pout:g} out per 1M tokens"
    return _panel("Prompt cost", grid + sugg + note, meta=_e(price))


def live_cost(prompt: str, model: str, max_tokens: float) -> str:
    if not (prompt or "").strip():
        return cost_html(None, model)
    return cost_html(PromptCostAnalyzer(model=model).analyze(prompt, int(max_tokens)), model)


# --------------------------------------------------------------------------- pipeline

async def observe(prompt: str, model: str, api_key: str, max_tokens: int
                  ) -> AsyncIterator[tuple[str, str, str]]:
    """Yield (stream_html, risk_html, export_json) as the reply streams in."""
    if not (prompt or "").strip():
        yield error_html("Write a prompt first (or press <b>Load example</b>)."), empty_risk_html(), ""
        return
    provider = _detect_provider(model)
    missing = _missing_key_message(provider, api_key or "")
    if missing:
        yield error_html(missing), empty_risk_html(), ""
        return

    cost_report = PromptCostAnalyzer(model=model).analyze(prompt, estimated_output_tokens=max_tokens)
    yield waiting_html(model), empty_risk_html(), ""

    # The provider stream is read in its own task so chunk timestamps are taken
    # the moment each chunk arrives, not when the page has finished drawing the
    # previous update (which would squash every gap to 0 ms).
    interceptor = TokenStreamInterceptor()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def pump() -> None:
        try:
            raw = _stream_fn(provider)(prompt=prompt, model=model, api_key=api_key or None,
                                       max_tokens=max_tokens)
            async for _event, sess in interceptor.intercept(raw, prompt, model, provider):
                queue.put_nowait(("chunk", sess))
            queue.put_nowait(("done", None))
        except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
            queue.put_nowait(("error", exc))

    task = asyncio.create_task(pump())
    session: StreamSession | None = None
    try:
        while True:
            kind, value = await queue.get()
            if kind == "chunk":
                session = value
            while kind == "chunk" and not queue.empty():
                kind, value = queue.get_nowait()  # skip to the newest state
                if kind == "chunk":
                    session = value
            if kind == "chunk":
                yield stream_html(value, None, done=False), empty_risk_html(), ""
                await asyncio.sleep(0.04)
            elif kind == "error":
                yield error_html(_friendly_error(provider, model, value)), empty_risk_html(), ""
                return
            else:
                break
    finally:
        task.cancel()

    if session is None or not session.tokens:
        yield error_html("The model returned no text. Try again or raise max output tokens."), \
            empty_risk_html(), ""
        return
    report = HallucinationScorer().score(session)
    yield (stream_html(session, report, done=True), risk_html(report, session),
           to_json(session, report, cost_report))


async def _run_analysis(prompt: str, model: str, api_key: str, max_tokens: int
                        ) -> tuple[str, str, str, str]:
    """Run one prompt to completion. Returns (stream_html, risk_html, cost_html, export_json)."""
    last: tuple[str, str, str] = ("", "", "")
    async for out in observe(prompt, model, api_key, max_tokens):
        last = out
    return last[0], last[1], live_cost(prompt, model, max_tokens), last[2]


# --------------------------------------------------------------------------- UI

MASTHEAD = """
<div class="po po-mast">
  <svg class="po-mark" viewBox="0 0 44 44" aria-hidden="true">
    <rect x="3" y="26" width="4" height="12" rx="1"/><rect x="10" y="30" width="4" height="8" rx="1"/>
    <rect x="17" y="12" width="4" height="26" rx="1" class="hot"/><rect x="24" y="28" width="4" height="10" rx="1"/>
    <rect x="31" y="20" width="4" height="18" rx="1"/><rect x="38" y="31" width="4" height="7" rx="1"/>
  </svg>
  <div>
    <h1>Prompt Observatory</h1>
    <p>Watch a model's reply arrive chunk by chunk, see where it paused, check the claims worth
    checking, and know what the prompt costs before you send it.</p>
  </div>
  <span class="po-ver">v__VERSION__ &#183; runs locally</span>
</div>
""".replace("__VERSION__", __version__)

CSS = """
.po{--paper:#f6f3ec;--card:#fffdf8;--ink:#1c2230;--muted:#6b6f78;--rule:#e2dccf;
  --amber:#d9891c;--heat:#f0a830;--flag:#d4442e;--flag-bg:#fbe3dc;--mono:ui-monospace,'Cascadia Mono','SF Mono',Menlo,Consolas,monospace;
  color:var(--ink);font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif}
.dark .po{--paper:#0e1116;--card:#151a21;--ink:#e8e4da;--muted:#9097a1;--rule:#28303a;
  --amber:#f0a830;--heat:#f0a830;--flag:#ff6b52;--flag-bg:#3a1d19}
.po-mast{display:flex;gap:16px;align-items:center;padding:6px 2px 2px;flex-wrap:wrap}
.po-mark{width:44px;height:44px;flex:none;fill:var(--ink);opacity:.85}.po-mark .hot{fill:var(--amber);opacity:1}
.po-mast h1{margin:0;font-size:26px;letter-spacing:-.01em;font-weight:700;color:var(--ink)}
.po-mast p{margin:2px 0 0;color:var(--muted);max-width:720px;font-size:14.5px;line-height:1.45}
.po-mast>div{flex:1;min-width:240px}
.po-ver{font:12px var(--mono);color:var(--muted);border:1px solid var(--rule);border-radius:99px;padding:3px 10px}
.po-panel{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:14px 16px 12px}
.po-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap}
.po-head h3{margin:0;font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.po-meta{font:12px var(--mono);color:var(--muted)}
.po-live .po-meta{color:var(--amber)}
.po-muted{color:var(--muted);font-size:14px;margin:4px 0}
.po-plate{font:15px/1.9 var(--mono);white-space:pre-wrap;word-break:break-word;min-height:84px;
  padding:12px 14px;border-radius:8px;background:var(--paper);border:1px dashed var(--rule)}
.po .tk{background:color-mix(in srgb,var(--heat) calc(var(--h) * 55%),transparent);border-radius:3px;padding:1px 0}
.po .tk.fl{box-shadow:inset 0 -2px 0 var(--flag);background:color-mix(in srgb,var(--flag) calc(12% + var(--h) * 30%),transparent)}
.po-caret{display:inline-block;width:8px;height:1.1em;vertical-align:-3px;background:var(--amber);animation:po-blink 1s steps(2) infinite}
@keyframes po-blink{50%{opacity:0}}
.po-chart{margin:12px 0 0}.po-chart figcaption{font:11px var(--mono);color:var(--muted);margin-bottom:4px}
.po-chart svg{width:100%;height:64px;display:block;border-bottom:1px solid var(--rule)}
.po-chart rect{fill:var(--ink);opacity:.55}.po-chart rect.fl{fill:var(--flag);opacity:1}
.po-stats,.po-figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin:12px 0 4px}
.po-stats div,.po-figs div{border-left:2px solid var(--rule);padding:2px 0 2px 10px}
.po dt{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.po dd{margin:2px 0 0;font:600 18px var(--mono);color:var(--ink)}
.po-legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
.po-legend span{display:inline-flex;gap:6px;align-items:center}
.sw{display:inline-block;width:14px;height:10px;border-radius:2px}
.sw-h0{background:color-mix(in srgb,var(--heat) 8%,transparent);border:1px solid var(--rule)}
.sw-h1{background:color-mix(in srgb,var(--heat) 55%,transparent)}
.sw-fl{background:var(--flag-bg);box-shadow:inset 0 -2px 0 var(--flag)}
.po-empty{display:flex;gap:18px;align-items:center;padding:18px 6px;color:var(--muted);font-size:14.5px;line-height:1.5}
.po-empty p{margin:0;max-width:640px}.po-empty b{color:var(--ink)}
.po-empty-mark{display:flex;align-items:flex-end;gap:4px;height:34px;flex:none}
.po-empty-mark i{width:5px;background:var(--rule);border-radius:1px}
.po-wait{display:flex;align-items:center;gap:10px;padding:22px 6px;color:var(--muted)}
.po-pulse{width:10px;height:10px;border-radius:50%;background:var(--amber);animation:po-pulse 1.1s ease-in-out infinite}
@keyframes po-pulse{50%{transform:scale(.5);opacity:.4}}
.po-error{border-left:3px solid var(--flag);background:var(--flag-bg);padding:12px 14px;border-radius:6px;line-height:1.55}
.po code{font-family:var(--mono);font-size:.92em;background:var(--paper);border:1px solid var(--rule);border-radius:4px;padding:0 4px}
.po-risk{display:flex;gap:14px;align-items:center;margin-bottom:10px}
.po-risk-level{font:700 13px var(--mono);letter-spacing:.14em;text-transform:uppercase;padding:8px 12px;border-radius:6px;border:1.5px solid}
.po-risk-low .po-risk-level{color:#3d7a4a;border-color:#3d7a4a}
.dark .po-risk-low .po-risk-level{color:#7fc48d;border-color:#7fc48d}
.po-risk-medium .po-risk-level{color:var(--amber);border-color:var(--amber)}
.po-risk-high .po-risk-level{color:var(--flag);border-color:var(--flag)}
.po-risk-score{font:600 15px var(--mono)}
.po-chips{display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 10px}
.po-chip{font-size:12px;border:1px solid var(--rule);border-radius:99px;padding:2px 10px;color:var(--muted)}.po-chip b{color:var(--ink)}
.po-flags{list-style:none;margin:0;padding:0}
.po-flags li{display:grid;grid-template-columns:118px 1fr auto;gap:10px;align-items:baseline;padding:6px 0;border-top:1px solid var(--rule);font-size:14px}
.po-tag{font:11px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--flag)}
.po-tag-hedge_phrase,.po-tag-high_latency{color:var(--amber)}
.po-detail{font-family:var(--mono);word-break:break-word}.po-idx{font:11px var(--mono);color:var(--muted)}
.po h4{margin:12px 0 6px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.po-sugg{list-style:none;margin:0;padding:0}
.po-sugg li{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;padding:5px 0;border-top:1px solid var(--rule);font:14px var(--mono)}
.po-sugg s{color:var(--muted)}.po-arrow{font-size:11px;color:var(--muted)}.po-sugg b{color:var(--ink)}
.po-save{margin-left:auto;color:var(--amber);font-size:12px}
.po-foot{font-size:12px;color:var(--muted);margin:10px 0 0}
#po-run{min-height:46px;font-size:16px}
.po-html .html-container,.po-html.html-container{padding:0 !important}
@media (max-width:640px){.po-flags li{grid-template-columns:1fr auto}.po-flags .po-tag{grid-column:1/-1}.po-mast h1{font-size:22px}}
"""


def _theme(gr):  # type: ignore[no-untyped-def]
    sans = ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif"]
    mono = ["ui-monospace", "Cascadia Mono", "SF Mono", "Menlo", "Consolas", "monospace"]
    return gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        font=sans,
        font_mono=mono,
        radius_size=gr.themes.sizes.radius_md,
    ).set(
        body_background_fill="#f6f3ec",
        body_background_fill_dark="#0e1116",
        block_background_fill="#fffdf8",
        block_background_fill_dark="#151a21",
        block_border_color="#e2dccf",
        block_border_color_dark="#28303a",
        input_background_fill="#fffdf8",
        input_background_fill_dark="#10141a",
        button_primary_background_fill="#1c2230",
        button_primary_background_fill_hover="#2c3446",
        button_primary_text_color="#fdf8ee",
        button_primary_background_fill_dark="#f0a830",
        button_primary_background_fill_hover_dark="#f5bb55",
        button_primary_text_color_dark="#16130d",
        slider_color="#d9891c",
        slider_color_dark="#f0a830",
    )


def _style_kwargs(gr) -> dict:  # type: ignore[no-untyped-def,type-arg]
    return {"theme": _theme(gr), "css": CSS}


def _gradio_major(gr) -> int:  # type: ignore[no-untyped-def]
    try:
        return int(str(gr.__version__).split(".")[0])
    except (AttributeError, ValueError):
        return 0


def _use_bundled_encodings() -> None:
    """Point tiktoken at the encodings shipped inside the prebuilt executable.

    tiktoken normally downloads them on first use; the release binaries carry
    them so token counting works offline. A user-set TIKTOKEN_CACHE_DIR wins.
    """
    base = getattr(sys, "_MEIPASS", None)
    if base and "TIKTOKEN_CACHE_DIR" not in os.environ:
        cache = os.path.join(base, "tiktoken_cache")
        if os.path.isdir(cache):
            os.environ["TIKTOKEN_CACHE_DIR"] = cache


def build_ui() -> Any:
    try:
        import gradio as gr  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Run: pip install gradio") from exc

    choices, default_model = available_models()
    # Gradio 6 moved theme and css from Blocks() to launch().
    style = {} if _gradio_major(gr) >= 6 else _style_kwargs(gr)
    with gr.Blocks(title="Prompt Observatory", **style) as demo:
        gr.HTML(MASTHEAD, elem_classes=["po-html"])
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=320):
                prompt_in = gr.Textbox(label="Prompt", lines=6, max_lines=18,
                                       placeholder="Ask the model something. The cost panel "
                                                   "updates as you type.")
                with gr.Row():
                    model_in = gr.Dropdown(label="Model", choices=choices, value=default_model,
                                           allow_custom_value=True, min_width=200,
                                           info="ollama/... models run locally with no key")
                    max_tokens_in = gr.Slider(label="Max output tokens", minimum=64, maximum=4096,
                                              value=512, step=64, min_width=200)
                api_key_in = gr.Textbox(
                    label="API key (optional)", type="password",
                    placeholder="sk-ant-... or sk-...  (or set ANTHROPIC_API_KEY / OPENAI_API_KEY)",
                    info="Used for this session only. Never saved or sent anywhere except the provider.")
                with gr.Row():
                    run_btn = gr.Button("Observe", variant="primary", elem_id="po-run", scale=3)
                    example_btn = gr.Button("Load example", variant="secondary", scale=1)
            with gr.Column(scale=4, min_width=300):
                cost_out = gr.HTML(cost_html(None, default_model), elem_classes=["po-html"])
        stream_out = gr.HTML(empty_stream_html(), elem_classes=["po-html"])
        risk_out = gr.HTML(empty_risk_html(), elem_classes=["po-html"])
        with gr.Accordion("Session JSON (everything above, for your notes or a bug report)",
                          open=False):
            export_out = gr.Code(language="json", value="", show_label=False)

        cost_inputs = [prompt_in, model_in, max_tokens_in]
        for comp in cost_inputs:
            comp.change(live_cost, cost_inputs, cost_out, show_progress="hidden")
        example_btn.click(lambda: EXAMPLE_PROMPT, None, prompt_in)

        async def on_run(prompt, model, api_key, max_tokens):  # type: ignore[no-untyped-def]
            async for out in observe(prompt, model, api_key, int(max_tokens)):
                yield out

        run_btn.click(
            lambda: gr.update(value="Observing ...", interactive=False), None, run_btn,
            show_progress="hidden",
        ).then(
            on_run, [prompt_in, model_in, api_key_in, max_tokens_in],
            [stream_out, risk_out, export_out], show_progress="hidden",
        ).then(
            lambda: gr.update(value="Observe", interactive=True), None, run_btn,
            show_progress="hidden",
        )

    return demo


def _startup_report() -> None:
    local = _ollama.list_models()
    have = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        have.append("Anthropic key found")
    if os.environ.get("OPENAI_API_KEY"):
        have.append("OpenAI key found")
    if local:
        have.append(f"Ollama: {len(local)} local model{'s' if len(local) != 1 else ''}")
    if have:
        print("Ready: " + ", ".join(have) + ".")
    else:
        print("No API key set and no Ollama models found. You can paste a key in the page, "
              "or run a model locally for free: install Ollama, then `ollama pull qwen2.5:0.5b`.")


def main(argv: list[str] | None = None) -> None:
    frozen = bool(getattr(sys, "frozen", False))
    parser = argparse.ArgumentParser(
        prog="prompt-observatory",
        description="Prompt Observatory: a local dashboard that streams a Claude, GPT or local "
                    "Ollama reply chunk by chunk, flags spans worth a second look and prices "
                    "the prompt.",
        epilog="examples:\n"
               "  prompt-observatory                 open the dashboard at http://127.0.0.1:7860\n"
               "  prompt-observatory --port 8000     use another port\n"
               "  ANTHROPIC_API_KEY=sk-ant-... prompt-observatory\n"
               "  ollama pull qwen2.5:0.5b && prompt-observatory   free, no key",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=7860, help="port to serve on (default 7860)")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    parser.add_argument("--api-key", default="", help="sets ANTHROPIC_API_KEY if unset")
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--open", dest="open_browser", action="store_true", default=frozen,
                         help="open the dashboard in your browser"
                              + (" (default for the prebuilt app)" if frozen else ""))
    browser.add_argument("--no-open", dest="open_browser", action="store_false",
                         help="do not open a browser")
    parser.add_argument("--version", action="version", version=f"prompt-observatory {__version__}")
    args = parser.parse_args(argv)

    if args.api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", args.api_key)

    _use_bundled_encodings()
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

    import gradio as gr  # type: ignore[import]

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"Prompt Observatory {__version__}")
    _startup_report()
    print(f"Starting the dashboard at {url}")
    print("Leave this window open while you use it. Press Ctrl+C to stop.", flush=True)

    demo = build_ui()
    launch_kwargs = _style_kwargs(gr) if _gradio_major(gr) >= 6 else {}
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
