"""
Prompt Observatory — Gradio UI entrypoint.

Launches a local web interface combining:
  - Live token stream with perplexity heatmap
  - Hallucination confidence scoring
  - Prompt cost analysis and compression suggestions
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Iterator

from .core.cost import PromptCostAnalyzer
from .core.hallucination import HallucinationScorer
from .core.stream import StreamSession, TokenStreamInterceptor
from .ui.export import to_html, to_json


def _detect_provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    return "anthropic"


async def _run_analysis(
    prompt: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> tuple[str, str, str, str]:
    """
    Core analysis pipeline. Returns (stream_html, hallucination_md, cost_md, export_json).
    """
    provider = _detect_provider(model)

    if provider == "anthropic":
        from .providers.anthropic import stream_tokens
    else:
        from .providers.openai import stream_tokens  # type: ignore[no-redef]

    interceptor = TokenStreamInterceptor()
    scorer = HallucinationScorer()
    analyzer = PromptCostAnalyzer(model=model)

    # Cost report on the prompt (synchronous — no API call needed)
    cost_report = analyzer.analyze(prompt, estimated_output_tokens=max_tokens)

    # Stream and intercept
    session: StreamSession | None = None
    raw_stream = stream_tokens(
        prompt=prompt,
        model=model,
        api_key=api_key or None,
        max_tokens=max_tokens,
    )

    async for _event, sess in interceptor.intercept(raw_stream, prompt, model, provider):
        session = sess

    if session is None:
        return "No output received.", "N/A", _cost_md(cost_report), "{}"

    hallucination_report = scorer.score(session)

    from .ui.export import to_html as _html, to_json as _json
    export = _json(session, hallucination_report, cost_report)

    # Build colored token stream (simple HTML)
    stream_html = _build_stream_html(session, hallucination_report)
    halluc_md = _hallucination_md(hallucination_report)
    cost_md = _cost_md(cost_report)

    return stream_html, halluc_md, cost_md, export


def _build_stream_html(session: StreamSession, report) -> str:  # type: ignore[type-arg]
    parts: list[str] = []
    for token in session.tokens:
        text = token.text.replace("&", "&amp;").replace("<", "&lt;")
        flagged = token.index in report.flagged_token_indices
        bg = f"rgba(255,100,50,{token.latency_signal:.2f})" if flagged else \
             f"rgba(255,200,50,{token.latency_signal * 0.5:.2f})"
        parts.append(f'<span style="background:{bg};border-radius:2px">{text}</span>')
    return "".join(parts)


def _hallucination_md(report) -> str:  # type: ignore[type-arg]
    lines = [
        f"**Risk level:** {report.risk_level.upper()}  \n"
        f"**Overall score:** {report.overall_score:.2%}  \n"
        f"**Flags detected:** {len(report.flags)}\n\n"
    ]
    if report.flags:
        lines.append("| Token # | Type | Confidence | Note |\n|---|---|---|---|")
        for f in report.flags[:20]:
            lines.append(f"| {f.token_index} | {f.flag_type.value} | {f.confidence:.2f} | {f.explanation} |")
    else:
        lines.append("✅ No hallucination signals detected.")
    return "\n".join(lines)


def _cost_md(report) -> str:
    lines = [
        f"**Input tokens:** {report.input_tokens}  \n"
        f"**Efficiency:** {report.efficiency_ratio:.1f} chars/token  \n"
        f"**Input cost:** ${report.input_cost_usd:.6f}  \n"
        f"**Total cost:** ${report.total_cost_usd:.6f}  \n"
        f"**Potential savings:** {report.potential_savings_tokens} tokens "
        f"(${report.potential_savings_usd:.6f})\n"
    ]
    if report.suggestions:
        lines.append("\n**Compression suggestions:**")
        for s in report.suggestions:
            lines.append(f"- `{s.pattern}` → `{s.replacement}` (saves ~{s.estimated_savings} tokens)")
    return "\n".join(lines)


def build_ui():
    try:
        import gradio as gr  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("Run: pip install gradio") from exc

    with gr.Blocks(
        title="Prompt Observatory",
        theme=gr.themes.Soft(),
        css=".token-stream { font-family: monospace; line-height: 2; }",
    ) as demo:
        gr.Markdown(
            "# 🔭 Prompt Observatory\n"
            "> Unified LLM interpretability — token streams · hallucination scoring · cost analysis\n\n"
            "Built on [Every-Other-Token](https://github.com/Mattbusel/Every-Other-Token) · "
            "[LLM-Hallucination-Detection](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script) · "
            "[Token-Visualizer](https://github.com/Mattbusel/Token-Visualizer)"
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_in = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=6,
                )
                with gr.Row():
                    model_in = gr.Dropdown(
                        label="Model",
                        choices=[
                            "claude-sonnet-4-6",
                            "claude-opus-4-6",
                            "claude-haiku-4-5",
                            "gpt-4o",
                            "gpt-4o-mini",
                        ],
                        value="claude-sonnet-4-6",
                    )
                    max_tokens_in = gr.Slider(
                        label="Max output tokens",
                        minimum=64,
                        maximum=4096,
                        value=512,
                        step=64,
                    )
                api_key_in = gr.Textbox(
                    label="API Key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
                    type="password",
                    placeholder="sk-ant-... or sk-...",
                )
                run_btn = gr.Button("🔬 Analyze", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Token Stream (latency heatmap)")
                stream_out = gr.HTML(elem_classes=["token-stream"])

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Hallucination Analysis")
                halluc_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### Cost & Compression")
                cost_out = gr.Markdown()

        with gr.Row():
            export_out = gr.Code(label="Export JSON", language="json", visible=False)
            export_btn = gr.Button("📥 Show Export JSON")

        export_btn.click(fn=lambda x: gr.update(visible=True), inputs=[], outputs=[export_out])

        async def on_run(prompt, model, api_key, max_tokens):
            stream_html, halluc_md, cost_md, export_json = await _run_analysis(
                prompt, model, api_key, int(max_tokens)
            )
            return stream_html, halluc_md, cost_md, export_json

        run_btn.click(
            fn=on_run,
            inputs=[prompt_in, model_in, api_key_in, max_tokens_in],
            outputs=[stream_out, halluc_out, cost_out, export_out],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Observatory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    if args.api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", args.api_key)

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
