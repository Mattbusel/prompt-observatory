# prompt-observatory

[![CI](https://github.com/Mattbusel/prompt-observatory/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/prompt-observatory/actions/workflows/ci.yml)

A local Gradio dashboard that streams a response from Claude or GPT, colors each chunk by how long it took to arrive, flags risky spans (hedges, numbers, named entities), and prices the prompt with compression suggestions, all on one screen.

When you are tuning a prompt you usually juggle three questions: what did the model say, which parts of it deserve a second look, and what did it cost. This puts all three next to each other for a single run, and exports the whole session as JSON.

## What it does

- **Token stream panel.** Streams the response through the Anthropic or OpenAI SDK and records the inter-arrival time of every chunk. Latency is normalized over a rolling 5-chunk window and drawn as a background heatmap. Treat it as a rough "the model paused here" signal, not true perplexity: the APIs used here do not expose logprobs.
- **Hallucination panel.** A heuristic scorer (`core/hallucination.py`) flags high-latency chunks, hedge phrases (`probably`, `I think`, `approximately`, ...), numeric claims (years, percentages, decimals) and multi-word proper nouns, maps each back to its token, and reports an overall score and a low / medium / high risk level.
- **Cost panel.** Counts prompt tokens with `tiktoken` (Claude models are approximated with `cl100k_base`), estimates input and output cost from a built-in price table, and suggests shorter phrasings (`in order to` to `to`, `due to the fact that` to `because`, ...).
- **Export.** The session, flags and cost report as JSON in the UI. `ui/export.py` also has a `to_html` report builder you can call from code.

Models in the dropdown: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5`, `gpt-4o`, `gpt-4o-mini`. Names starting with `claude` go to Anthropic, `gpt`, `o1` and `o3` go to OpenAI.

## Quick start

Python 3.10+. Not published on PyPI; install from source.

```bash
git clone https://github.com/Mattbusel/prompt-observatory
cd prompt-observatory
pip install -e ".[dev]"

export ANTHROPIC_API_KEY=sk-ant-...     # and/or OPENAI_API_KEY
observatory                             # or: python -m observatory
```

Open http://127.0.0.1:7860. You can also paste a key into the API key field in the UI.

Options: `--port 7860`, `--host 127.0.0.1`, `--share` (Gradio public link), `--api-key KEY` (sets `ANTHROPIC_API_KEY` if unset).

Run the tests (no API key needed):

```bash
pytest
```

## Architecture

```
observatory/
  app.py                 Gradio UI, CLI entry point, analysis pipeline
  __main__.py            python -m observatory
  core/
    stream.py            TokenStreamInterceptor: timestamps chunks, latency signal
    hallucination.py     HallucinationScorer: latency + regex heuristics
    cost.py              PromptCostAnalyzer: tiktoken counts, pricing, suggestions
  providers/
    anthropic.py         async text stream via anthropic.AsyncAnthropic
    openai.py            async text stream via openai.AsyncOpenAI
  ui/
    export.py            to_json / to_html session reports
tests/                   32 unit tests for stream, scorer, cost and export
```

Flow: prompt goes to the cost analyzer (no API call), then to the provider stream, which the interceptor wraps; the finished session goes to the scorer; all three reports go to the UI and the exporter.

## Limitations

- It is a heuristic dashboard, not a fact checker. A flag means "look here", not "this is wrong". Capitalized phrases and any four-digit number get flagged.
- The latency signal includes network jitter and server batching, so it is noisy.
- Prices in `core/cost.py` are hardcoded and need updating as providers change them.
- One prompt per run; there is no side-by-side comparison mode.

## Related projects

The three core modules started from ideas in these repos by the same author:

- [Every-Other-Token](https://github.com/Mattbusel/Every-Other-Token): token stream interception
- [LLM-Hallucination-Detection-Script](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script): per-token confidence visualizer
- [Token-Visualizer](https://github.com/Mattbusel/Token-Visualizer): prompt token and compression analysis
