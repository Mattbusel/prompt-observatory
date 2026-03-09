# prompt-observatory 🔭

> A unified visual dashboard for LLM interpretability — real-time token streams, hallucination scoring, and prompt cost analysis in one interface.

Combines the power of three tools:
- **[Every-Other-Token](https://github.com/Mattbusel/Every-Other-Token)** — token-level stream interception & perplexity
- **[LLM-Hallucination-Detection-Script](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script)** — confidence scoring & flag detection
- **[Token-Visualizer](https://github.com/Mattbusel/Token-Visualizer)** — prompt cost & compression analysis

## Features

- **Live Token Stream Panel** — watch tokens arrive with perplexity heatmap coloring
- **Hallucination Score Panel** — real-time confidence levels and flagged tokens
- **Prompt Cost Panel** — token count, cost estimate, compression suggestions
- **Side-by-side comparison** — run two prompts simultaneously and diff the results
- **Export** — save sessions as JSON or HTML reports

## Quick Start

```bash
pip install prompt-observatory
observatory --api-key $ANTHROPIC_API_KEY
```

Or run from source:

```bash
git clone https://github.com/Mattbusel/prompt-observatory
cd prompt-observatory
pip install -e ".[dev]"
python -m observatory
```

Open `http://localhost:7860` in your browser.

## Architecture

```
prompt-observatory/
├── observatory/
│   ├── __init__.py
│   ├── __main__.py
│   ├── app.py              # Gradio UI entrypoint
│   ├── core/
│   │   ├── stream.py       # Token stream interceptor
│   │   ├── hallucination.py # Hallucination scorer
│   │   └── cost.py         # Prompt cost analyzer
│   ├── ui/
│   │   ├── panels.py       # Gradio panel components
│   │   └── export.py       # Session export
│   └── providers/
│       ├── anthropic.py
│       └── openai.py
└── tests/
```

## Related Projects by @Mattbusel

- [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) — production-grade Rust orchestration for LLM pipelines
- [Every-Other-Token](https://github.com/Mattbusel/Every-Other-Token) — token stream interceptor (core component)
- [llm-cpp](https://github.com/Mattbusel/llm-cpp) — C++ single-header LLM infrastructure libraries

## License

MIT
