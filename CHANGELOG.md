# Changelog

## [0.2.0] - 2026-09-25

- Prebuilt single-file executables for Windows, macOS (Apple Silicon and Intel) and Linux on every GitHub Release. Double-click it (or run it from a terminal): it starts the dashboard, prints the local URL and opens your browser. No Python needed.
- `--open` / `--no-open` and `--version`. The app prints the URL it is serving before Gradio starts.
- Works with Gradio 6: theme and CSS are passed where each Gradio version expects them, so the startup warning is gone.
- Fixed: the "Show Export JSON" button called its handler with the wrong number of arguments.
- Fixed: prompts containing special-token text such as `<|endoftext|>` crashed the cost panel.
- CI now fails when tests fail (it used to swallow every error) and runs on Linux and Windows. Stray `__pycache__` files removed from the repository.

## [0.1.0]

- Initial release: token stream latency heatmap, heuristic hallucination flags, prompt cost panel, JSON export.
