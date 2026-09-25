# Changelog

## [0.3.0] - 2026-09-25

- Works with no API key: every model in a running Ollama shows up as `ollama/<name>` and streams locally for free.
- Redesigned page: live prompt cost that updates as you type (no API call), the reply streams in as it arrives, a strip chart of the gap before every chunk, time to first chunk and median gap, and a "Second look" list that quotes each flagged number, date and name. Empty, waiting and error states explain what to do next; dark mode and phone widths are handled.
- Timing fix: chunks are now timestamped the moment they arrive. Before, drawing the page could delay reading the stream and squash the measured gaps.
- A chunk now counts as a pause only if the model waited at least 120 ms and 4x the run's median gap, so a fast, even stream is no longer flagged almost everywhere.
- Helpful errors for a missing key, a rejected key, Ollama not running and unknown models, instead of a stack trace.
- Prices updated (Claude Opus 4.6 $5/$25, Haiku 4.5 $1/$5 per 1M tokens) and Claude Opus 5 and Sonnet 5 added.
- `prompt-observatory` command name for pipx installs (same as the prebuilt app; `observatory` still works). `--help` shows examples; startup says which keys and local models it found.

## [0.2.0] - 2026-09-25

- Prebuilt single-file executables for Windows, macOS (Apple Silicon and Intel) and Linux on every GitHub Release. Double-click it (or run it from a terminal): it starts the dashboard, prints the local URL and opens your browser. No Python needed.
- `--open` / `--no-open` and `--version`. The app prints the URL it is serving before Gradio starts.
- Works with Gradio 6: theme and CSS are passed where each Gradio version expects them, so the startup warning is gone.
- Fixed: the "Show Export JSON" button called its handler with the wrong number of arguments.
- Fixed: prompts containing special-token text such as `<|endoftext|>` crashed the cost panel.
- CI now fails when tests fail (it used to swallow every error) and runs on Linux and Windows. Stray `__pycache__` files removed from the repository.

## [0.1.0]

- Initial release: token stream latency heatmap, heuristic hallucination flags, prompt cost panel, JSON export.
