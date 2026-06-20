<h1 align="center">Research Agent</h1>

<p align="center">
  <em>A command-line research agent that plans, searches, and writes a cited report — no agent framework, no paid APIs.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-28%20passing-brightgreen" alt="Tests: 28 passing">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT">
</p>

---

## Overview

Research Agent is a command-line tool that turns a research topic into a structured, cited Markdown report without any manual searching. Give it a question, and it plans a handful of focused sub-queries, searches the web for each one, fetches and reads the resulting pages concurrently, and asks an LLM to weave everything into a report with inline citations. The plan → gather → synthesize loop is plain Python with no agent framework underneath, so every step is easy to read, test, and explain. It runs entirely on free-tier infrastructure — DuckDuckGo for search, Groq or Gemini for the language model.

---

## What it does

Given a topic, `ResearchAgent.run()` drives three phases end to end:

**Plan.** The topic goes to the LLM with instructions to break it into a handful of distinct sub-queries — background, recent developments, technical depth, comparisons, criticism — so the eventual report has real breadth instead of restating one angle five times. The model is asked to return a plain JSON array of strings. If that call fails, returns something unparseable, or comes back empty, the agent falls back to treating the original topic as the only query, so a planning hiccup never kills the run.

**Gather.** Every sub-query is searched on DuckDuckGo (no API key required), and the resulting URLs are de-duplicated across queries — overlapping sub-queries shouldn't mean fetching the same page twice. All the unique pages are then downloaded *concurrently*, each one parsed down to its `<article>` or `<main>` content (falling back to `<p>` tags when neither exists), and truncated to a few thousand characters so the eventual LLM prompt stays a sane size.

**Synthesize.** Every fetched source is numbered and handed to the LLM in a single prompt, with instructions to write a Markdown report — an overview, several key-finding sections, a limitations section, and a numbered source list — citing sources inline as `[Source N]`. If synthesis itself fails, the agent doesn't return nothing; it falls back to a raw-snippet dump of whatever was gathered, so a run is never a total loss.

---

## Why it's built this way

**Why no agent framework?**
I wrote the plan → gather → synthesize loop by hand instead of reaching for LangChain or a similar framework. With three phases and a handful of functions, a framework would mostly hide control flow I want to be able to point at and explain — what gets called, in what order, with what fallback. The honest trade-off is that this doesn't scale indefinitely: if the pipeline grew into dozens of tool calls, branching logic, or multiple cooperating agents, hand-rolled orchestration would start costing more than it saves. At this size, transparency wins.

**Why concurrent fetching?**
Page downloads are the part of a run that's I/O-bound — a dozen-plus HTTP requests that each spend most of their time waiting on the network, not the CPU. That's the textbook case for `asyncio` over threads, so `gather()` fetches every unique URL through a single `httpx.AsyncClient`, bounded by a semaphore (`Settings.max_concurrent_fetches`, default 8) so the agent never opens more sockets at once than that. In practice this turns what would be a dozen sequential multi-second fetches into one wave, for a measured ~8× speedup over fetching sequentially. The semaphore cap is also a deliberate ceiling — fast, but not so aggressive that it looks like abuse to whatever site it's hitting.

**Why a provider abstraction for the LLM?**
`LLMClient` hides the differences between Groq and Gemini behind two methods, `chat()` and `chat_json()`. The two APIs don't agree on much — Gemini has no system role and wants chat history shaped differently than Groq's OpenAI-style messages — so that adaptation happens once, in one place, instead of leaking into `agent.py`. Switching providers is a single `.env` line (`LLM_PROVIDER=gemini`), not a code change, and adding a third provider costs one `_call_<name>` method plus one dispatch entry. The trade-off is that the abstraction is deliberately thin: it normalizes exactly what these two providers need and nothing more, so a genuinely different provider (multi-modal input, native function-calling) might outgrow it.

**Why centralize configuration in one `Settings` object?**
Every tunable knob in the project — timeouts, retry counts, the concurrency cap, model names — loads from the environment into one frozen dataclass. Frozen means a `Settings` instance can't be mutated mid-run by accident, which is a small but real correctness guarantee once you're juggling async tasks. The alternative — reading `os.getenv()` calls scattered through the modules that need them — is exactly the kind of magic-number sprawl that makes a codebase hard to tune later, and I wanted one place to look.

**Why keep prompts as named constants?**
All of the LLM prompts live in `prompts.py` as plain strings, not f-strings assembled inline inside `plan()` or `synthesize()`. Tuning behavior — the citation format, how many key-finding sections to ask for, the JSON contract for planning — means editing a string in a file dedicated to exactly that, without touching orchestration logic at all. It also makes the model's actual instructions reviewable in one read, rather than reconstructed by tracing through function bodies.

**Why retry-then-skip instead of retry-forever or crash-on-first-error?**
At this scale — a dozen-plus network calls and at least two LLM calls per run — a dead URL, a blocked search, or a transient API error isn't an edge case, it's expected. Every fetch and every LLM call retries once with a short exponential backoff to absorb transient blips, then gives up and degrades rather than aborting the whole run: a failed page fetch becomes empty text and is skipped at synthesis time, a failed plan falls back to the raw topic, and a failed synthesis falls back to raw source snippets. The honest limitation is that failures are logged to the console, not surfaced to the end user as a structured "here's what didn't work" report — for a one-person CLI tool, a warning line and the source count in the output header have been enough.

---

## Project structure

```
research-agent/
├── main.py                       — CLI entry point: args, logging, run, save
├── research_agent/
│   ├── __init__.py                — package marker + version
│   ├── agent.py                   — ResearchAgent: the plan → gather → synthesize loop
│   ├── config.py                  — frozen Settings dataclass, env-driven
│   ├── llm.py                     — LLMClient: Groq / Gemini provider abstraction
│   ├── prompts.py                 — all LLM prompts, as named string constants
│   └── search.py                  — DuckDuckGo search + concurrent fetch + text extraction
├── tests/                         — pytest suite (28 tests), fully mocked, no network
│   ├── test_agent.py
│   ├── test_config.py
│   ├── test_llm.py
│   └── test_search.py
├── .github/workflows/ci.yml       — lint + test on Python 3.11 and 3.12
├── pyproject.toml                 — project metadata, dependencies, ruff + pytest config
├── requirements.txt               — runtime dependencies (alternative to pyproject extras)
├── .env.example                   — template for the environment variables below
├── .gitignore
└── LICENSE
```

---

## Getting started

### Prerequisites

- Python 3.11 or newer
- A free API key from Groq (default) or Google Gemini

### Install

```bash
git clone https://github.com/xTrapcoder/research-agent.git
cd research-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# or, for development (adds pytest + ruff):
pip install -e ".[dev]"
```

Installing in editable mode also exposes a `research-agent` console command equivalent to `python main.py`.

### Get a free API key

**Groq (default)** — fast inference on open-weight models, a generous free tier.
1. Sign up at [console.groq.com](https://console.groq.com).
2. Go to **API Keys → Create API Key** and copy the `gsk_…` value.

**Google Gemini (alternative)** — one `.env` line to switch.
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
2. Click **Create API key**.

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_key_here
```

### Run

```bash
# Basic
python main.py "What are the latest breakthroughs in nuclear fusion?"

# Broader coverage, more sources per query
python main.py "How does Python's asyncio event loop work?" --queries 5 --results 4

# Custom output path, debug logging
python main.py "History of the internet" --out reports/internet.md --verbose

# Help — works without an API key
python main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--queries N` | `4` | Sub-queries to generate (clamped to 2–8). |
| `--results N` | `3` | Search results fetched per sub-query (clamped to 1–8). |
| `--out PATH` | auto | Output Markdown path. Defaults to `outputs/<slug>_<timestamp>.md`. |
| `--verbose` | off | Enable DEBUG-level logging. |

---

## How it works

```
TOPIC
  │
  ▼
PLAN          ResearchAgent.plan()
  │             LLM (chat_json) decomposes the topic into N sub-queries
  │             on failure or empty result → falls back to [topic]
  ▼
GATHER        ResearchAgent.gather()
  │             search_web() each sub-query on DuckDuckGo        (sequential)
  │             de-duplicate hits by URL across all sub-queries
  │             fetch_many() downloads every unique page          (concurrent:
  │               asyncio.gather + semaphore + httpx.AsyncClient)
  │             each page → BeautifulSoup → <article>/<main>/<p> → text
  │             a failed fetch yields "" instead of raising
  ▼
SYNTHESIZE    ResearchAgent.synthesize()
  │             every source numbered [Source N] and handed to the LLM
  │             LLM (chat) writes Overview → Key Findings → Limitations → Sources
  │             on failure → falls back to a raw-snippet dump
  ▼
outputs/<slug>_<timestamp>.md   (+ metadata header: sources used, runtime)
```

---

## Development

```bash
pip install -e ".[dev]"   # installs pytest, pytest-asyncio, ruff

pytest -v                 # run the test suite (28 tests, fully mocked — no network)
ruff check .              # lint
```

CI (`.github/workflows/ci.yml`) runs both on every push and pull request, across Python 3.11 and 3.12.

---

## Roadmap

Near term:
- [x] Plan → gather → synthesize loop, written by hand (no framework)
- [x] Concurrent page fetching (`asyncio` + semaphore, ~8× faster than sequential)
- [x] Pluggable LLM providers (Groq / Gemini) selected via `.env`
- [x] Centralized, env-driven configuration
- [x] Graceful failure handling with retry-then-skip
- [x] Full unit test suite (28 tests) + GitHub Actions CI (Python 3.11 & 3.12)
- [ ] Response caching for search results and fetched pages, so re-running a topic skips redundant network calls

Mid term:
- [ ] Source ranking before synthesis (keyword overlap or embedding similarity) to improve report quality and trim token usage
- [ ] Follow-up query loop — read the first-pass findings and generate a second round of queries to fill coverage gaps
- [ ] Additional retrievers (arXiv, Wikipedia) for academic topics

Long term:
- [ ] Export formats beyond Markdown (PDF / HTML)
- [ ] Token / cost reporting surfaced in the run-stats header

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| HTTP | `httpx` — async client used for concurrent page fetching |
| Search | DuckDuckGo via `ddgs` (no API key required) |
| LLM providers | Groq (default) · Google Gemini (swap-in via `.env`) |
| Parsing | BeautifulSoup4 + lxml |
| Testing / CI | pytest, pytest-asyncio, ruff · GitHub Actions (3.11, 3.12) |

---

<p align="center">
Built by <a href="https://github.com/xTrapcoder">Aryan Sharma</a> · 2026<br/>
Email: <a href="mailto:aryansharma10011@gmail.com">aryansharma10011@gmail.com</a>
</p>
