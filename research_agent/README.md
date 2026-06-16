# Research Agent

A command-line agent that takes a research topic, autonomously searches the web,
and synthesizes a clean, cited Markdown report — built from scratch in plain Python,
with no agent framework and no paid APIs.

```bash
python main.py "How does nuclear fusion work and what's the current state of the field?"
```

[![CI](https://github.com/your-username/research-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/research-agent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What It Does

Given a topic, the agent runs a three-phase loop:

1. **Plan** — an LLM decomposes the topic into 3–5 focused, distinct search queries.
2. **Gather** — each query is searched on DuckDuckGo; the resulting pages are then
   **fetched concurrently** and reduced to clean text.
3. **Synthesize** — all collected text is passed to the LLM, which writes a
   structured report with an overview, labelled key-finding sections, a limitations
   section, and a numbered source list with inline `[Source N]` citations.

The report is saved to `outputs/` with a metadata header (sources used, runtime).

---

## Why It's Built This Way

| Decision | Reasoning |
|---|---|
| **No agent framework** | The plan→gather→synthesize loop is written by hand, so every step is explicit and traceable rather than hidden behind framework magic. |
| **Concurrent fetching** | Page downloads are I/O-bound and dominate runtime. They run in parallel via `asyncio` + `httpx.AsyncClient`, bounded by a semaphore — roughly an 8× speedup over sequential fetching. |
| **Provider abstraction** | `LLMClient` hides Groq/Gemini differences. Switching providers is an `.env` change, not a code change. |
| **Centralized config** | Every tunable knob lives in one frozen `Settings` dataclass loaded from the environment — no magic numbers scattered across files. |
| **Prompts as constants** | All prompts live in `prompts.py`, so behavior is tuned by editing strings, not logic. |
| **Graceful degradation** | A dead URL, an empty search, or an LLM timeout is retried once then skipped. A single bad source can never crash a run. |
| **Tested** | 28 unit tests cover config parsing, retry logic, HTML extraction, concurrent-fetch failure handling, and the full orchestration — all with mocked I/O (no network in CI). |

---

## Project Structure

```
research-agent/
├── main.py                       # thin CLI: args, logging, run, save
├── research_agent/
│   ├── __init__.py
│   ├── config.py                 # frozen Settings dataclass (env-driven)
│   ├── prompts.py                # all LLM prompts
│   ├── llm.py                    # LLMClient — Groq / Gemini abstraction
│   ├── search.py                 # DuckDuckGo search + concurrent fetch + extract
│   └── agent.py                  # ResearchAgent — the three-phase loop
├── tests/                        # 28 pytest tests, fully mocked
│   ├── test_config.py
│   ├── test_llm.py
│   ├── test_search.py
│   └── test_agent.py
├── .github/workflows/ci.yml      # lint + test on 3.11 & 3.12
├── pyproject.toml                # deps, ruff, pytest config
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Install

```bash
git clone https://github.com/your-username/research-agent.git
cd research-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Get a free API key

**Groq (default — fast, generous free tier)**
1. Sign up at [console.groq.com](https://console.groq.com).
2. **API Keys → Create API Key**, copy the `gsk_…` value.

**Google Gemini (alternative)**
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
2. **Create API key**.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_key_here
```

---

## Usage

```bash
# Basic
python main.py "What are the latest breakthroughs in quantum computing?"

# Broader coverage / deeper sourcing
python main.py "How does Python's asyncio event loop work?" --queries 5 --results 4

# Custom output path + debug logging
python main.py "History of the internet" --out reports/internet.md --verbose

# Help
python main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--queries N` | 4 | Sub-queries to generate (clamped 2–8). |
| `--results N` | 3 | Web results to fetch per query (clamped 1–8). |
| `--out PATH` | auto | Output Markdown path. |
| `--verbose` | off | DEBUG-level logging. |

---

## How It Works (the loop)

```
USER TOPIC
   │
   ▼
PHASE 1 — PLAN            llm.chat_json(PLAN_PROMPT)
   │   topic → ["sub-query 1", "sub-query 2", "sub-query 3", "sub-query 4"]
   ▼
PHASE 2 — GATHER
   │   for each sub-query:  DuckDuckGo → list of URLs        (sequential, fast)
   │   de-duplicate URLs across all queries
   │   fetch ALL pages CONCURRENTLY  (asyncio.gather + semaphore + httpx)
   │   each page → BeautifulSoup → <article>/<main>/<p> text → truncate
   ▼
PHASE 3 — SYNTHESIZE     llm.chat(SYNTHESIS_PROMPT + numbered sources)
   │   → structured Markdown: Overview · Key Findings · Limitations · Sources
   ▼
outputs/<slug>_<timestamp>.md   (+ metadata header)
```

---

## Development

```bash
pip install -e ".[dev]"   # installs pytest, pytest-asyncio, ruff

pytest -v                 # run the test suite (28 tests)
ruff check .              # lint
```

CI runs both on every push and pull request across Python 3.11 and 3.12.

---

## Extending It

| Goal | Where to change |
|---|---|
| Add a search backend (Bing, Serper…) | `search.py` — add a `search_*` function returning `SearchResult`s |
| Add an LLM provider | `llm.py` — add a `_call_<name>` method and a dispatch entry |
| Change report structure | `prompts.py` — edit `SYNTHESIS_USER` |
| Tune timeouts / concurrency | `.env` or `config.py` defaults |

---

## Roadmap

**Done**
- [x] Plan → gather → synthesize loop, written by hand (no framework)
- [x] Concurrent page fetching (`asyncio` + semaphore, ~8× faster than sequential)
- [x] Pluggable LLM providers (Groq / Gemini) selected via `.env`
- [x] Centralized, env-driven configuration
- [x] Graceful failure handling with retry-then-skip
- [x] 28 unit tests + GitHub Actions CI (Python 3.11 & 3.12)

**Planned**
- [ ] **Response caching** — cache search results and fetched pages on disk so
      re-running a topic skips redundant network calls.
- [ ] **Source ranking** — score and keep only the most relevant pages before
      synthesis (e.g. keyword overlap or embedding similarity), to improve
      report quality and trim token usage.
- [ ] **Follow-up query loop** — let the agent read its first-pass findings and
      generate a second round of queries to fill gaps (iterative depth).
- [ ] **Export formats** — optional PDF / HTML output in addition to Markdown.
- [ ] **Additional retrievers** — arXiv and Wikipedia backends for academic topics.
- [ ] **Token / cost reporting** — surface tokens used per run in the stats header.

Contributions and suggestions are welcome — open an issue to discuss.

---

## License

MIT.
