"""
agent.py — The research agent orchestration.

Implements the three-phase loop end to end:
    plan() → gather() → synthesize()

The ResearchAgent class owns a Settings object and an LLMClient, and exposes
a single async run() method that returns the finished Markdown report plus a
small RunStats record for observability.

The loop is written by hand (no agent framework) so every step is explicit and
traceable — useful both for understanding and for explaining in interviews.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import prompts, search
from .config import Settings
from .llm import LLMClient
from .search import Source

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """Lightweight metrics captured during a run, for reporting and observability."""

    n_queries: int = 0
    n_sources: int = 0
    n_sources_with_text: int = 0
    seconds_plan: float = 0.0
    seconds_search: float = 0.0
    seconds_synthesize: float = 0.0

    @property
    def seconds_total(self) -> float:
        return self.seconds_plan + self.seconds_search + self.seconds_synthesize


@dataclass
class RunResult:
    """The output of a run: the report text and the stats for it."""

    topic: str
    report: str
    sources: list[Source] = field(default_factory=list)
    stats: RunStats = field(default_factory=RunStats)


class ResearchAgent:
    """Plans, gathers, and synthesizes a cited research report for a topic."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)

    # ------------------------------------------------------------------
    # Phase 1 — Plan
    # ------------------------------------------------------------------
    def plan(self, topic: str, n_queries: int) -> list[str]:
        """
        Ask the LLM to decompose the topic into focused sub-queries.

        Falls back to ``[topic]`` if the LLM call or JSON parse fails, so the
        run can always proceed to searching.
        """
        logger.info("Planning %d sub-queries for: %s", n_queries, topic)
        messages = [
            {"role": "system", "content": prompts.PLAN_SYSTEM},
            {"role": "user", "content": prompts.PLAN_USER.format(topic=topic, n_queries=n_queries)},
        ]
        try:
            queries = self.llm.chat_json(messages, temperature=0.4)
            if not isinstance(queries, list) or not queries:
                raise ValueError("LLM returned a non-list or empty list")
            queries = [str(q) for q in queries[:n_queries]]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Planning failed (%s); falling back to raw topic.", exc)
            return [topic]

        for i, q in enumerate(queries, 1):
            logger.info("  query %d: %s", i, q)
        return queries

    # ------------------------------------------------------------------
    # Phase 2 — Gather (search every query, then fetch all pages concurrently)
    # ------------------------------------------------------------------
    async def gather(self, queries: list[str], results_per_query: int) -> list[Source]:
        """
        Search each sub-query for URLs, then fetch every page concurrently.

        We deliberately separate "collect URLs" (fast, sequential) from "fetch
        bodies" (slow, parallel): all fetches across all queries are batched
        into one concurrent wave for maximum throughput.
        """
        # Collect search hits from every query into one flat list
        all_hits: list[search.SearchResult] = []
        for i, query in enumerate(queries, 1):
            hits = search.search_web(query, results_per_query, self.settings)
            logger.info("  [%d/%d] %r → %d hits", i, len(queries), query, len(hits))
            all_hits.extend(hits)

        # De-duplicate by URL so we don't fetch the same page twice
        seen: set[str] = set()
        unique_hits = []
        for h in all_hits:
            if h.url not in seen:
                seen.add(h.url)
                unique_hits.append(h)

        logger.info("Fetching %d unique pages concurrently…", len(unique_hits))
        sources = await search.fetch_many(unique_hits, self.settings)
        return sources

    # ------------------------------------------------------------------
    # Phase 3 — Synthesize
    # ------------------------------------------------------------------
    def synthesize(self, topic: str, sources: list[Source]) -> str:
        """
        Build a numbered-source prompt and have the LLM write the final report.

        Sources are numbered [Source 1..N]; the prompt instructs the model to
        cite them inline. On LLM failure we return a fallback report containing
        raw snippets so a run is never a total loss.
        """
        if not sources:
            return (
                f"# Research Report: {topic}\n\n"
                "> **Error:** No sources could be retrieved. Check your connection and retry."
            )

        block = "\n".join(
            f"--- SOURCE {i} ---\n"
            f"Title: {s.title}\n"
            f"URL: {s.url}\n"
            f"Content:\n{(s.text or '[no content]')[:3000]}\n"
            for i, s in enumerate(sources, 1)
        )
        messages = [
            {"role": "system", "content": prompts.SYNTHESIS_SYSTEM},
            {"role": "user", "content": prompts.SYNTHESIS_USER.format(
                topic=topic, n_sources=len(sources), sources_block=block
            )},
        ]
        logger.info("Synthesizing report from %d sources…", len(sources))
        try:
            return self.llm.chat(messages, temperature=0.5)
        except Exception as exc:  # noqa: BLE001
            logger.error("Synthesis failed (%s); emitting raw-snippet fallback.", exc)
            snippets = "\n\n".join(
                f"**[Source {i}]** {s.url}\n{(s.text or '')[:300]}…"
                for i, s in enumerate(sources, 1)
                if s.text
            )
            return (
                f"# Research Report: {topic}\n\n"
                f"> **Warning:** Synthesis failed; raw snippets follow.\n\n{snippets}"
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    async def run(self, topic: str, n_queries: int, results_per_query: int) -> RunResult:
        """Execute the full plan → gather → synthesize pipeline with timing."""
        stats = RunStats()

        t0 = time.perf_counter()
        queries = self.plan(topic, n_queries)
        stats.seconds_plan = time.perf_counter() - t0
        stats.n_queries = len(queries)

        t0 = time.perf_counter()
        sources = await self.gather(queries, results_per_query)
        stats.seconds_search = time.perf_counter() - t0
        stats.n_sources = len(sources)
        stats.n_sources_with_text = sum(1 for s in sources if s.text)

        t0 = time.perf_counter()
        report = self.synthesize(topic, sources)
        stats.seconds_synthesize = time.perf_counter() - t0

        return RunResult(topic=topic, report=report, sources=sources, stats=stats)
