"""Tests for ResearchAgent orchestration. LLM and search are mocked."""

import pytest

from research_agent import agent as agent_mod
from research_agent.agent import ResearchAgent
from research_agent.config import Settings
from research_agent.search import SearchResult, Source

SETTINGS = Settings()


def make_agent() -> ResearchAgent:
    return ResearchAgent(SETTINGS)


# --------------------------------------------------------------------------
# Phase 1 — plan
# --------------------------------------------------------------------------
def test_plan_returns_queries(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(a.llm, "chat_json", lambda m, temperature=0.4: ["q1", "q2", "q3"])
    assert a.plan("topic", 3) == ["q1", "q2", "q3"]


def test_plan_caps_to_n_queries(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(a.llm, "chat_json", lambda m, temperature=0.4: ["q1", "q2", "q3", "q4", "q5"])
    assert a.plan("topic", 2) == ["q1", "q2"]


def test_plan_falls_back_on_error(monkeypatch):
    a = make_agent()
    def boom(m, temperature=0.4): raise RuntimeError("llm down")
    monkeypatch.setattr(a.llm, "chat_json", boom)
    assert a.plan("my topic", 3) == ["my topic"]


def test_plan_falls_back_on_empty_list(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(a.llm, "chat_json", lambda m, temperature=0.4: [])
    assert a.plan("my topic", 3) == ["my topic"]


# --------------------------------------------------------------------------
# Phase 2 — gather (de-dup + concurrent fetch mocked)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gather_dedupes_urls(monkeypatch):
    a = make_agent()

    # Two queries return overlapping URLs
    def fake_search(query, n, settings):
        return [SearchResult("T", "https://dup.com", "")]

    captured = {}
    async def fake_fetch_many(results, settings):
        captured["count"] = len(results)
        return [Source(r.title, r.url, r.snippet, "text") for r in results]

    monkeypatch.setattr(agent_mod.search, "search_web", fake_search)
    monkeypatch.setattr(agent_mod.search, "fetch_many", fake_fetch_many)

    await a.gather(["q1", "q2"], results_per_query=1)
    assert captured["count"] == 1  # duplicate URL fetched only once


# --------------------------------------------------------------------------
# Phase 3 — synthesize
# --------------------------------------------------------------------------
def test_synthesize_no_sources_returns_error_report():
    a = make_agent()
    report = a.synthesize("topic", [])
    assert "No sources" in report


def test_synthesize_calls_llm(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(a.llm, "chat", lambda m, temperature=0.5: "# Final Report")
    sources = [Source("T", "https://x.com", "s", "body text")]
    assert a.synthesize("topic", sources) == "# Final Report"


def test_synthesize_fallback_on_llm_failure(monkeypatch):
    a = make_agent()
    def boom(m, temperature=0.5): raise RuntimeError("synth down")
    monkeypatch.setattr(a.llm, "chat", boom)
    sources = [Source("T", "https://x.com", "s", "important body")]
    report = a.synthesize("topic", sources)
    assert "Synthesis failed" in report
    assert "important body" in report  # raw snippet preserved


# --------------------------------------------------------------------------
# Full run end-to-end (everything mocked)
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_end_to_end(monkeypatch):
    a = make_agent()
    monkeypatch.setattr(a, "plan", lambda topic, n: ["q1", "q2"])

    async def fake_gather(queries, rpq):
        return [Source("T", "https://x.com", "s", "body")]

    monkeypatch.setattr(a, "gather", fake_gather)
    monkeypatch.setattr(a, "synthesize", lambda topic, sources: "# Report")

    result = await a.run("topic", n_queries=2, results_per_query=1)
    assert result.report == "# Report"
    assert result.stats.n_queries == 2
    assert result.stats.n_sources == 1
    assert result.stats.seconds_total >= 0
