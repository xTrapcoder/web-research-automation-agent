"""Tests for search and fetch logic. Network and HTTP are mocked throughout."""

import pytest

from research_agent import search
from research_agent.config import Settings
from research_agent.search import SearchResult, _extract_text, fetch_many

SETTINGS = Settings(fetch_retries=0, max_concurrent_fetches=4, max_text_chars=100)


# --------------------------------------------------------------------------
# HTML extraction
# --------------------------------------------------------------------------
def test_extract_prefers_article():
    html = "<html><body><article><p>Real content here.</p></article>"\
           "<footer>junk footer</footer></body></html>"
    text = _extract_text(html, max_chars=1000)
    assert "Real content here." in text
    assert "junk footer" not in text


def test_extract_falls_back_to_paragraphs():
    html = "<html><body><p>First.</p><p>Second.</p></body></html>"
    text = _extract_text(html, max_chars=1000)
    assert "First." in text and "Second." in text


def test_extract_strips_scripts():
    html = "<html><body><p>Keep this.</p><script>evil()</script></body></html>"
    text = _extract_text(html, max_chars=1000)
    assert "Keep this." in text
    assert "evil" not in text


def test_extract_truncates():
    html = "<p>" + ("x" * 500) + "</p>"
    text = _extract_text(html, max_chars=50)
    assert len(text) <= 51  # 50 chars + the ellipsis


# --------------------------------------------------------------------------
# search_web error handling (DDGS mocked)
# --------------------------------------------------------------------------
def test_search_web_handles_exception(monkeypatch):
    class BoomDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise RuntimeError("blocked")

    monkeypatch.setattr(search, "DDGS", BoomDDGS)
    assert search.search_web("anything", 3, SETTINGS) == []


def test_search_web_normalizes_fields(monkeypatch):
    class FakeDDGS:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k):
            return [{"title": "T", "href": "https://x.com", "body": "snippet"}]

    monkeypatch.setattr(search, "DDGS", FakeDDGS)
    results = search.search_web("q", 1, SETTINGS)
    assert results[0].url == "https://x.com"
    assert results[0].snippet == "snippet"


# --------------------------------------------------------------------------
# Concurrent fetch (httpx mocked) — verifies a dead URL doesn't kill the wave
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_many_continues_past_failures(monkeypatch):
    good = SearchResult("Good", "https://good.com", "")
    bad = SearchResult("Bad", "https://bad.com", "")

    class FakeResponse:
        text = "<article><p>good page</p></article>"
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            if "bad" in url:
                raise RuntimeError("connection refused")
            return FakeResponse()

    monkeypatch.setattr(search.httpx, "AsyncClient", FakeClient)
    sources = await fetch_many([good, bad], SETTINGS)

    by_url = {s.url: s for s in sources}
    assert "good page" in by_url["https://good.com"].text
    assert by_url["https://bad.com"].text == ""  # failed fetch → empty, not a crash


@pytest.mark.asyncio
async def test_fetch_many_empty_input():
    assert await fetch_many([], SETTINGS) == []
