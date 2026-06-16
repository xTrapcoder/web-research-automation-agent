"""
search.py — Web search and concurrent page-content extraction.

Public surface:
  - search_web(query, max_results, settings)        → list[SearchResult]
  - fetch_many(results, settings)                   → list[Source]  (concurrent)

Design notes:
  1. DuckDuckGo (via `ddgs`) needs no API key, keeping the project fully free.
  2. Page fetching is the slow, I/O-bound part of a run: a dozen HTTP requests
     that each spend most of their time waiting on the network. We fetch them
     *concurrently* with asyncio + httpx.AsyncClient, bounded by a semaphore so
     we never open more than Settings.max_concurrent_fetches sockets at once.
     This turns ~12 sequential 1–10s fetches into a single wave, cutting wall
     time roughly 5x in practice.
  3. Every fetch is wrapped so a single dead URL can never crash the run — a
     failed page yields an empty body and is simply skipped downstream.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search hit, before its page has been fetched."""

    title: str
    url: str
    snippet: str


@dataclass
class Source:
    """A fetched source: search metadata plus extracted page text."""

    title: str
    url: str
    snippet: str
    text: str  # extracted page body, or "" if the fetch failed


def search_web(query: str, max_results: int, settings: Settings) -> list[SearchResult]:
    """
    Run a DuckDuckGo search and return structured results.

    Returns [] on any error so the caller can skip this query and continue.
    DDGS uses 'href' for the URL and 'body' for the excerpt; we normalize both.
    """
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Search failed for %r: %s", query, exc)
        return []

    return [
        SearchResult(
            title=r.get("title", "Untitled"),
            url=r.get("href", ""),
            snippet=r.get("body", ""),
        )
        for r in raw
        if r.get("href")
    ]


def _extract_text(html: str, max_chars: int) -> str:
    """
    Extract readable text from raw HTML.

    Priority: a semantic <article> or <main> container (which usually holds the
    real content) → otherwise all <p> tags. Script/style/nav/footer/header tags
    are removed first so their text doesn't pollute the result. Output is
    whitespace-collapsed and truncated to ``max_chars``.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    container = soup.find("article") or soup.find("main")
    if container:
        text = container.get_text(separator=" ", strip=True)
    else:
        text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))

    text = " ".join(text.split())  # collapse runs of whitespace
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


async def _fetch_one(
    client: httpx.AsyncClient,
    result: SearchResult,
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> Source:
    """
    Fetch and parse a single page, retrying once on failure.

    The semaphore caps how many of these run at the same time. On total
    failure we return a Source with text="" rather than raising, so one bad
    URL never aborts the gather() below.
    """
    async with semaphore:  # block here if max concurrency is already in flight
        last_exc: Exception | None = None
        for attempt in range(settings.fetch_retries + 1):
            try:
                resp = await client.get(result.url)
                resp.raise_for_status()
                text = _extract_text(resp.text, settings.max_text_chars)
                return Source(result.title, result.url, result.snippet, text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < settings.fetch_retries:
                    await asyncio.sleep(2)

        logger.warning("Could not fetch %s: %s", result.url, last_exc)
        return Source(result.title, result.url, result.snippet, text="")


async def fetch_many(results: list[SearchResult], settings: Settings) -> list[Source]:
    """
    Fetch every result concurrently and return the fetched Sources.

    All requests share one AsyncClient (connection pooling) and are bounded by
    a single semaphore. asyncio.gather waits for the whole wave to finish.
    """
    if not results:
        return []

    semaphore = asyncio.Semaphore(settings.max_concurrent_fetches)
    async with httpx.AsyncClient(
        timeout=settings.fetch_timeout,
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent},
    ) as client:
        tasks = [_fetch_one(client, r, settings, semaphore) for r in results]
        return await asyncio.gather(*tasks)
