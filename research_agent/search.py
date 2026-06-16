"""
search.py — Web search and page-content extraction for the research agent.

Two public functions that main.py calls:
  - search_web(query, max_results)  → list of {title, url, snippet} dicts
  - fetch_page_text(url)            → cleaned plain text (or "" on failure)

Design decisions:
  1. DuckDuckGo via `ddgs` needs no API key, making the project fully free.
  2. httpx is used instead of requests because it has cleaner timeout handling
     and is considered more modern, but the logic is identical — easy to swap.
  3. BeautifulSoup is the de-facto standard for HTML parsing in Python; we
     target <article>, <main>, and <p> tags in priority order to get the meat
     of the page while skipping nav/footer noise.
  4. All failures are caught and returned as empty values so the caller
     (main.py) can skip bad sources without crashing the whole run.
"""

import time

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Constants — tuning knobs in one place
# ---------------------------------------------------------------------------
FETCH_TIMEOUT_SECONDS = 10      # per-page HTTP timeout
MAX_TEXT_CHARS = 4_000          # truncate page text to keep LLM context small
USER_AGENT = (
    "Mozilla/5.0 (compatible; ResearchAgent/1.0; +https://github.com/example)"
)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a DuckDuckGo search and return structured results.

    Args:
        query:       The search query string.
        max_results: How many results to request from DDGS.

    Returns:
        List of dicts, each with keys: "title", "url", "snippet".
        Returns [] if the search errors or returns nothing.

    Why DuckDuckGo? No API key, no rate-limit tiers, good result quality
    for general research queries, and the `ddgs` package is actively maintained.
    The downside is that it can be blocked if you hammer it; for a portfolio
    project running a handful of queries this is fine.
    """
    try:
        # DDGS() is a context manager; .text() returns a generator of result dicts
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        if not raw_results:
            return []

        # Normalize to our internal schema — DDGS uses "href" for the URL
        results = []
        for r in raw_results:
            results.append({
                "title":   r.get("title", "Untitled"),
                "url":     r.get("href", ""),
                "snippet": r.get("body", ""),   # DDGS calls the excerpt "body"
            })
        return results

    except Exception as exc:
        # Don't let a search failure crash the agent — just log and return empty
        print(f"  [search] DuckDuckGo error for '{query}': {exc}")
        return []


def fetch_page_text(url: str, retries: int = 1) -> str:
    """
    Download a web page and extract readable plain text from its HTML.

    Strategy:
      1. Prefer <article> or <main> tags — these usually hold the main content.
      2. Fall back to all <p> tags if neither semantic container exists.
      3. Strip excess whitespace and truncate to MAX_TEXT_CHARS.

    Args:
        url:     The URL to fetch.
        retries: Retry count on network errors (default: 1 = try twice total).

    Returns:
        Extracted text string, or "" if fetching/parsing fails.

    Interview talking point: "We truncate to 4 000 chars per page. With 5
    queries × 3 pages that's ~60 KB of context fed to the LLM for synthesis.
    Enough signal, not enough to blow the context window."
    """
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            # httpx.get with a timeout so we don't hang on slow servers
            response = httpx.get(
                url,
                timeout=FETCH_TIMEOUT_SECONDS,
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            )
            # Raise an exception for 4xx / 5xx status codes
            response.raise_for_status()

            # --- HTML → text extraction ---
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script / style tags so their text doesn't pollute output
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Priority: semantic article/main container > all paragraphs
            container = soup.find("article") or soup.find("main")
            if container:
                text = container.get_text(separator=" ", strip=True)
            else:
                paragraphs = soup.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs)

            # Collapse runs of whitespace into single spaces
            text = " ".join(text.split())

            # Truncate to keep LLM prompts a manageable size
            if len(text) > MAX_TEXT_CHARS:
                text = text[:MAX_TEXT_CHARS] + "…"

            return text

        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2
                print(f"  [fetch] Attempt {attempt + 1} failed for {url} ({exc}). Retrying in {wait}s…")
                time.sleep(wait)

    # All attempts failed — log and return empty string so caller can skip
    print(f"  [fetch] Could not fetch {url}: {last_exc}")
    return ""


def search_and_fetch(
    query: str,
    max_results: int = 3,
) -> list[dict]:
    """
    High-level helper: search → fetch each result → return enriched dicts.

    Combines search_web() and fetch_page_text() into a single call that
    main.py uses per sub-query. Skips sources whose page text is empty.

    Returns:
        List of dicts with keys: "title", "url", "snippet", "text".
        "text" is the full extracted page content (or "" if fetch failed).
    """
    results = search_web(query, max_results=max_results)
    enriched = []

    for i, r in enumerate(results, start=1):
        url = r["url"]
        if not url:
            continue

        print(f"    Fetching source {i}/{len(results)}: {url[:70]}…")
        page_text = fetch_page_text(url)

        enriched.append({
            "title":   r["title"],
            "url":     url,
            "snippet": r["snippet"],
            "text":    page_text,
        })

    return enriched
