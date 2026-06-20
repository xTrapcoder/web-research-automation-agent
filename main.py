"""
main.py — Command-line entry point.

Usage:
    python main.py "What are the latest breakthroughs in nuclear fusion?"
    python main.py "How does Python's GIL work?" --queries 5 --results 4
    python main.py "Best RAG practices" --out reports/rag.md --verbose

This file is intentionally thin: it parses arguments, configures logging,
validates the API key, runs the agent, and writes the report. All real work
lives in the research_agent package.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from research_agent.agent import ResearchAgent, RunResult
from research_agent.config import Settings


def configure_logging(verbose: bool) -> None:
    """Set up console logging. --verbose enables DEBUG; default is a clean INFO."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s" if not verbose else "%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the research agent."""
    parser = argparse.ArgumentParser(
        description="Web Research Automation Agent — plan, search, synthesize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "How does transformer attention work?"\n'
            '  python main.py "Latest AI safety research" --queries 5 --results 4\n'
            '  python main.py "Python asyncio patterns" --out reports/asyncio.md\n'
        ),
    )
    parser.add_argument("topic", help="The research topic or question.")
    parser.add_argument("--queries", type=int, default=4, metavar="N",
                        help="Sub-queries to generate (default: 4, clamped 2–8).")
    parser.add_argument("--results", type=int, default=3, metavar="N",
                        help="Results per query (default: 3, clamped 1–8).")
    parser.add_argument("--out", default="", metavar="PATH",
                        help="Output .md path (default: outputs/<slug>_<timestamp>.md).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    return parser


def default_output_path(topic: str) -> str:
    """Derive a readable default filename from the topic."""
    slug = "".join(c if c.isalnum() or c in " _" else "" for c in topic.lower())
    slug = "_".join(slug.split())[:50]
    return f"outputs/{slug}_{datetime.now():%Y%m%d_%H%M%S}.md"


def save_report(result: RunResult, output_path: str) -> Path:
    """Write the report to disk with a metadata header derived from RunStats."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = result.stats
    header = (
        f"<!-- Research Agent Report -->\n"
        f"<!-- Topic: {result.topic} -->\n"
        f"<!-- Generated: {datetime.now():%Y-%m-%d %H:%M:%S} -->\n"
        f"<!-- Sources: {s.n_sources_with_text}/{s.n_sources} usable | "
        f"Runtime: {s.seconds_total:.1f}s -->\n\n"
    )
    path.write_text(header + result.report, encoding="utf-8")
    return path


async def run() -> None:
    """Parse arguments, run one research agent pass, and print + save the report."""
    # Parse arguments first so --help (and argparse's own error handling)
    # works with zero configuration — neither one needs an API key.
    args = build_parser().parse_args()
    configure_logging(args.verbose)

    load_dotenv()
    settings = Settings.from_env()

    # Validate the API key before doing any real work
    if not os.getenv(settings.required_key_name):
        print(
            f"Error: {settings.required_key_name} not set. Add it to your .env file.\n"
            f"See README.md for how to get a free key."
        )
        sys.exit(1)

    n_queries = max(2, min(args.queries, 8))
    n_results = max(1, min(args.results, 8))
    output_path = args.out or default_output_path(args.topic)

    logging.info("Topic   : %s", args.topic)
    logging.info("Provider: %s | queries: %d | results/query: %d",
                 settings.provider, n_queries, n_results)

    agent = ResearchAgent(settings)
    result = await agent.run(args.topic, n_queries, n_results)

    saved = save_report(result, output_path)
    s = result.stats
    logging.info(
        "\nDone in %.1fs (plan %.1fs · search %.1fs · synth %.1fs) | %d/%d sources usable",
        s.seconds_total, s.seconds_plan, s.seconds_search, s.seconds_synthesize,
        s.n_sources_with_text, s.n_sources,
    )
    logging.info("Report saved to: %s\n", saved)
    print(result.report)


def main() -> None:
    """Synchronous entry point used by ``python main.py`` and the console script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
