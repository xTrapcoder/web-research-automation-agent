"""
config.py — Centralized configuration.

Every tunable knob in the project lives here, loaded from environment
variables with sensible defaults. This means behavior can be changed via
the .env file without editing any logic code, and there are no "magic
numbers" scattered across modules.

The Settings dataclass is frozen (immutable) so config can't be mutated
mid-run by accident — a small correctness guarantee.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default`` if unset/invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for a single agent run."""

    # --- LLM provider ---
    provider: str = "groq"           # "groq" or "gemini"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"
    llm_retries: int = 1             # retry count after first failure

    # --- Search & fetch ---
    fetch_timeout: int = 10          # seconds per HTTP request
    fetch_retries: int = 1           # retry count for a failed page fetch
    max_text_chars: int = 4_000      # truncate extracted page text to this length
    max_concurrent_fetches: int = 8  # semaphore limit for parallel downloads
    user_agent: str = (
        "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
    )

    @classmethod
    def from_env(cls) -> Settings:
        """Build a Settings instance from environment variables (after load_dotenv)."""
        return cls(
            provider=os.getenv("LLM_PROVIDER", "groq").lower(),
            groq_model=os.getenv("GROQ_MODEL", cls.groq_model),
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            llm_retries=_int_env("LLM_RETRIES", cls.llm_retries),
            fetch_timeout=_int_env("FETCH_TIMEOUT", cls.fetch_timeout),
            fetch_retries=_int_env("FETCH_RETRIES", cls.fetch_retries),
            max_text_chars=_int_env("MAX_TEXT_CHARS", cls.max_text_chars),
            max_concurrent_fetches=_int_env(
                "MAX_CONCURRENT_FETCHES", cls.max_concurrent_fetches
            ),
        )

    @property
    def required_key_name(self) -> str:
        """The env var name of the API key needed for the selected provider."""
        return "GROQ_API_KEY" if self.provider == "groq" else "GEMINI_API_KEY"
