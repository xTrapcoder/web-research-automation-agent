"""Tests for the Settings configuration loader."""

from research_agent.config import Settings


def test_defaults():
    s = Settings()
    assert s.provider == "groq"
    assert s.max_concurrent_fetches == 8
    assert s.required_key_name == "GROQ_API_KEY"


def test_from_env_reads_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    s = Settings.from_env()
    assert s.provider == "gemini"
    assert s.required_key_name == "GEMINI_API_KEY"


def test_from_env_coerces_ints(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_FETCHES", "16")
    monkeypatch.setenv("FETCH_TIMEOUT", "30")
    s = Settings.from_env()
    assert s.max_concurrent_fetches == 16
    assert s.fetch_timeout == 30


def test_from_env_ignores_bad_ints(monkeypatch):
    # An unparsable value should fall back to the default, not crash.
    monkeypatch.setenv("FETCH_TIMEOUT", "not-a-number")
    s = Settings.from_env()
    assert s.fetch_timeout == 10


def test_settings_is_frozen():
    s = Settings()
    try:
        s.provider = "gemini"  # type: ignore[misc]
    except Exception as exc:
        assert "frozen" in str(exc).lower() or isinstance(exc, AttributeError)
    else:
        raise AssertionError("Settings should be immutable")
