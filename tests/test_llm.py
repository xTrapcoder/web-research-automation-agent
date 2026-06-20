"""Tests for LLMClient — all provider calls are mocked, no network used."""

import json

import pytest

from research_agent.config import Settings
from research_agent.llm import LLMClient, LLMError


def make_client() -> LLMClient:
    return LLMClient(Settings(provider="groq", llm_retries=1))


def test_chat_returns_text(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "_call_groq", lambda msgs, temp: "hello world")
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello world"


def test_chat_retries_then_succeeds(monkeypatch):
    client = make_client()
    calls = {"n": 0}

    def flaky(messages, temperature):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return "recovered"

    monkeypatch.setattr(client, "_call_groq", flaky)
    monkeypatch.setattr("time.sleep", lambda _: None)  # don't actually wait
    assert client.chat([{"role": "user", "content": "hi"}]) == "recovered"
    assert calls["n"] == 2


def test_chat_raises_after_exhausting_retries(monkeypatch):
    client = make_client()

    def always_fail(messages, temperature):
        raise RuntimeError("down")

    monkeypatch.setattr(client, "_call_groq", always_fail)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_json_parses_plain_json(monkeypatch):
    client = make_client()
    monkeypatch.setattr(client, "_call_groq", lambda m, t: json.dumps(["a", "b"]))
    assert client.chat_json([{"role": "user", "content": "x"}]) == ["a", "b"]


def test_chat_json_strips_markdown_fences(monkeypatch):
    client = make_client()
    fenced = '```json\n["x", "y", "z"]\n```'
    monkeypatch.setattr(client, "_call_groq", lambda m, t: fenced)
    assert client.chat_json([{"role": "user", "content": "x"}]) == ["x", "y", "z"]


def test_unknown_provider_raises():
    client = LLMClient(Settings(provider="bogus"))
    with pytest.raises(ValueError):
        client.chat([{"role": "user", "content": "hi"}])
