"""
llm.py — LLM provider abstraction.

Exposes a single LLMClient class that hides provider differences behind two
methods: chat() (returns text) and chat_json() (parses the reply as JSON).

Two providers are supported and selected via Settings.provider:
  - groq   (default) — open-weight models on fast inference hardware, free tier
  - gemini           — Google's free-tier models

Swapping providers requires only an env change (LLM_PROVIDER=gemini); no code
edits. New providers are added by writing one private _call_<name> method.
"""

from __future__ import annotations

import json
import logging
import os
import time

from .config import Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when all LLM attempts (including retries) have failed."""


class LLMClient:
    """Thin, provider-agnostic wrapper around a chat-completion API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------
    def _call_groq(self, messages: list[dict], temperature: float) -> str:
        """Call the Groq chat-completions API and return the assistant text."""
        from groq import Groq  # lazy import: gemini users don't need this installed

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=self.settings.groq_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _call_gemini(self, messages: list[dict], temperature: float) -> str:
        """
        Call the Gemini API. Gemini has no 'system' role, so we fold any
        system message into the first user turn before sending.
        """
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(
            model_name=self.settings.gemini_model,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )

        history: list[dict] = []
        carry = ""  # holds a pending system message to prepend to next user turn
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                carry = content + "\n\n"
            elif role == "user":
                history.append({"role": "user", "parts": [carry + content]})
                carry = ""
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        last = history.pop() if history else {"role": "user", "parts": [carry]}
        chat = model.start_chat(history=history)
        return chat.send_message(last["parts"][0]).text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        """
        Send a chat request, retrying once on failure (per Settings.llm_retries).

        Retry-then-raise keeps the agent resilient to transient API hiccups
        without silently swallowing genuine errors.
        """
        dispatch = {"groq": self._call_groq, "gemini": self._call_gemini}
        fn = dispatch.get(self.settings.provider)
        if fn is None:
            raise ValueError(
                f"Unknown provider '{self.settings.provider}'. Use 'groq' or 'gemini'."
            )

        retries = self.settings.llm_retries
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return fn(messages, temperature)
            except Exception as exc:  # noqa: BLE001 — we deliberately retry on any error
                last_exc = exc
                if attempt < retries:
                    wait = 2 ** attempt  # 1s, 2s, 4s … exponential back-off
                    logger.warning(
                        "LLM attempt %d failed (%s). Retrying in %ds.",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)

        raise LLMError(f"LLM call failed after {retries + 1} attempts") from last_exc

    def chat_json(self, messages: list[dict], *, temperature: float = 0.3) -> dict | list:
        """
        Call chat() and parse the reply as JSON.

        Strips ```json fences defensively, since some models wrap JSON in a
        markdown code block even when instructed not to.
        """
        raw = self.chat(messages, temperature=temperature)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]        # drop opening fence line
            cleaned = cleaned.rsplit("```", 1)[0].strip()  # drop closing fence
        return json.loads(cleaned)
