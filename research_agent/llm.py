"""
llm.py — LLM call wrapper for the research agent.

Design decision: All prompts live in main.py as named string constants.
This module handles only the mechanics of calling the API, retrying once
on failure, and parsing JSON when the caller expects structured output.

Currently wired to Groq (free tier, ~10x faster than OpenAI on simple tasks).
To swap to Gemini: flip LLM_PROVIDER in your .env to "gemini" — no code
changes needed beyond that.
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# Provider detection — read once at import time so callers don't need to know
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()


def _call_groq(messages: list[dict], model: str, temperature: float) -> str:
    """
    Send a chat request to the Groq API and return the assistant's text.

    Why Groq? It runs open-weight models (Llama-3, Mixtral) on custom
    hardware and offers a generous free tier with very low latency —
    ideal for a portfolio project that makes many sequential LLM calls.
    """
    from groq import Groq  # lazy import so Gemini users don't need this

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model or "llama-3.3-70b-versatile",  # default: best free model on Groq
        messages=messages,
        temperature=temperature,
    )
    # response.choices[0].message.content is the standard OpenAI-style field
    return response.choices[0].message.content


def _call_gemini(messages: list[dict], model: str, temperature: float) -> str:
    """
    Send a chat request to Google Gemini (free tier via google-generativeai).

    The Gemini SDK uses a different message format than OpenAI, so we adapt:
    - "system" role → prepended to the first user message (Gemini has no system role)
    - "user" / "model" roles map directly
    """
    import google.generativeai as genai  # lazy import

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(
        model_name=model or "gemini-1.5-flash",  # fastest free-tier model
        generation_config=genai.GenerationConfig(temperature=temperature),
    )

    # Adapt OpenAI-style messages → Gemini history format
    history = []
    prompt_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            # Gemini has no system role; prepend to prompt
            prompt_text = content + "\n\n"
        elif role == "user":
            history.append({"role": "user", "parts": [prompt_text + content]})
            prompt_text = ""
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})

    # The last user message drives the generation
    last_user = history.pop() if history else {"role": "user", "parts": [prompt_text]}
    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(last_user["parts"][0])
    return response.text


def chat(
    messages: list[dict],
    *,
    model: str = "",
    temperature: float = 0.3,
    retries: int = 1,
) -> str:
    """
    Call the configured LLM provider with retry-once-on-failure semantics.

    Args:
        messages:    OpenAI-style list of {"role": ..., "content": ...} dicts.
        model:       Override the default model name. Empty = use default.
        temperature: Sampling temperature (0 = deterministic, 1 = creative).
        retries:     How many times to retry after a failure before raising.

    Returns:
        The assistant's reply as a plain string.

    Interview talking point: "retry once, then propagate" keeps the agent
    resilient to transient API errors without masking real bugs.
    """
    dispatch = {"groq": _call_groq, "gemini": _call_gemini}
    fn = dispatch.get(LLM_PROVIDER)
    if fn is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Choose 'groq' or 'gemini'.")

    last_exc: Exception | None = None
    for attempt in range(retries + 1):  # attempt 0 = first try, attempt 1 = retry
        try:
            return fn(messages, model, temperature)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                wait = 2 ** attempt  # exponential back-off: 1s, 2s, 4s…
                print(f"  [LLM] Attempt {attempt + 1} failed ({exc}). Retrying in {wait}s…")
                time.sleep(wait)

    # All attempts exhausted — re-raise so the caller can decide what to do
    raise RuntimeError(f"LLM call failed after {retries + 1} attempts") from last_exc


def chat_json(messages: list[dict], **kwargs) -> dict | list:
    """
    Convenience wrapper: call chat() and parse the response as JSON.

    The LLM is instructed (via the caller's prompt) to respond with pure JSON.
    We strip markdown fences (```json ... ```) defensively because some models
    add them even when told not to.

    Returns the parsed Python object (dict or list).
    Raises json.JSONDecodeError if the model returns non-JSON text.
    """
    raw = chat(messages, **kwargs)

    # Strip ```json fences if present — a common model quirk
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Find the first newline (end of opening fence) and last ``` closing fence
        cleaned = cleaned.split("\n", 1)[-1]           # drop "```json" line
        cleaned = cleaned.rsplit("```", 1)[0].strip()  # drop closing "```"

    return json.loads(cleaned)
