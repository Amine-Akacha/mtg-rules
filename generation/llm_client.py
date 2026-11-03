"""
Abstraction over Anthropic and OpenAI LLM APIs.

The active provider is selected by the LLM_PROVIDER environment variable
(or overridden per-request). Clients are created lazily and reused.
"""

from functools import lru_cache

import anthropic
import openai

from config import settings


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot produce an answer."""


class LLMConfigError(LLMError):
    """Raised when the provider is not configured (e.g. missing API key)."""


@lru_cache(maxsize=1)
def _anthropic_client() -> anthropic.Anthropic:
    if not settings.ANTHROPIC_API_KEY:
        raise LLMConfigError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def _openai_client() -> openai.OpenAI:
    if not settings.OPENAI_API_KEY:
        raise LLMConfigError("OPENAI_API_KEY is not set.")
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """Send a request to the Anthropic Messages API and return the text."""
    try:
        message = _anthropic_client().messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        raise LLMError(f"Anthropic API error: {exc}") from exc
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai(system_prompt: str, user_message: str) -> str:
    """Send a request to the OpenAI Chat Completions API and return the text."""
    try:
        response = _openai_client().chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
    except openai.APIError as exc:
        raise LLMError(f"OpenAI API error: {exc}") from exc
    return response.choices[0].message.content or ""


def generate(
    system_prompt: str,
    user_message: str,
    provider: str | None = None,
) -> str:
    """
    Generate a response using the configured (or overridden) LLM provider.

    Raises:
        LLMConfigError: When the provider's API key is missing.
        LLMError: When the provider API call fails.
        ValueError: When an unknown provider string is supplied.
    """
    active = (provider or settings.LLM_PROVIDER).lower()

    if active == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    if active == "openai":
        return _call_openai(system_prompt, user_message)
    raise ValueError(f"Unknown LLM provider {active!r}. Must be 'anthropic' or 'openai'.")
