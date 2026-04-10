"""
Abstraction over Anthropic and OpenAI LLM APIs.

The active provider is selected by the LLM_PROVIDER environment variable
(or overridden per-request). Both clients are instantiated lazily.
"""

import anthropic
import openai

from config import settings


def _call_anthropic(system_prompt: str, user_message: str) -> str:
    """
    Send a request to the Anthropic Messages API.

    Args:
        system_prompt: The system instructions for the model.
        user_message: The user turn containing context and question.

    Returns:
        The model's response as a plain string.
    """
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def _call_openai(system_prompt: str, user_message: str) -> str:
    """
    Send a request to the OpenAI Chat Completions API.

    Args:
        system_prompt: The system instructions for the model.
        user_message: The user turn containing context and question.

    Returns:
        The model's response as a plain string.
    """
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


def generate(
    system_prompt: str,
    user_message: str,
    provider: str | None = None,
) -> str:
    """
    Generate a response using the configured (or overridden) LLM provider.

    Args:
        system_prompt: System instructions to pass to the model.
        user_message: User message (with retrieved context embedded).
        provider: Optional per-request override for settings.LLM_PROVIDER.
                  Must be 'anthropic' or 'openai' when supplied.

    Returns:
        The model's response text.

    Raises:
        ValueError: When an unknown provider string is supplied.
    """
    active = (provider or settings.LLM_PROVIDER).lower()

    if active == "anthropic":
        return _call_anthropic(system_prompt, user_message)
    elif active == "openai":
        return _call_openai(system_prompt, user_message)
    else:
        raise ValueError(
            f"Unknown LLM provider {active!r}. Must be 'anthropic' or 'openai'."
        )
