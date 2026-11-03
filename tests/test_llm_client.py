import pytest

from config import settings
from generation import llm_client


@pytest.fixture(autouse=True)
def _clear_client_cache():
    llm_client._anthropic_client.cache_clear()
    llm_client._openai_client.cache_clear()
    yield
    llm_client._anthropic_client.cache_clear()
    llm_client._openai_client.cache_clear()


@pytest.mark.parametrize("provider, key", [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_missing_key_raises_config_error(monkeypatch, provider, key):
    monkeypatch.setattr(settings, key, "")
    with pytest.raises(llm_client.LLMConfigError, match=key):
        llm_client.generate("sys", "user", provider=provider)


def test_unknown_provider():
    with pytest.raises(ValueError):
        llm_client.generate("sys", "user", provider="gemini")
