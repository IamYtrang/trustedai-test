"""Factory for building the configured `LLMClient` without leaking SDK imports."""

from __future__ import annotations

from llm.base import LLMClient
from llm.openai_client import OpenAIClient


class LLMClientFactory:
    """Builds the `LLMClient` implementation selected by configuration."""

    @staticmethod
    def create(provider: str, api_key: str, model: str) -> LLMClient:
        """Build an `LLMClient` for the given provider.

        Args:
            provider: Provider identifier. Currently only `"openai"` is
                implemented; new providers plug in here without callers
                changing.
            api_key: API key for the provider.
            model: Model name to use.

        Returns:
            A concrete `LLMClient`.

        Raises:
            ValueError: If `provider` is not supported.
        """
        if provider == "openai":
            return OpenAIClient(api_key=api_key, model=model)
        raise ValueError(f"Unsupported LLM provider: {provider!r}")
