"""Central application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from llm.pricing import ModelPricing

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Application settings, overridable via environment variables or `.env`.

    Attributes:
        data_dir: Directory containing the MovieLens CSV files.
        embeddings_dir: Directory used to cache precomputed plot embeddings.
        embedding_model_name: HuggingFace model id used for plot/query
            embeddings, as a fallback if `local_embedding_model_dir` isn't
            present (e.g. a different model is configured).
        local_embedding_model_dir: A self-contained local copy of the
            embedding model, committed to the repo so a fresh clone works
            without internet access. Preferred over `embedding_model_name`
            whenever it exists — see `resolve_embedding_model_path`.
        openai_api_key: API key for the OpenAI LLM client. Required at
            runtime for the chat agent; not needed for evaluation scripts
            that only exercise the recommenders.
        openai_model: OpenAI chat model to use for tool-calling.
        max_agent_tool_turns: Safety cap on how many tool-calling round trips
            the agent will do for a single user message, to avoid infinite
            loops if the LLM keeps requesting tools.
        logs_dir: Directory for the per-day JSONL chat audit logs.
        llm_price_input_per_1m: USD per 1M uncached input tokens. Defaults
            match gpt-4o-mini list prices; override when using another model.
            Set to `none` in `.env` to disable cost estimation.
        llm_price_cached_input_per_1m: USD per 1M cached input tokens.
        llm_price_output_per_1m: USD per 1M output tokens.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_parse_none_str="none")

    data_dir: Path = REPO_ROOT / "data" / "ml-latest-small-filtered"
    embeddings_dir: Path = REPO_ROOT / "data" / "embeddings"
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    local_embedding_model_dir: Path = REPO_ROOT / "data" / "models" / "bge-small-en-v1.5"
    openai_api_key: str = ""
    openai_model: str = ""
    max_agent_tool_turns: int = 6
    logs_dir: Path = REPO_ROOT / "data" / "chat_logs"
    llm_price_input_per_1m: float | None = 0.15
    llm_price_cached_input_per_1m: float | None = 0.075
    llm_price_output_per_1m: float | None = 0.60

    def llm_pricing(self) -> ModelPricing | None:
        """Build the configured model pricing, or `None` if incomplete."""
        if self.llm_price_input_per_1m is None or self.llm_price_output_per_1m is None:
            return None
        return ModelPricing(
            input_per_1m=self.llm_price_input_per_1m,
            output_per_1m=self.llm_price_output_per_1m,
            cached_input_per_1m=self.llm_price_cached_input_per_1m,
        )

    def resolve_embedding_model_path(self) -> str:
        """Pick where to load the embedding model from.

        Returns:
            The local model directory as a string if it exists on disk
            (the common case — it's committed to the repo); otherwise
            `embedding_model_name`, which `sentence-transformers` downloads
            from the HuggingFace Hub on first use.
        """
        if self.local_embedding_model_dir.is_dir():
            return str(self.local_embedding_model_dir)
        return self.embedding_model_name


settings = Settings()
