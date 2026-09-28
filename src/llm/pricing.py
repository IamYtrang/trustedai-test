"""Per-token cost estimation for LLM usage."""

from __future__ import annotations

from dataclasses import dataclass

from llm.base import TokenUsage

_TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1M tokens for one model.

    Attributes:
        input_per_1m: Price of uncached input tokens.
        output_per_1m: Price of output tokens.
        cached_input_per_1m: Price of cached input tokens. Falls back to
            `input_per_1m` when the provider doesn't discount cache hits.
    """

    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None


def estimate_cost_usd(usage: TokenUsage, pricing: ModelPricing | None) -> float | None:
    """Estimate the USD cost of the given token usage.

    Args:
        usage: Token counts to price.
        pricing: Model prices, or `None` if not configured.

    Returns:
        Estimated cost in USD, or `None` when pricing is unknown (so callers
        show "unknown" rather than a misleading $0).
    """
    if pricing is None:
        return None
    cached_price = pricing.cached_input_per_1m if pricing.cached_input_per_1m is not None else pricing.input_per_1m
    uncached_prompt = max(usage.prompt_tokens - usage.cached_prompt_tokens, 0)
    cost = (
        uncached_prompt * pricing.input_per_1m
        + usage.cached_prompt_tokens * cached_price
        + usage.completion_tokens * pricing.output_per_1m
    ) / _TOKENS_PER_UNIT
    return round(cost, 8)
