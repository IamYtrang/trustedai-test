"""Tests for `estimate_cost_usd`."""

from __future__ import annotations

from llm.base import TokenUsage
from llm.pricing import ModelPricing, estimate_cost_usd


class TestEstimateCostUsd:
    def test_returns_none_without_pricing(self) -> None:
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        assert estimate_cost_usd(usage, None) is None

    def test_prices_uncached_and_output_tokens(self) -> None:
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        pricing = ModelPricing(input_per_1m=0.15, output_per_1m=0.60)
        assert estimate_cost_usd(usage, pricing) == 0.75

    def test_discounts_cached_prompt_tokens(self) -> None:
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, cached_prompt_tokens=1_000_000)
        pricing = ModelPricing(input_per_1m=0.15, output_per_1m=0.60, cached_input_per_1m=0.075)
        assert estimate_cost_usd(usage, pricing) == 0.075

    def test_falls_back_to_input_price_when_no_cached_price(self) -> None:
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=0, cached_prompt_tokens=1_000_000)
        pricing = ModelPricing(input_per_1m=0.15, output_per_1m=0.60)
        assert estimate_cost_usd(usage, pricing) == 0.15
