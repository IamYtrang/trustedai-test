"""OpenAI implementation of `LLMClient`."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm.base import LLMClient, LLMResponse, TokenUsage, ToolCall

logger = logging.getLogger(__name__)

# Only retry on transient failures (rate limits, timeouts, network issues,
# 5xx). Anything else (bad request, auth, content filter) is a real error
# that retrying won't fix.
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


class OpenAIClient(LLMClient):
    """Chat/tool-calling backed by the OpenAI Chat Completions API.

    Args:
        api_key: OpenAI API key.
        model: Chat model name, e.g. "gpt-4o-mini".
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """See `LLMClient.chat`. Retries transient API errors with backoff."""
        started = time.perf_counter()
        response = self._create(messages, tools)
        latency_ms = (time.perf_counter() - started) * 1000
        message = response.choices[0].message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (message.tool_calls or [])
        ]
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            usage=_usage_from(response),
            latency_ms=round(latency_ms, 1),
        )

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
        )


def _usage_from(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_prompt_tokens=cached,
    )
