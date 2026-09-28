"""Adapter interface for LLM chat/tool-calling backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the LLM.

    Attributes:
        id: Provider-assigned id, must be echoed back with the tool result.
        name: Name of the tool to call.
        arguments: Parsed keyword arguments for the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the provider for one or more LLM calls.

    Attributes:
        prompt_tokens: Input tokens, including cached ones.
        completion_tokens: Output tokens.
        cached_prompt_tokens: Subset of `prompt_tokens` served from the
            provider's prompt cache (billed at a lower rate).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result of one LLM turn.

    Attributes:
        content: Final assistant text, if the LLM produced a direct answer.
        tool_calls: Tools the LLM wants called before it can answer. Empty
            when `content` is the final answer.
        usage: Token counts for this call.
        latency_ms: Wall-clock time of the call, including retries.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_ms: float = 0.0


class LLMClient(ABC):
    """Common interface for any chat/tool-calling LLM backend.

    Hides the concrete SDK (OpenAI, or others added later) behind one
    method, so `ConversationAgent` never imports a provider SDK directly.
    """

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        """Run one chat turn.

        Args:
            messages: Conversation so far, in OpenAI chat message format
                (`role`/`content`, plus `tool_calls`/`tool_call_id` for
                tool turns).
            tools: Tool specs in OpenAI function-calling format, as
                produced by `ToolRegistry.openai_tool_specs`.

        Returns:
            The LLM's response, normalized to `LLMResponse`.
        """
        raise NotImplementedError
