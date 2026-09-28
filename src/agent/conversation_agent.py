"""Facade over the tool-calling loop: the assistant's single entry point."""

from __future__ import annotations

import json
import time
from typing import Any

from llm.base import LLMClient, TokenUsage
from tools.registry import ToolRegistry
from agent.prompt import SYSTEM_PROMPT, TOOL_BUDGET_FALLBACK_MESSAGE, USER_ID_INSTRUCTION_TEMPLATE


class ConversationAgent:
    """Runs the LLM <-> tool dispatch loop for a single user turn.

    Args:
        llm_client: LLM backend used for reasoning and tool selection.
        tool_registry: Tools the LLM is allowed to call.
        max_tool_turns: Safety cap on tool-calling round trips per message.
    """

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry, max_tool_turns: int = 6) -> None:
        self._llm = llm_client
        self._tools = tool_registry
        self._max_tool_turns = max_tool_turns

    def initial_messages(self, user_id: int | None) -> list[dict[str, Any]]:
        """Build the starting message history for a new session.

        Args:
            user_id: The identified user, if any, embedded in the system
                prompt so the LLM knows whose ratings to use by default.

        Returns:
            A one-message history containing the system prompt.
        """
        content = SYSTEM_PROMPT
        if user_id is not None:
            content += USER_ID_INSTRUCTION_TEMPLATE.format(user_id=user_id)
        return [{"role": "system", "content": content}]

    def handle_message(self, history: list[dict[str, Any]], message: str) -> dict[str, Any]:
        """Append a user message and run the tool-calling loop to a final answer.

        Args:
            history: Prior conversation, starting with the system prompt
                (see `initial_messages`).
            message: The user's new message.

        Returns:
            Dict with `reply` (final assistant text), `tool_trace` (ordered
            list of `{tool, arguments, result, latency_ms}` for
            transparency), `history` (updated message list, to persist for
            the next turn), and `metrics` (token usage and latency summed
            over every LLM/tool call in this turn).
        """
        messages = [*history, {"role": "user", "content": message}]
        tool_specs = self._tools.openai_tool_specs()
        trace: list[dict[str, Any]] = []
        usage = TokenUsage()
        llm_calls = 0
        llm_latency_ms = 0.0
        tool_latency_ms = 0.0

        def result(reply: str | None) -> dict[str, Any]:
            metrics = {
                "llm_calls": llm_calls,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "total_tokens": usage.total_tokens,
                "llm_latency_ms": round(llm_latency_ms, 1),
                "tool_latency_ms": round(tool_latency_ms, 1),
            }
            return {"reply": reply, "tool_trace": trace, "history": messages, "metrics": metrics, "usage": usage}

        for _ in range(self._max_tool_turns):
            response = self._llm.chat(messages, tool_specs)
            llm_calls += 1
            usage = usage + response.usage
            llm_latency_ms += response.latency_ms

            if not response.tool_calls:
                messages.append({"role": "assistant", "content": response.content})
                return result(response.content)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            for tool_call in response.tool_calls:
                started = time.perf_counter()
                tool_result = self._tools.dispatch(tool_call.name, tool_call.arguments)
                elapsed_ms = (time.perf_counter() - started) * 1000
                tool_latency_ms += elapsed_ms
                trace.append(
                    {
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": tool_result,
                        "latency_ms": round(elapsed_ms, 1),
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result)}
                )

        messages.append({"role": "assistant", "content": TOOL_BUDGET_FALLBACK_MESSAGE})
        return result(TOOL_BUDGET_FALLBACK_MESSAGE)
