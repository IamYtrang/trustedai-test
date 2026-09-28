"""Pydantic request/response models for the chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message.

    Attributes:
        session_id: Client-generated id identifying the conversation. A new
            id starts a new session.
        user_id: The identified user. Only used when starting a new
            session (embedded into the system prompt); ignored on
            subsequent turns of the same session.
        message: The user's message text.
    """

    session_id: str
    user_id: int | None = None
    message: str


class ToolTraceEntry(BaseModel):
    """One tool call made while answering a message, for UI transparency.

    Attributes:
        tool: Tool name.
        arguments: Arguments the LLM passed to the tool.
        result: The tool's JSON result.
        latency_ms: Time spent executing the tool.
    """

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    latency_ms: float = 0.0


class TurnMetrics(BaseModel):
    """Cost and latency of answering one message.

    Attributes:
        model: LLM model name used.
        llm_calls: Number of LLM round trips (1 + one per tool-calling step).
        prompt_tokens: Input tokens across all LLM calls, including cached.
        completion_tokens: Output tokens across all LLM calls.
        cached_prompt_tokens: Portion of `prompt_tokens` served from cache.
        total_tokens: `prompt_tokens + completion_tokens`.
        llm_latency_ms: Time spent waiting on the LLM, including retries.
        tool_latency_ms: Time spent executing tools.
        total_latency_ms: End-to-end server time for the request.
        cost_usd: Estimated cost, or `None` if pricing isn't configured.
    """

    model: str
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    total_tokens: int
    llm_latency_ms: float
    tool_latency_ms: float
    total_latency_ms: float
    cost_usd: float | None


class ChatResponse(BaseModel):
    """Assistant's answer to a chat message.

    Attributes:
        turn_id: Unique id of this turn, matching its record in the chat log.
        reply: Final assistant text.
        tool_trace: Ordered list of tools used to ground the answer.
        metrics: Cost and latency of this turn.
    """

    turn_id: str
    reply: str
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    metrics: TurnMetrics


class UserSummary(BaseModel):
    """Short summary of a known user, for the UI's user picker.

    Attributes:
        user_id: MovieLens user identifier.
        num_ratings: Total ratings by this user.
        avg_rating: Mean rating given by this user.
    """

    user_id: int
    num_ratings: int
    avg_rating: float
