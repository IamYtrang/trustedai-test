"""Command pattern: every agent capability is a self-describing tool."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Base class for a single callable agent capability.

    Subclasses set `name`, `description`, and `parameters_schema` as class
    attributes and implement `run`.
    """

    name: str = ""
    description: str = ""
    # JSON Schema object describing the keyword arguments accepted by `run`.
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool.

        Args:
            **kwargs: Arguments matching `parameters_schema`, as decided by
                the LLM's tool call.

        Returns:
            A JSON-serializable dict describing the result. Should include
            an `"error"` key with a human-readable message instead of
            raising, so the agent loop can feed the failure back to the LLM.
        """
        raise NotImplementedError

    def to_openai_tool_spec(self) -> dict[str, Any]:
        """Describe this tool in the OpenAI function-calling format.

        Returns:
            A `{"type": "function", "function": {...}}` spec.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
