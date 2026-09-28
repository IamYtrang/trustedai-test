"""Registry pattern: looks tools up by name so the agent loop stays generic."""

from __future__ import annotations

from tools.base import BaseTool


class ToolRegistry:
    """Collection of tools, dispatchable by the name the LLM requests.

    Args:
        tools: Tools to register. Names must be unique.
    """

    def __init__(self, tools: list[BaseTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("Tool names must be unique.")

    def openai_tool_specs(self) -> list[dict]:
        """Describe every registered tool for the LLM's tool-calling API.

        Returns:
            List of OpenAI function-calling tool specs.
        """
        return [tool.to_openai_tool_spec() for tool in self._tools.values()]

    def dispatch(self, name: str, arguments: dict) -> dict:
        """Run a tool by name.

        Args:
            name: Tool name, as requested by the LLM.
            arguments: Keyword arguments to pass to the tool's `run`.

        Returns:
            The tool's result dict, or `{"error": ...}` if the tool is
            unknown or raised an exception (callers should never need to
            catch exceptions from this method).
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return tool.run(**arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced to the LLM, not swallowed silently
            return {"error": f"Tool '{name}' failed: {exc}"}
