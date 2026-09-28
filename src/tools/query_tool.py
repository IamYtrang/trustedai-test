"""General-purpose data-access tool: the agent writes SQL instead of the
codebase pre-defining every possible read/aggregate/join.

"""

from __future__ import annotations

from typing import Any

from agent.prompt import QUERY_DATASET_TOOL_DESCRIPTION
from data.sql_store import SqlDataStore
from tools.base import BaseTool


class QueryDatasetTool(BaseTool):
    """Runs a read-only SQL query against the movies/ratings/tags dataset."""

    name = "query_dataset"
    description = QUERY_DATASET_TOOL_DESCRIPTION
    parameters_schema = {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "A single read-only SQL statement."}},
        "required": ["sql"],
    }

    def __init__(self, sql_store: SqlDataStore) -> None:
        self._sql_store = sql_store

    def run(self, **kwargs: Any) -> dict[str, Any]:
        sql = kwargs["sql"].strip()
        if not sql.lower().lstrip("(").startswith(("select", "with")):
            return {"error": "Only read-only SELECT (or WITH ... SELECT) statements are allowed."}
        return self._sql_store.execute_readonly_query(sql)
