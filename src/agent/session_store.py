"""In-memory conversation history store, keyed by session id."""

from __future__ import annotations

from typing import Any


class SessionStore:
    """Keeps each session's message history in memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, Any]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Fetch a session's message history.

        Args:
            session_id: Session identifier.

        Returns:
            The stored history, or `None` if the session doesn't exist yet.
        """
        return self._sessions.get(session_id)

    def save_history(self, session_id: str, history: list[dict[str, Any]]) -> None:
        """Persist a session's updated message history.

        Args:
            session_id: Session identifier.
            history: Full message history to store, replacing any prior value.
        """
        self._sessions[session_id] = history
