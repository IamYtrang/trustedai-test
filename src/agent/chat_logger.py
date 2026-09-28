"""Append-only JSONL audit log of chat turns, one file per UTC day."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ChatLogger:
    """Persists one JSON record per chat turn under `logs_dir`.

    Args:
        logs_dir: Directory for `chat-YYYY-MM-DD.jsonl` files; created on
            first write.
    """

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._lock = threading.Lock()

    def log_turn(self, record: dict[str, Any]) -> Path:
        """Append a turn record, stamping it with the current UTC time.

        Args:
            record: JSON-serializable turn data (ids, message, reply,
                tool_trace, metrics, status, error).

        Returns:
            The file the record was written to.
        """
        now = datetime.now(timezone.utc)
        line = json.dumps({"timestamp": now.isoformat(), **record}, ensure_ascii=False, default=str)
        path = self._logs_dir / f"chat-{now:%Y-%m-%d}.jsonl"
        with self._lock:
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return path
