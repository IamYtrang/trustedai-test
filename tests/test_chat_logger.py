"""Tests for `ChatLogger`."""

from __future__ import annotations

import json
from pathlib import Path

from agent.chat_logger import ChatLogger


class TestChatLogger:
    def test_log_turn_appends_one_json_line_per_call(self, tmp_path: Path) -> None:
        logger = ChatLogger(tmp_path)
        logger.log_turn({"turn_id": "a", "message": "hi"})
        logger.log_turn({"turn_id": "b", "message": "chào bạn"})

        files = list(tmp_path.glob("chat-*.jsonl"))
        assert len(files) == 1

        lines = files[0].read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert records[0]["turn_id"] == "a"
        assert records[1]["message"] == "chào bạn"
        assert "chào bạn" in lines[1]  # not escaped to \uXXXX
        assert "timestamp" in records[0]

    def test_creates_logs_dir_if_missing(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "nested" / "chat_logs"
        logger = ChatLogger(logs_dir)
        logger.log_turn({"turn_id": "a"})
        assert logs_dir.is_dir()
