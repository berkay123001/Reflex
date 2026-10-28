"""Every eval task record carries the agent's last answer and the hosts it visited."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from reflex.config import ReflexConfig
from scripts.eval import run_task


class _Session:
    def __init__(self, url: str, goal: str, config: ReflexConfig) -> None:
        self.url = url
        self.goal = goal
        self.config = config
        self.status = "ready"
        self.last_error = None
        self.history: list[dict[str, Any]] = [
            {"url": "https://alpha.example/a"},
            {"url": "https://alpha.example/b"},
            {"url": "https://beta.example/c"},
            {"url": "https://alpha.example/d"},
            {"url": ""},
            {"url": "not a url"},
        ]
        self.browser = None
        self._snapshots = 0

    def step(self) -> dict[str, Any]:
        self.status = "done"
        return {}

    def snapshot(self) -> dict[str, Any]:
        self._snapshots += 1
        return {
            "status": self.status,
            "last_error": self.last_error,
            "history": list(self.history),
            "decisions": [],
            "escalations": [],
            "text_calls": [],
            "interpret_calls": [],
            "metrics": {"s1_steps": 1, "s1_errors": 0, "total_cost_usd": 0.0, "known_cost_usd": 0.0, "usage": {}},
            "context": {
                "chat_log": [
                    {"role": "user", "content": "do the task", "kind": "chat"},
                    {"role": "assistant", "content": "earlier note", "kind": "chat"},
                    {"role": "assistant", "content": "the answer", "kind": "escalation_note"},
                    {"role": "assistant", "content": "   ", "kind": "chat"},
                ]
            },
        }

    def close(self) -> None:
        return None


def test_success_record_has_last_assistant_message_and_hosts() -> None:
    with patch("scripts.eval.ReflexSession", _Session):
        record = run_task(
            {"id": "rec-1", "url": "https://alpha.example/start", "goal": "Finish the page"},
            config=ReflexConfig(decision_backend="local_turnike"),
        )
    assert record["success"] is True
    assert record["final_answer"] == "the answer"
    assert record["visited_hosts"] == ["alpha.example", "beta.example", "alpha.example"]


def test_init_failure_record_has_empty_answer_and_hosts() -> None:
    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("browser failed to start")

    with patch("scripts.eval.ReflexSession", explode):
        record = run_task(
            {"id": "rec-2", "url": "https://alpha.example/start", "goal": "Finish the page"},
            config=ReflexConfig(decision_backend="local_turnike"),
        )
    assert record["status"] == "error"
    assert record["success"] is False
    assert record["final_answer"] is None
    assert record["visited_hosts"] == []


def test_unexpected_error_record_still_reports_answer_and_hosts() -> None:
    class Broken(_Session):
        def snapshot(self) -> dict[str, Any]:
            self._snapshots += 1
            if self._snapshots == 1:
                raise RuntimeError("snapshot failed")
            return super().snapshot()

    with patch("scripts.eval.ReflexSession", Broken):
        record = run_task(
            {"id": "rec-3", "url": "https://alpha.example/start", "goal": "Finish the page"},
            config=ReflexConfig(decision_backend="local_turnike"),
        )
    assert record["status"] == "error"
    assert "Unexpected runner error" in str(record["last_error"])
    assert record["final_answer"] == "the answer"
    assert record["visited_hosts"] == ["alpha.example", "beta.example", "alpha.example"]
