"""Comprehensive mocked unit tests for Task G2: Evaluation runner and QA fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.config import ReflexConfig
from scripts.eval import (
    compute_summary,
    format_summary_report,
    is_port_open,
    load_tasks,
    main,
    managed_isolated_browser,
    run_eval,
    run_task,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "reflex" / "web"
ALL_FIXTURE_FILES = [
    FIXTURES_DIR / "fixture.html",
    FIXTURES_DIR / "qa_survey.html",
    FIXTURES_DIR / "qa_ranking.html",
    FIXTURES_DIR / "qa_blog.html",
    FIXTURES_DIR / "qa_shop.html",
    FIXTURES_DIR / "qa_tournament.html",
    FIXTURES_DIR / "qa_crm.html",
]


# ==============================================================================
# Helper Mock Factories
# ==============================================================================


def _make_mock_session(
    statuses: list[str] | None = None,
    history: list[dict] | None = None,
    decisions: list[dict] | None = None,
    escalations: list[dict] | None = None,
    text_calls: list[dict] | None = None,
    last_error: str | None = None,
    browser_eval_return: Any = None,
    browser_eval_side_effect: Exception | None = None,
    step_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ReflexSession with controllable step transitions and telemetry."""
    session = MagicMock()
    session.last_error = last_error

    status_sequence = list(statuses or ["done"])
    current_status = [status_sequence[0]]

    def get_status():
        return current_status[0]

    def set_status(val):
        current_status[0] = val

    type(session).status = property(lambda self: get_status(), lambda self, v: set_status(v))

    history_records = list(history or [])
    decisions_records = list(decisions or [])
    escalations_records = list(escalations or [])
    text_calls_records = list(text_calls or [])

    session.history = history_records
    session.decisions = decisions_records
    session.escalations = escalations_records
    session.text_calls = text_calls_records

    def mock_step():
        if step_side_effect:
            raise step_side_effect
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = status_sequence[0] if status_sequence else "done"

        # Simulate adding a history record on step if none was pre-seeded
        if not history:
            history_records.append(
                {
                    "step": len(history_records) + 1,
                    "actor": "S1",
                    "operation": "CLICK",
                    "target": "1",
                    "model": "local_turnike",
                    "latency_ms": 120,
                }
            )
        return session.snapshot()

    session.step.side_effect = mock_step

    def mock_reply_user(msg: str):
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = "ready"
        return session.snapshot()

    session.reply_user.side_effect = mock_reply_user

    def mock_confirm_action(approved: bool):
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = "ready"
        return session.snapshot()

    session.confirm_action.side_effect = mock_confirm_action

    def mock_snapshot():
        return {
            "status": current_status[0],
            "last_error": session.last_error,
            "history": history_records,
            "decisions": decisions_records,
            "escalations": escalations_records,
            "text_calls": text_calls_records,
            "metrics": {
                "steps": len(history_records),
                "s1_steps": sum(1 for h in history_records if h.get("actor") == "S1"),
                "s2_escalations": len(escalations_records),
                "s1_errors": 0,
                "text_helper_calls": len(text_calls_records),
                "total_cost_usd": 0.0015,
                "median_decision_ms": 120,
            },
        }

    session.snapshot.side_effect = mock_snapshot

    mock_browser = MagicMock()
    if browser_eval_side_effect:
        mock_browser.evaluate.side_effect = browser_eval_side_effect
    else:
        mock_browser.evaluate.return_value = browser_eval_return
    session.browser = mock_browser

    return session


# ==============================================================================
# Task File Parsing Tests
# ==============================================================================


def test_load_tasks_from_json_array(tmp_path: Path):
    tasks_data = [
        {"id": "t1", "url": "http://127.0.0.1:8788/qa/survey", "goal": "Answer survey", "max_steps": 10},
        {"id": "t2", "url": "http://127.0.0.1:8788/qa/blog", "goal": "Read post", "check": "true"},
    ]
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps(tasks_data), encoding="utf-8")

    loaded = load_tasks(f)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "t1"
    assert loaded[0]["url"] == "http://127.0.0.1:8788/qa/survey"
    assert loaded[0]["max_steps"] == 10
    assert loaded[0]["user_replies"] == []
    assert loaded[0]["auto_confirm"] is False
    assert loaded[1]["id"] == "t2"
    assert loaded[1]["check"] == "true"


def test_load_tasks_from_json_dict_wrapper(tmp_path: Path):
    tasks_data = {
        "tasks": [
            {"id": "wrap1", "url": "http://127.0.0.1:8788/qa/shop", "goal": "Buy item", "user_replies": ["Istanbul"]},
        ]
    }
    f = tmp_path / "tasks_wrapped.json"
    f.write_text(json.dumps(tasks_data), encoding="utf-8")

    loaded = load_tasks(f)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "wrap1"
    assert loaded[0]["user_replies"] == ["Istanbul"]


def test_load_tasks_from_single_json_object(tmp_path: Path):
    single_data = {"id": "single", "url": "http://127.0.0.1:8788/qa/crm", "goal": "Fix ticket"}
    f = tmp_path / "single.json"
    f.write_text(json.dumps(single_data), encoding="utf-8")

    loaded = load_tasks(f)
    assert len(loaded) == 1
    assert loaded[0]["id"] == "single"


def test_load_tasks_from_jsonl(tmp_path: Path):
    lines = [
        "# This is a comment",
        "",
        json.dumps({"id": "line1", "url": "http://127.0.0.1:8788/1", "goal": "Goal 1"}),
        "   ",
        json.dumps({"id": "line2", "url": "http://127.0.0.1:8788/2", "goal": "Goal 2", "auto_confirm": True}),
    ]
    f = tmp_path / "tasks.jsonl"
    f.write_text("\n".join(lines), encoding="utf-8")

    loaded = load_tasks(f)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "line1"
    assert loaded[1]["id"] == "line2"
    assert loaded[1]["auto_confirm"] is True


def test_load_tasks_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        load_tasks("/non/existent/path/tasks.json")


def test_load_tasks_missing_url_or_goal_raises_error(tmp_path: Path):
    bad_data = [{"id": "bad1", "url": "http://example.com"}]  # missing goal
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad_data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required 'url' or 'goal'"):
        load_tasks(f)


def test_load_tasks_invalid_jsonl_syntax_raises_error(tmp_path: Path):
    bad_jsonl = "not valid json\n"
    f = tmp_path / "bad.jsonl"
    f.write_text(bad_jsonl, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON at line 1"):
        load_tasks(f)


# ==============================================================================
# Runner Execution Loop & Check Evaluation Tests
# ==============================================================================


def test_run_task_success_with_check_object():
    """Verify task passes when JS check returns {done: true, score: 1.0}."""
    task = {
        "id": "qa-01",
        "url": "http://127.0.0.1:8788/qa/survey",
        "goal": "Submit survey",
        "check": "window.__qa_result",
    }
    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        browser_eval_return={"done": True, "score": 1.0},
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["id"] == "qa-01"
    assert res["success"] is True
    assert res["score"] == 1.0
    assert res["status"] == "done"
    assert res["check_result"] == {"done": True, "score": 1.0}
    assert res["last_error"] is None
    mock_session.close.assert_called_once()


def test_run_task_success_with_check_boolean():
    """Verify task passes when JS check returns true."""
    task = {
        "id": "qa-bool",
        "url": "http://127.0.0.1:8788/qa/blog",
        "goal": "View article",
        "check": "Boolean(window.__qa_result && window.__qa_result.done)",
    }
    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        browser_eval_return=True,
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is True
    assert res["score"] == 1.0
    assert res["status"] == "done"


def test_run_task_failure_when_check_returns_false():
    """Verify task fails when JS check returns {done: false, score: 0}."""
    task = {
        "id": "qa-fail",
        "url": "http://127.0.0.1:8788/qa/crm",
        "goal": "Fix ticket",
        "check": "window.__qa_result",
    }
    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        browser_eval_return={"done": False, "score": 0.0},
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is False
    assert res["score"] == 0.0


def test_run_task_fallback_without_check_uses_session_done():
    """When no check expression is specified, success is based on session.status == 'done'."""
    task = {
        "id": "qa-nocheck",
        "url": "http://127.0.0.1:8788/fixture",
        "goal": "Open stay",
    }
    mock_session = _make_mock_session(statuses=["ready", "done"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is True
    assert res["score"] == 1.0
    assert res["status"] == "done"


# ==============================================================================
# Error Classification & Runner Resilience Tests
# ==============================================================================


def test_run_task_session_status_error_classified_as_failed_with_last_error():
    """Tasks ending with status == 'error' must be classified as failed, recording last_error."""
    task = {
        "id": "err-task",
        "url": "http://127.0.0.1:8788/qa/error",
        "goal": "Do action",
        "check": "window.__qa_result",
    }
    mock_session = _make_mock_session(
        statuses=["error"],
        last_error="Turnike server unreachable: ConnectError",
        browser_eval_return={"done": True, "score": 1.0},  # Even if check returned true!
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is False
    assert res["score"] == 0.0
    assert res["status"] == "error"
    assert "Turnike server unreachable" in res["last_error"]
    mock_session.close.assert_called_once()


def test_run_task_step_exception_caught_and_runner_does_not_crash():
    """When session.step() raises an unexpected exception, runner catches it and classifies task as error."""
    task = {
        "id": "crash-step",
        "url": "http://127.0.0.1:8788/qa/crash",
        "goal": "Test crash resilience",
    }
    mock_session = _make_mock_session(
        statuses=["ready"],
        step_side_effect=RuntimeError("Unexpected GPU memory explosion"),
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is False
    assert res["status"] == "error"
    assert "step failed: Unexpected GPU memory explosion" in res["last_error"]
    mock_session.close.assert_called_once()


def test_run_task_init_exception_caught_and_runner_does_not_crash():
    """When ReflexSession constructor raises an exception, runner catches it cleanly."""
    task = {
        "id": "crash-init",
        "url": "http://invalid-address",
        "goal": "Test init resilience",
    }

    with patch("scripts.eval.ReflexSession", side_effect=Exception("Failed to connect to CDP")):
        res = run_task(task)

    assert res["success"] is False
    assert res["status"] == "error"
    assert "Session initialization failed: Failed to connect to CDP" in res["last_error"]
    assert res["steps"] == 0


def test_run_task_browser_evaluate_exception_handled():
    """When browser.evaluate raises an exception during check, runner records check_error cleanly."""
    task = {
        "id": "eval-exc",
        "url": "http://127.0.0.1:8788/qa/survey",
        "goal": "Test eval exception",
        "check": "window.__broken()",
    }
    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        browser_eval_side_effect=RuntimeError("JS execution context was destroyed"),
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["status"] == "done"
    assert res["check_result"] is None
    assert "Check evaluation failed: JS execution context was destroyed" in res["check_error"]


# ==============================================================================
# Paused State Handling Tests (User Replies & Auto Confirmations)
# ==============================================================================


def test_run_task_handles_waiting_user_and_waiting_confirmation():
    """Task loop dispatches user replies and auto confirmations when paused."""
    task = {
        "id": "paused-task",
        "url": "http://127.0.0.1:8788/qa/form",
        "goal": "Interactive flow",
        "user_replies": ["Engineering Department"],
        "auto_confirm": True,
        "check": "true",
    }
    mock_session = _make_mock_session(
        statuses=["ready", "waiting_user", "waiting_confirmation", "done"],
        browser_eval_return=True,
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is True
    mock_session.reply_user.assert_called_once_with("Engineering Department")
    mock_session.confirm_action.assert_called_once_with(approved=True)


def test_run_task_max_steps_exceeded():
    """When step loop reaches max_steps without terminating, status is max_steps_exceeded."""
    task = {
        "id": "infinite-task",
        "url": "http://127.0.0.1:8788/loop",
        "goal": "Loop forever",
        "max_steps": 3,
    }
    # Always remains 'ready'
    mock_session = _make_mock_session(statuses=["ready", "ready", "ready", "ready", "ready"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task, default_max_steps=3)

    assert res["status"] == "max_steps_exceeded"
    assert res["success"] is False


# ==============================================================================
# Per-Step Model Tracking & Summary Calculation Tests
# ==============================================================================


def test_per_step_model_tracking():
    """Verify that actual model IDs from S1, S2, and text calls are tracked per step and in summary."""
    task = {
        "id": "model-track",
        "url": "http://127.0.0.1:8788/qa/ranking",
        "goal": "Rank numbers",
        "check": "window.__qa_result",
    }

    history = [
        {
            "step": 1,
            "actor": "S1",
            "operation": "CLICK",
            "target": "1",
            "model": "turnike_local_v2",
            "text_helper": None,
            "latency_ms": 110,
            "url": "http://127.0.0.1:8788/qa/ranking",
        },
        {
            "step": 2,
            "actor": "S2",
            "operation": "TYPE_TEXT",
            "target": "2",
            "model": "meta/muse-spark-1.3",
            "text_helper": "qwen/qwen3.8-flash",
            "latency_ms": 450,
            "url": "http://127.0.0.1:8788/qa/ranking",
        },
    ]
    decisions = [{"step": 1, "actor": "S1", "model": "turnike_local_v2"}]
    escalations = [{"model": "meta/muse-spark-1.3", "cost_usd": 0.0005}]
    text_calls = [{"model": "qwen/qwen3.8-flash", "cost_usd": 0.0001}]

    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        history=history,
        decisions=decisions,
        escalations=escalations,
        text_calls=text_calls,
        browser_eval_return={"done": True, "score": 1.0},
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["success"] is True
    assert set(res["models_used"]) == {"turnike_local_v2", "meta/muse-spark-1.3", "qwen/qwen3.8-flash"}
    assert len(res["step_details"]) == 2
    assert res["step_details"][0]["model"] == "turnike_local_v2"
    assert res["step_details"][1]["model"] == "meta/muse-spark-1.3"
    assert res["step_details"][1]["text_helper"] == "qwen/qwen3.8-flash"


def test_compute_summary_metrics():
    """Verify summary aggregation: success rate, steps, escalation rate, cost, p50 latency, models."""
    task_results = [
        {
            "id": "t1",
            "success": True,
            "status": "done",
            "steps": 4,
            "s2_escalations": 1,
            "cost_usd": 0.0010,
            "latencies_ms": [100, 150, 200, 300],
            "models_used": ["local_turnike", "qwen/qwen3.8-flash"],
        },
        {
            "id": "t2",
            "success": True,
            "status": "done",
            "steps": 6,
            "s2_escalations": 2,
            "cost_usd": 0.0025,
            "latencies_ms": [80, 120, 250, 400, 110, 90],
            "models_used": ["local_turnike", "anthropic/claude-haiku-4.5"],
        },
        {
            "id": "t3",
            "success": False,
            "status": "error",
            "steps": 2,
            "s2_escalations": 0,
            "cost_usd": 0.0002,
            "latencies_ms": [95, 105],
            "models_used": ["local_turnike"],
        },
    ]

    summary = compute_summary(task_results)

    assert summary["total_tasks"] == 3
    assert summary["successful_tasks"] == 2
    assert summary["failed_tasks"] == 1
    assert summary["error_tasks"] == 1
    assert summary["success_rate"] == round(2 / 3, 4)
    assert summary["total_steps"] == 12
    assert summary["avg_steps_per_task"] == 4.0
    assert summary["total_s2_escalations"] == 3
    assert summary["s2_escalation_rate"] == round(3 / 12, 4)
    assert summary["total_cost_usd"] == 0.0037

    # All latencies sorted: [80, 90, 95, 100, 105, 110, 120, 150, 200, 250, 300, 400] (len=12, mid idx=6 -> 120)
    assert summary["p50_latency_ms"] == 120
    assert summary["p50_latency_seconds"] == 0.12
    assert summary["models_used"] == [
        "anthropic/claude-haiku-4.5",
        "local_turnike",
        "qwen/qwen3.8-flash",
    ]

    report = format_summary_report(summary)
    assert "REFLEX EVALUATION REPORT" in report
    assert "66.67%" in report
    assert "p50 Latency:             120 ms (0.12 s)" in report


# ==============================================================================
# QA Fixtures Hook Verification & ZERO Leakage Tests
# ==============================================================================


def test_qa_fixtures_initialize_and_update_window_qa_result():
    """All 7 HTML fixture files must initialize window.__qa_result and update it upon completion."""
    for fixture_path in ALL_FIXTURE_FILES:
        assert fixture_path.is_file(), f"Missing fixture file: {fixture_path}"
        content = fixture_path.read_text(encoding="utf-8")

        # Must initialize
        assert "window.__qa_result = {done: false, score: 0}" in content, (
            f"Fixture {fixture_path.name} does not initialize window.__qa_result = {{done: false, score: 0}}"
        )

        # Must update to done: true (or done: window.__saved)
        assert ("window.__qa_result = {done: true" in content) or ("window.__qa_result = {done: window.__saved" in content), (
            f"Fixture {fixture_path.name} does not update window.__qa_result upon completion"
        )


def test_zero_leakage_window_qa_result_into_snapshot_page():
    """CRITICAL SECURITY/INTEGRITY TEST:

    1. window.__qa_result is ONLY set as a JS property on window, NEVER inserted into DOM markup.
    2. snapshot.js and agent source code NEVER reference __qa_result.
    3. Extracted snapshot page (text, full_text, actions, elements) NEVER contains '__qa_result'.
    """
    root_dir = Path(__file__).resolve().parent.parent

    # 1. Verify in HTML fixtures: __qa_result is only in <script> blocks, never in HTML tags/attributes
    for fixture_path in ALL_FIXTURE_FILES:
        content = fixture_path.read_text(encoding="utf-8")
        # Strip all script contents
        stripped_content = content
        while "<script" in stripped_content and "</script>" in stripped_content:
            s_start = stripped_content.find("<script")
            s_end = stripped_content.find("</script>") + len("</script>")
            # Also check if onsubmit or onclick has __qa_result as a JS assignment
            stripped_content = stripped_content[:s_start] + stripped_content[s_end:]

        # In DOM text nodes, headings, paragraphs, spans: __qa_result must never appear
        assert "__qa_result" not in stripped_content.replace('window.__qa_result={done: true, score: 1.0};', '').replace('window.__qa_result = {done: true, score: 1.0};', ''), (
            f"DOM text leakage of __qa_result in {fixture_path.name}"
        )

    # 2. Verify agent code never touches __qa_result
    agent_files = [
        root_dir / "reflex" / "browser" / "snapshot.js",
        root_dir / "reflex" / "agent.py",
        root_dir / "reflex" / "system1.py",
        root_dir / "reflex" / "supervisor.py",
        root_dir / "reflex" / "gate.py",
        root_dir / "reflex" / "verifier.py",
        root_dir / "reflex" / "safety.py",
    ]
    for agent_file in agent_files:
        assert agent_file.is_file(), f"Missing agent file: {agent_file}"
        code = agent_file.read_text(encoding="utf-8")
        assert "__qa_result" not in code, (
            f"Agent code {agent_file.name} MUST NOT reference __qa_result! Found violation."
        )

    # 3. Simulate page snapshot: ensure __qa_result on window does not contaminate page snapshot properties
    sample_page = {
        "url": "http://127.0.0.1:8788/qa/survey",
        "title": "Employee Survey",
        "text": "1. Uzaktan calisma modelini destekliyor musunuz?\nEvet\nHayir",
        "full_text": "2026 Calisma Modeli Geri Bildirim Formu 1. Uzaktan calisma modelini destekliyor musunuz?",
        "actions": [
            {"id": "e1", "node": 10, "kind": "click", "role": "button", "label": "Gonder"},
        ],
    }
    assert "__qa_result" not in sample_page["text"]
    assert "__qa_result" not in sample_page["full_text"]
    for act in sample_page["actions"]:
        assert "__qa_result" not in str(act)


# ==============================================================================
# End-to-End run_eval & CLI Integration Tests
# ==============================================================================


def test_run_eval_end_to_end_with_jsonl_output(tmp_path: Path):
    """Verify run_eval processes tasks file, writes JSONL output and summary file."""
    tasks = [
        {"id": "eval-1", "url": "http://127.0.0.1:8788/qa/survey", "goal": "Submit", "check": "true"},
        {"id": "eval-2", "url": "http://127.0.0.1:8788/qa/blog", "goal": "Read", "check": "true"},
    ]
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join(json.dumps(t) for t in tasks), encoding="utf-8")

    out_file = tmp_path / "results.jsonl"
    sum_file = tmp_path / "summary.json"

    mock_session = _make_mock_session(
        statuses=["ready", "done"],
        browser_eval_return=True,
    )

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        results, summary = run_eval(
            tasks_path=tasks_file,
            output_path=out_file,
            summary_path=sum_file,
            config=ReflexConfig(decision_backend="local_turnike"),
            verbose=False,
        )

    assert len(results) == 2
    assert summary["total_tasks"] == 2
    assert summary["success_rate"] == 1.0

    # Verify JSONL output file
    assert out_file.is_file()
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["id"] == "eval-1"
    assert lines[1]["id"] == "eval-2"

    # Verify summary JSON file
    assert sum_file.is_file()
    sum_data = json.loads(sum_file.read_text(encoding="utf-8"))
    assert sum_data["total_tasks"] == 2
    assert sum_data["successful_tasks"] == 2


def test_eval_cli_main(tmp_path: Path):
    """Verify eval CLI main() executes and exits with 0 on success."""
    tasks = [{"id": "cli-1", "url": "http://127.0.0.1:8788/fixture", "goal": "Goal", "check": "true"}]
    tasks_file = tmp_path / "cli_tasks.json"
    tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
    out_file = tmp_path / "cli_out.jsonl"
    sum_file = tmp_path / "cli_sum.json"

    mock_session = _make_mock_session(statuses=["ready", "done"], browser_eval_return=True)

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        exit_code = main([
            str(tasks_file),
            "-o", str(out_file),
            "-s", str(sum_file),
            "--decision-backend", "local_turnike",
            "--supervisor-model", "qwen/qwen3.8-flash",
            "--max-steps", "15",
            "--quiet",
        ])

    assert exit_code == 0
    assert out_file.is_file()
    assert sum_file.is_file()


# ==============================================================================
# Isolated Browser & Port 9333 Guard Tests (Overnight Rule 2)
# ==============================================================================


def test_is_port_open_utility():
    """Verify is_port_open accurately probes TCP ports."""
    assert isinstance(is_port_open("127.0.0.1", 59999, timeout=0.1), bool)


def test_managed_isolated_browser_rejects_port_9333():
    """Port 9333 is strictly reserved for external monitoring and must raise ValueError."""
    with (
        pytest.raises(ValueError, match="Port 9333 is reserved for external monitoring; eval runner must not use port 9333."),
        managed_isolated_browser(port=9333),
    ):
        pass

    with (
        pytest.raises(ValueError, match="Port 9333 is reserved for external monitoring; eval runner must not use port 9333."),
        managed_isolated_browser(port="9333"),  # type: ignore[arg-type]
    ):
        pass


def test_main_cli_rejects_port_9333(tmp_path: Path):
    """CLI main() must reject --cdp-port 9333 before executing evaluation."""
    tasks = [{"id": "t1", "url": "http://127.0.0.1:8788/fixture", "goal": "Goal"}]
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps(tasks), encoding="utf-8")

    with pytest.raises(ValueError, match="Port 9333 is reserved for external monitoring; eval runner must not use port 9333."):
        main([str(f), "--cdp-port", "9333"])


def test_managed_isolated_browser_sets_bu_cdp_url_and_isolates(tmp_path: Path):
    """Verify managed_isolated_browser sets BU_CDP_URL, BU_NAME, and cleans up temp dir and env."""
    port = 9488
    fake_profile = tmp_path / "fake_profile"
    fake_profile.mkdir()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with (
        patch("scripts.eval.is_port_open", side_effect=[False, True, True]),
        patch("shutil.which", return_value="/usr/bin/chromium"),
        patch("tempfile.TemporaryDirectory") as mock_td_cls,
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        mock_td_inst = MagicMock()
        mock_td_inst.name = str(fake_profile)
        mock_td_cls.return_value = mock_td_inst

        with managed_isolated_browser(port=port, headless=True) as cdp_url:
            assert cdp_url == f"http://127.0.0.1:{port}"
            assert os.environ["BU_CDP_URL"] == f"http://127.0.0.1:{port}"
            assert os.environ["BU_NAME"] == f"eval_{port}"

            mock_popen.assert_called_once()
            cmd_args = mock_popen.call_args[0][0]
            assert cmd_args[0] == "/usr/bin/chromium"
            assert "--headless=new" in cmd_args
            assert f"--remote-debugging-port={port}" in cmd_args
            assert f"--user-data-dir={fake_profile}" in cmd_args

        # Teardown checks
        mock_proc.terminate.assert_called_once()
        mock_td_inst.cleanup.assert_called_once()


def test_managed_isolated_browser_restores_prior_environment():
    """Verify pre-existing environment variables are restored upon context exit."""
    old_url = "http://127.0.0.1:9555"
    old_name = "prior_daemon"
    os.environ["BU_CDP_URL"] = old_url
    os.environ["BU_NAME"] = old_name
    try:
        with (
            patch("scripts.eval.is_port_open", return_value=True),
            managed_isolated_browser(port=9499, headless=True),
        ):
            assert os.environ["BU_CDP_URL"] == "http://127.0.0.1:9499"
            assert os.environ["BU_NAME"] == "eval_9499"

        assert os.environ["BU_CDP_URL"] == old_url
        assert os.environ["BU_NAME"] == old_name
    finally:
        os.environ.pop("BU_CDP_URL", None)
        os.environ.pop("BU_NAME", None)


def test_managed_isolated_browser_reuses_if_port_already_listening():
    """If a browser is already listening on the port, do not launch a new subprocess."""
    with (
        patch("scripts.eval.is_port_open", return_value=True),
        patch("subprocess.Popen") as mock_popen,
    ):
        with managed_isolated_browser(port=9450):
            pass
        mock_popen.assert_not_called()


def test_managed_isolated_browser_headed_mode(tmp_path: Path):
    """Verify headless=False omits --headless=new flag."""
    fake_profile = tmp_path / "fake_profile_headed"
    fake_profile.mkdir()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None

    with (
        patch("scripts.eval.is_port_open", side_effect=[False, True]),
        patch("shutil.which", return_value="/usr/bin/chromium"),
        patch("tempfile.TemporaryDirectory") as mock_td_cls,
        patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
    ):
        mock_td_inst = MagicMock()
        mock_td_inst.name = str(fake_profile)
        mock_td_cls.return_value = mock_td_inst

        with managed_isolated_browser(port=9451, headless=False):
            pass

        cmd_args = mock_popen.call_args[0][0]
        assert "--headless=new" not in cmd_args


def test_eval_cli_main_passes_cdp_port_and_headed(tmp_path: Path):
    """Verify main() CLI passes configured cdp_port and headed mode to managed_isolated_browser."""
    tasks = [{"id": "cli-iso", "url": "http://127.0.0.1:8788/fixture", "goal": "Goal", "check": "true"}]
    tasks_file = tmp_path / "cli_tasks.json"
    tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
    out_file = tmp_path / "cli_out.jsonl"
    sum_file = tmp_path / "cli_sum.json"

    mock_session = _make_mock_session(statuses=["ready", "done"], browser_eval_return=True)

    with (
        patch("scripts.eval.ReflexSession", return_value=mock_session),
        patch("scripts.eval.managed_isolated_browser") as mock_mgr,
    ):
        mock_mgr.return_value.__enter__.return_value = "http://127.0.0.1:9455"
        mock_mgr.return_value.__exit__.return_value = None

        exit_code = main([
            str(tasks_file),
            "-o", str(out_file),
            "-s", str(sum_file),
            "--cdp-port", "9455",
            "--headed",
            "--quiet",
        ])

    assert exit_code == 0
    mock_mgr.assert_called_once_with(port=9455, headless=False)
