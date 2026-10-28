"""Comprehensive unit tests for Milestone M2: Eval runner and dev_qa evaluation tasks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.eval import (
    compute_summary,
    format_summary_report,
    load_tasks,
    managed_isolated_browser,
    managed_web_server,
    run_eval,
    run_task,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEV_QA_PATH = ROOT_DIR / "scripts" / "eval_tasks" / "dev_qa.json"
WEB_DIR = ROOT_DIR / "reflex" / "web"


# ==============================================================================
# Helper Mock Factories
# ==============================================================================


def _make_mock_session(
    statuses: list[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    last_error: str | None = None,
    browser_eval_return: Any = True,
) -> MagicMock:
    """Create a mock ReflexSession with controllable step status transitions."""
    session = MagicMock()
    session.last_error = last_error

    status_sequence = list(statuses or ["done"])
    current_status = [status_sequence[0]]

    def get_status() -> str:
        return current_status[0]

    def set_status(val: str) -> None:
        current_status[0] = val

    type(session).status = property(lambda self: get_status(), lambda self, v: set_status(v))

    history_records = list(history or [])
    session.history = history_records
    session.decisions = []
    session.escalations = []
    session.text_calls = []

    def mock_step() -> dict[str, Any]:
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = status_sequence[0] if status_sequence else "done"

        history_records.append(
            {
                "step": len(history_records) + 1,
                "actor": "S1",
                "operation": "CLICK",
                "target": "1",
                "model": "local_turnike",
                "latency_ms": 100,
            }
        )
        return session.snapshot()

    session.step.side_effect = mock_step

    def mock_reply_user(msg: str) -> dict[str, Any]:
        if not msg:
            raise ValueError("Enter a non-empty message")
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = "ready"
        return session.snapshot()

    session.reply_user.side_effect = mock_reply_user

    def mock_confirm_action(approved: bool) -> dict[str, Any]:
        if len(status_sequence) > 1:
            status_sequence.pop(0)
            current_status[0] = status_sequence[0]
        else:
            current_status[0] = "ready"
        return session.snapshot()

    session.confirm_action.side_effect = mock_confirm_action

    def mock_snapshot() -> dict[str, Any]:
        return {
            "status": current_status[0],
            "last_error": session.last_error,
            "history": history_records,
            "decisions": [],
            "escalations": [],
            "text_calls": [],
            "metrics": {
                "steps": len(history_records),
                "s1_steps": len(history_records),
                "s2_escalations": 0,
                "s1_errors": 0,
                "total_cost_usd": 0.001,
            },
        }

    session.snapshot.side_effect = mock_snapshot

    mock_browser = MagicMock()
    mock_browser.evaluate.return_value = browser_eval_return
    session.browser = mock_browser

    return session


# ==============================================================================
# 1. dev_qa.json Schema and Content Tests
# ==============================================================================


def test_dev_qa_json_exists_and_contains_exactly_9_tasks():
    """dev_qa.json must exist and contain exactly 9 specific evaluation tasks."""
    assert DEV_QA_PATH.is_file(), f"Missing tasks file at {DEV_QA_PATH}"

    tasks = load_tasks(DEV_QA_PATH)
    assert len(tasks) == 9, f"Expected exactly 9 tasks, found {len(tasks)}"

    task_ids = [t["id"] for t in tasks]
    expected_ids = [
        "qa_survey",
        "qa_ranking",
        "qa_blog",
        "qa_shop",
        "qa_tournament",
        "qa_crm",
        "fixture",
        "qa_survey_fill_only",
        "qa_hidden_choices",
    ]
    assert task_ids == expected_ids, f"Task IDs mismatch: {task_ids} != {expected_ids}"


def test_dev_qa_json_tasks_have_required_fields_and_valid_structure():
    """Every task in dev_qa.json must have id, url with {BASE}, natural goal, check, check_kind."""
    tasks = load_tasks(DEV_QA_PATH)

    for task in tasks:
        assert isinstance(task["id"], str) and task["id"]
        assert isinstance(task["url"], str) and "{BASE}" in task["url"]
        assert isinstance(task["goal"], str) and len(task["goal"]) > 10
        assert isinstance(task["check"], str) and task["check"]
        assert task["check_kind"] in {"correctness", "completion"}
        assert isinstance(task["auto_confirm"], bool)
        assert isinstance(task["user_replies"], list)

        # Goal must be written in natural language without leaking agent internal keywords
        goal = task["goal"]
        forbidden_internals = ["system1", "system-1", "system2", "system-2", "selector", "snapshot", "nodeid"]
        for forbidden in forbidden_internals:
            assert forbidden not in goal.lower(), f"Goal '{goal}' contains internal keyword '{forbidden}'"


def test_dev_qa_task_8_survey_fill_only_check_expression():
    """Task 8 (qa_survey_fill_only) check must verify all questions answered and form not submitted."""
    tasks = {t["id"]: t for t in load_tasks(DEV_QA_PATH)}
    task8 = tasks["qa_survey_fill_only"]

    assert task8["url"] == "{BASE}/qa/survey"
    assert "!window.__submitted" in task8["check"]
    assert "input[name='q1']:checked" in task8["check"]
    assert "input[name='q2']:checked" in task8["check"]
    assert "dept-select" in task8["check"]

    expected_check = (
        'Boolean(!window.__submitted && document.querySelector("input[name=\'q1\']:checked") && document.querySelector("input[name=\'q2\']:checked") && document.getElementById(\'dept-select\')?.value)'
    )
    assert task8["check"] == expected_check


def test_dev_qa_qa_crm_has_canned_user_reply():
    """T4: qa_crm task in dev_qa.json must include the resolution note in user_replies."""
    tasks = {t["id"]: t for t in load_tasks(DEV_QA_PATH)}
    assert tasks["qa_crm"]["user_replies"] == [
        "Önbellek sunucusu yeniden başlatıldı, gecikme normale döndü."
    ]


def test_dev_qa_check_expressions_pass_node_syntax():
    """All check expressions in dev_qa.json must pass JavaScript syntax check via node --check."""
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js is not installed; skipping JS check syntax test")

    tasks = load_tasks(DEV_QA_PATH)
    for task in tasks:
        check_expr = task["check"]
        res = subprocess.run([node_bin, "--check"], check=False, input=check_expr, text=True, capture_output=True)
        assert res.returncode == 0, f"Syntax error in check for task {task['id']}:\n{res.stderr}\nExpression: {check_expr}"


# ==============================================================================
# 2. Local HTTP Web Server with Clean Route Rewrites Tests
# ==============================================================================


def test_managed_web_server_serves_fixtures_and_rewrites_clean_urls():
    """managed_web_server serves reflex/web directory with clean URL rewrites."""
    with managed_web_server(web_dir=WEB_DIR) as base_url:
        routes_to_test = [
            ("/fixture", "AeroVoyage"),
            ("/qa/survey", "2026 Çalışma Modeli Geri Bildirim Formu"),
            ("/qa/ranking", "1-50 ARASI KÜÇÜKTEN BÜYÜĞE KÖR SIRALAMA"),
            ("/qa/blog", "Doç. Dr. Özal Yıldırım"),
            ("/qa/blog?post=transformer", "TRANSFORMER-2026-OK"),
            ("/qa/shop", "ReflexStore"),
            ("/qa/tournament", "2000 YILININ EN ÖNEMLİ OLAYI NEDİR"),
            ("/qa/crm", "INC-4092: Redis Önbellek Gecikmesi Kaydı"),
        ]

        for route, expected_text in routes_to_test:
            target_url = f"{base_url}{route}"
            req = urllib.request.Request(target_url, headers={"User-Agent": "ReflexEvalTest"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                assert resp.status == 200
                content = resp.read().decode("utf-8")
                assert expected_text in content, f"Expected {expected_text!r} in {route} content"


def test_managed_web_server_404_on_missing_route():
    """managed_web_server returns 404 on non-existent route."""
    with managed_web_server(web_dir=WEB_DIR) as base_url:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base_url}/non_existent_page", timeout=2.0)
        assert exc_info.value.code == 404


# ==============================================================================
# 3. ASK_USER and Waiting State Transition Tests
# ==============================================================================


def test_run_task_transitions_to_asked_user_when_replies_exhausted():
    """When session.status == 'waiting_user' and user_replies is empty, status becomes 'asked_user' cleanly."""
    task = {
        "id": "ask-exhausted",
        "url": "http://127.0.0.1:8788/qa/form",
        "goal": "Interactive flow requiring user response",
        "user_replies": [],
        "check": "true",
    }
    mock_session = _make_mock_session(statuses=["ready", "waiting_user"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["status"] == "asked_user"
    assert res["success"] is False
    assert res["score"] == 0.0
    assert res["last_error"] is None
    # reply_user must NOT have been called with empty string (which raises ValueError)
    mock_session.reply_user.assert_not_called()


def test_run_task_dispatches_available_replies_then_transitions_to_asked_user():
    """Dispatches available canned user_replies, then transitions cleanly to 'asked_user' when depleted."""
    task = {
        "id": "ask-partial",
        "url": "http://127.0.0.1:8788/qa/form",
        "goal": "Interactive flow with two questions",
        "user_replies": ["First Answer"],
        "check": "true",
    }
    # Session starts ready, asks user once, then asks user again when no replies remain
    mock_session = _make_mock_session(statuses=["ready", "waiting_user", "waiting_user"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        res = run_task(task)

    assert res["status"] == "asked_user"
    assert res["success"] is False
    # reply_user was called exactly once with the provided reply
    mock_session.reply_user.assert_called_once_with("First Answer")


# ==============================================================================
# 4. Port 9333 Guard and BU_CDP_URL Isolation Tests
# ==============================================================================


def test_managed_isolated_browser_strictly_forbids_port_9333():
    """Port 9333 must be forbidden and raise ValueError."""
    with (
        pytest.raises(ValueError, match="Port 9333 is reserved"),
        managed_isolated_browser(port=9333),
    ):
        pass


def test_managed_isolated_browser_inherits_port_from_bu_cdp_url_and_rejects_9333():
    """BU_CDP_URL pointing to 9333 must raise ValueError, while other ports are inherited."""
    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9333"}),
        pytest.raises(ValueError, match="Port 9333 is reserved"),
        managed_isolated_browser(),
    ):
        pass

    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9455"}),
        patch("scripts.eval.is_port_open", return_value=True),
        managed_isolated_browser() as cdp_url,
    ):
        assert cdp_url == "http://127.0.0.1:9455"
        assert os.environ["BU_CDP_URL"] == "http://127.0.0.1:9455"


# ==============================================================================
# 5. Check Kind and BASE Replacement Tests
# ==============================================================================


def test_run_task_replaces_base_placeholder():
    """run_task replaces {BASE} in task url with provided base_url."""
    task = {
        "id": "base-test",
        "url": "{BASE}/qa/survey",
        "goal": "Test goal",
        "check": "true",
        "check_kind": "correctness",
    }
    mock_session = _make_mock_session(statuses=["ready", "done"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session) as mock_cls:
        res = run_task(task, base_url="http://127.0.0.1:8788")

    assert res["url"] == "http://127.0.0.1:8788/qa/survey"
    mock_cls.assert_called_once()
    assert mock_cls.call_args[1]["url"] == "http://127.0.0.1:8788/qa/survey"
    assert res["check_kind"] == "correctness"


def test_compute_summary_and_report_support_check_kind_and_asked_user():
    """compute_summary tracks asked_user and by_check_kind breakdowns."""
    task_results = [
        {"id": "t1", "success": True, "score": 1.0, "status": "done", "check_kind": "correctness", "steps": 5, "latencies_ms": [100]},
        {"id": "t2", "success": False, "score": 0.0, "status": "asked_user", "check_kind": "completion", "steps": 2, "latencies_ms": [80]},
        {"id": "t3", "success": False, "score": 0.0, "status": "error", "check_kind": "correctness", "steps": 1, "latencies_ms": [50]},
    ]
    summary = compute_summary(task_results)

    assert summary["total_tasks"] == 3
    assert summary["successful_tasks"] == 1
    assert summary["failed_tasks"] == 2
    assert summary["error_tasks"] == 1
    assert summary["asked_user_tasks"] == 1
    assert "by_check_kind" in summary
    assert summary["by_check_kind"]["correctness"]["total"] == 2
    assert summary["by_check_kind"]["correctness"]["successful"] == 1
    assert summary["by_check_kind"]["completion"]["total"] == 1
    assert summary["by_check_kind"]["completion"]["successful"] == 0

    report = format_summary_report(summary)
    assert "Asked User: 1" in report
    assert "Check Kinds:" in report
    assert "correctness: 1/2" in report
    assert "completion: 0/1" in report


def test_run_eval_automatically_serves_fixtures_when_base_present(tmp_path: Path):
    """run_eval automatically launches managed_web_server when tasks contain {BASE}."""
    tasks = [
        {"id": "auto-base-1", "url": "{BASE}/qa/survey", "goal": "Answer survey", "check": "true"},
    ]
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(tasks), encoding="utf-8")

    mock_session = _make_mock_session(statuses=["ready", "done"])

    with patch("scripts.eval.ReflexSession", return_value=mock_session):
        results, summary = run_eval(tasks_path=tasks_file, verbose=False)

    assert len(results) == 1
    assert results[0]["success"] is True
    assert results[0]["url"].startswith("http://127.0.0.1:")
    assert "/qa/survey" in results[0]["url"]
    assert "{BASE}" not in results[0]["url"]
    assert summary["successful_tasks"] == 1


def test_t3_eval_records_asked_question_error_step_and_flushes_stdout(tmp_path: Path):
    """T3: asked_user records asked_question, error records error_step in JSONL, and stdout prints flush."""
    tasks = [
        {"id": "ask-task", "url": "http://127.0.0.1:8788/qa/crm", "goal": "Update ticket", "check": "true"},
        {"id": "err-task", "url": "http://127.0.0.1:8788/qa/crm", "goal": "Update ticket", "check": "true"},
    ]
    tasks_file = tmp_path / "t3_tasks.json"
    tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
    out_file = tmp_path / "t3_results.jsonl"

    # Session 1: asks user a question after 1 step
    sess_ask = _make_mock_session(statuses=["ready", "waiting_user"])
    sess_ask.last_asked_question = "Çözüm notuna ne yazmamı istersiniz?"

    # Session 2: succeeds on step 1, then errors on step 2
    sess_err = _make_mock_session(statuses=["ready", "ready"])
    step_count = [0]

    def err_step_fn():
        step_count[0] += 1
        if step_count[0] == 1:
            sess_err.history.append({"step": 1, "actor": "S1", "operation": "CLICK", "target": "1", "latency_ms": 50})
            return sess_err.snapshot()
        raise RuntimeError("CDP connection dropped on step 2")

    sess_err.step.side_effect = err_step_fn

    with (
        patch("scripts.eval.ReflexSession", side_effect=[sess_ask, sess_err]),
        patch("builtins.print") as mock_print,
    ):
        results, _ = run_eval(tasks_path=tasks_file, output_path=out_file, verbose=True)

    assert len(results) == 2
    assert results[0]["status"] == "asked_user"
    assert results[0]["asked_question"] == "Çözüm notuna ne yazmamı istersiniz?"
    assert results[0]["error_step"] is None

    assert results[1]["status"] == "error"
    assert results[1]["error_step"] == 2
    assert results[1]["asked_question"] is None

    # Verify JSONL file has both fields persisted
    jsonl_rows = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().splitlines()]
    assert jsonl_rows[0]["asked_question"] == "Çözüm notuna ne yazmamı istersiniz?"
    assert jsonl_rows[1]["error_step"] == 2

    # Verify every console print in run_eval used flush=True
    assert mock_print.call_count > 0
    for call in mock_print.call_args_list:
        assert call.kwargs.get("flush") is True


def test_l3_step_duration_breakdown_and_summary_task_elapsed():
    """L3: Verify s1_ms, s2_ms, act_ms, observe_ms, step_ms in session.history & step_details, Studio app.js usage, and summary elapsed metrics."""
    from unittest.mock import MagicMock
    from reflex.agent import ReflexSession
    from reflex.config import ReflexConfig
    from reflex.supervisor import SupervisorDecision, UserMessageInterpretation

    page1 = {
        "url": "https://example.com/1",
        "title": "Page 1",
        "text": "Content 1",
        "fingerprint": "fp1",
        "actions": [
            {"id": "btn_1", "node": 10, "kind": "click", "role": "button", "label": "Next"},
        ],
    }
    page2 = {
        "url": "https://example.com/2",
        "title": "Page 2",
        "text": "Content 2",
        "fingerprint": "fp2",
        "actions": [
            {"id": "btn_2", "node": 20, "kind": "click", "role": "button", "label": "Finish"},
        ],
    }

    fake_browser = MagicMock()
    fake_browser.observe.side_effect = [page1, page2, page2]
    fake_browser.fresh.return_value = True
    fake_browser.act.return_value = {"rect": {"x": 10, "y": 10, "w": 20, "h": 20}}

    s1_step1 = {
        "operation": "CLICK",
        "target": "1",
        "choice": "btn_1",
        "confidence": 0.92,
        "target_confidence": 0.90,
        "target_margin": 0.80,
        "latency_ms": 310,
        "model": "turnike-4b",
        "usage": {},
    }
    s1_step2_escalate = {
        "operation": "CLICK",
        "target": "1",
        "choice": "btn_2",
        "confidence": 0.25,
        "target_confidence": 0.30,
        "target_margin": 0.10,
        "latency_ms": 280,
        "model": "turnike-4b",
        "usage": {},
    }
    sup_step2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Click Finish",
        model="muse-spark-1.3-contributor",
        latency_ms=4200,
    )

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", side_effect=[s1_step1, s1_step2_escalate]),
        patch("reflex.agent.resolve_escalation", return_value=sup_step2),
    ):
        session = ReflexSession(url="https://example.com/1", goal="Test L3 breakdown", config=ReflexConfig())
        session.step()
        snap = session.step()

    assert len(session.history) == 2
    h1, h2 = session.history[0], session.history[1]

    # Step 1: S1 fast path
    assert h1["s1_ms"] == 310
    assert h1["s2_ms"] == 0
    assert isinstance(h1["act_ms"], int) and h1["act_ms"] >= 0
    assert isinstance(h1["observe_ms"], int) and h1["observe_ms"] >= 0
    assert h1["step_ms"] >= h1["s1_ms"] + h1["s2_ms"] + h1["act_ms"] + h1["observe_ms"]

    # Step 2: S1 -> Gate Escalation -> S2 Override
    assert h2["s1_ms"] == 280
    assert h2["s2_ms"] == 4200
    assert isinstance(h2["act_ms"], int) and h2["act_ms"] >= 0
    assert isinstance(h2["observe_ms"], int) and h2["observe_ms"] >= 0
    assert h2["step_ms"] >= h2["s1_ms"] + h2["s2_ms"] + h2["act_ms"] + h2["observe_ms"]
    assert snap["metrics"]["median_step_ms"] >= 310

    # Verify compute_summary and format_summary_report include per-task and total elapsed time
    summary = compute_summary(
        [
            {"id": "task_a", "success": True, "score": 1.0, "status": "done", "steps": 2, "s2_escalations": 1, "elapsed_ms": 12000, "latencies_ms": [310, 4200]},
            {"id": "task_b", "success": True, "score": 1.0, "status": "done", "steps": 3, "s2_escalations": 0, "elapsed_ms": 8000, "latencies_ms": [250, 260, 270]},
        ]
    )
    assert summary["total_elapsed_ms"] == 20000
    assert summary["avg_elapsed_ms_per_task"] == 10000.0
    assert summary["task_elapsed_ms"] == {"task_a": 12000, "task_b": 8000}
    report = format_summary_report(summary)
    assert "Total Elapsed:" in report
    assert "Task Durations:" in report
    assert "task_a: 12000 ms" in report

    # Verify Studio app.js uses step_ms and median_step_ms
    app_js = (Path(__file__).resolve().parent.parent / "reflex" / "web" / "app.js").read_text(encoding="utf-8")
    assert "lastStep.step_ms" in app_js
    assert "m.median_step_ms" in app_js


