"""Unit and integration tests for Milestone 2 (R2): Optional Step Trace Logging for Distillation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig, get_default_trace_path
from reflex.context import DualTierContext
from reflex.supervisor import UserMessageInterpretation
from reflex.system1 import MAX_SYSTEM1_ELEMENTS, action_space
from reflex.verifier import VerificationResult


def _create_mock_session(
    *,
    url: str = "https://example.com",
    goal: str = "Test distillation trace logging",
    trace_path: str | None = None,
    s2_every_step: bool = False,
    s1_strict: bool = False,
    max_steps: int = 10,
    user_facts: dict[str, str] | None = None,
) -> ReflexSession:
    """Helper to instantiate ReflexSession with mocked CDP Browser and Supervisor."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {
            "url": url,
            "title": "Example Domain",
            "actions": [
                {"id": "1", "kind": "click", "role": "button", "label": "Search", "node": 10},
                {"id": "2", "kind": "fill", "role": "textbox", "label": "Query", "node": 20},
            ],
        }
        mock_browser_cls.return_value = mock_browser

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
            cfg = ReflexConfig(
                trace_path=trace_path,
                s2_every_step=s2_every_step,
                s1_strict=s1_strict,
                max_steps=max_steps,
                screenshots=False,
            )
            session = ReflexSession(
                url=url,
                goal=goal,
                config=cfg,
            )
            if user_facts:
                session.ctx.user_facts.update(user_facts)
            return session


def test_trace_config_resolution(monkeypatch, tmp_path):
    """Test ReflexConfig.trace_path defaults, env var resolution, and explicit overrides."""
    # 1. Default without env var is None
    monkeypatch.delenv("REFLEX_TRACE_PATH", raising=False)
    assert get_default_trace_path() is None
    cfg_default = ReflexConfig()
    assert cfg_default.trace_path is None

    # 2. Env var resolution
    env_target = str(tmp_path / "env_trace.jsonl")
    monkeypatch.setenv("REFLEX_TRACE_PATH", env_target)
    assert get_default_trace_path() == env_target
    cfg_env = ReflexConfig()
    assert cfg_env.trace_path == env_target

    # 3. Explicit override with None disables tracing even if env var is set
    cfg_disabled = ReflexConfig(trace_path=None)
    assert cfg_disabled.trace_path is None

    # 4. Explicit custom string or Path
    cfg_custom = ReflexConfig(trace_path=str(tmp_path / "custom.jsonl"))
    assert cfg_custom.trace_path == str(tmp_path / "custom.jsonl")

    cfg_path_obj = ReflexConfig(trace_path=tmp_path / "from_path.jsonl")  # type: ignore[arg-type]
    assert cfg_path_obj.trace_path == str(tmp_path / "from_path.jsonl")

    # 5. Empty / whitespace string is coerced to None
    cfg_empty = ReflexConfig(trace_path="   ")
    assert cfg_empty.trace_path is None


def test_trace_disabled_by_default(monkeypatch, tmp_path):
    """When trace_path is None, no file is written and metrics show 0 write errors."""
    monkeypatch.delenv("REFLEX_TRACE_PATH", raising=False)
    session = _create_mock_session(trace_path=None)
    assert session.trace_path is None
    assert session.trace_write_errors == 0

    trace_file = tmp_path / "should_not_exist.jsonl"
    session._log_trace_step(
        actor="S1",
        action={"id": "1", "kind": "click", "label": "Search"},
        operation="CLICK",
    )
    assert not trace_file.exists()
    snap = session.snapshot()
    assert snap["metrics"]["trace_write_errors"] == 0


def test_trace_enabled_all_expected_fields(tmp_path):
    """When enabled, all required fields are present in the JSONL line."""
    trace_file = tmp_path / "traces" / "session_trace.jsonl"
    session = _create_mock_session(
        url="https://example.com/flights",
        goal="Find one-way flight from IST to JFK",
        trace_path=str(trace_file),
    )

    action = {
        "id": "input_dest",
        "index": "2",
        "kind": "fill",
        "role": "textbox",
        "tag": "input",
        "label": "Destination Airport",
        "node": 102,
        "href": None,
    }
    verification = VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=False,
        element_state_changed=True,
        effective_progress=True,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="ok",
    )

    session._log_trace_step(
        actor="S1",
        action=action,
        operation="TYPE_TEXT",
        target="2",
        typed_text="JFK",
        goto_url=None,
        verification=verification,
        task_status=None,
        step_no=1,
    )

    assert trace_file.exists()
    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])

    # Check top-level required fields
    expected_keys = {
        "time",
        "timestamp",
        "session_id",
        "step_no",
        "url",
        "mode",
        "s2_model",
        "s1_input",
        "decision_maker",
        "actor",
        "action",
        "operation",
        "target_index",
        "target_tag",
        "target_href",
        "target_element",
        "typed_text",
        "goto_url",
        "validator",
        "progress_made",
        "task_status",
        "status",
    }
    assert expected_keys.issubset(set(row.keys()))

    assert row["step_no"] == 1
    assert row["url"] == "https://example.com/flights"
    assert row["mode"] == "hybrid"
    assert row["decision_maker"] == "S1"
    assert row["actor"] == "S1"
    assert row["operation"] == "TYPE_TEXT"
    assert row["target_index"] == "2"
    assert row["target_tag"] == "textbox"
    assert row["target_href"] is None
    assert row["typed_text"] == "JFK"
    assert row["goto_url"] is None
    assert row["task_status"] is None

    # Check validator nested structure
    validator = row["validator"]
    assert validator["effective_progress"] is True
    assert "flags" in validator
    assert validator["flags"]["page_changed"] is True
    assert validator["flags"]["url_changed"] is False
    assert validator["flags"]["element_state_changed"] is True
    assert validator["flags"]["loop_detected"] is False


def test_s1_input_parity_with_inference_format(tmp_path):
    """Verify S1 input in trace strictly matches compile_system1_goal and action_space(..., 60)."""
    trace_file = tmp_path / "s1_parity.jsonl"
    session = _create_mock_session(
        url="https://example.com/items",
        goal="Select item 65",
        trace_path=str(trace_file),
    )

    # Construct 75 actions to test 60-element truncation
    many_actions = [
        {
            "id": f"btn_{i}",
            "kind": "click",
            "role": "button",
            "label": f"Option {i}",
            "node": 1000 + i,
            "rect": {"x": 10, "y": i * 10, "width": 100, "height": 20},
            "href": f"/opt/{i}",
        }
        for i in range(1, 76)
    ]
    mock_page = {"url": "https://example.com/items", "actions": many_actions}

    session._log_trace_step(
        actor="S2",
        action=many_actions[0],
        operation="CLICK",
        target="1",
        page=mock_page,
        step_no=1,
    )

    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    row = json.loads(lines[0])

    s1_input = row["s1_input"]
    assert "compiled_goal" in s1_input
    assert "elements" in s1_input

    # Goal string parity
    expected_goal = session.ctx.compile_system1_goal()
    assert s1_input["compiled_goal"] == expected_goal

    # Elements truncation and schema parity
    elements = s1_input["elements"]
    assert len(elements) == MAX_SYSTEM1_ELEMENTS  # Must be exactly 60
    assert len(elements) == 60

    # Ensure stripped fields: 'rect', 'node', 'href' MUST NOT be in clean_elements
    for el in elements:
        assert "rect" not in el
        assert "node" not in el
        assert "href" not in el
        assert "role" in el
        assert "index" in el
        assert "operations" in el

    # Compare directly with action_space computation from system1
    expected_raw_elements, _, _ = action_space(many_actions, max_elements=MAX_SYSTEM1_ELEMENTS)
    expected_clean_elements = [
        {k: v for k, v in el.items() if k not in ("rect", "node", "href")}
        for el in expected_raw_elements
    ]
    assert elements == expected_clean_elements


def test_privacy_redaction_password_and_user_facts(tmp_path, monkeypatch):
    """Verify password values, user_facts, and active API keys are masked to '<redacted>'."""
    trace_file = tmp_path / "privacy.jsonl"
    session = _create_mock_session(
        url="https://example.com/login",
        goal="Log in to account",
        trace_path=str(trace_file),
        user_facts={
            "user_email": "alice_secret@example.com",
            "pin": "987654321",
        },
    )

    # Set active API keys in environment
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-my-secret-openrouter-key-999")
    monkeypatch.setenv("META_API_KEY", "LA-my-meta-llama-secret-888")

    # Step 1: Type into a password field
    pw_action = {
        "id": "pw_input",
        "kind": "fill",
        "type": "password",
        "role": "textbox",
        "label": "Account Password",
        "node": 55,
    }
    session._log_trace_step(
        actor="S1",
        action=pw_action,
        operation="TYPE_TEXT",
        target="pw_input",
        typed_text="SuperSecretPassword123!",
        step_no=1,
    )

    # Step 2: Type a user_fact (e.g. secret PIN) into a normal input
    pin_action = {
        "id": "pin_input",
        "kind": "fill",
        "role": "textbox",
        "label": "Security PIN",
        "node": 56,
    }
    session._log_trace_step(
        actor="S2",
        action=pin_action,
        operation="TYPE_TEXT",
        target="pin_input",
        typed_text="987654321",
        step_no=2,
    )

    raw_content = trace_file.read_text(encoding="utf-8")

    # Absolute guarantees: none of these sensitive strings may appear anywhere in the file!
    assert "SuperSecretPassword123!" not in raw_content
    assert "alice_secret@example.com" not in raw_content
    assert "987654321" not in raw_content
    assert "sk-or-v1-my-secret-openrouter-key-999" not in raw_content
    assert "LA-my-meta-llama-secret-888" not in raw_content

    lines = raw_content.strip().splitlines()
    assert len(lines) == 2

    row1 = json.loads(lines[0])
    assert row1["typed_text"] == "<redacted>"
    assert "<redacted>" in row1["s1_input"]["compiled_goal"]

    row2 = json.loads(lines[1])
    assert row2["typed_text"] == "<redacted>"


def test_fault_tolerance_write_errors_swallowed(tmp_path):
    """File write or permission errors do not crash session and increment trace_write_errors."""
    # Point trace_path to an unwriteable directory/file path
    unwritable_path = str(tmp_path / "nonexistent_dir" / "readonly" / "trace.jsonl")

    session = _create_mock_session(trace_path=unwritable_path)
    assert session.trace_write_errors == 0

    # Simulate an OS write error using patch
    with patch("builtins.open", side_effect=PermissionError("Permission denied: [Errno 13]")):
        # This must not raise!
        session._log_trace_step(
            actor="S1",
            action={"id": "1", "kind": "click", "label": "Search"},
            operation="CLICK",
        )

    assert session.trace_write_errors == 1
    snap = session.snapshot()
    assert snap["metrics"]["trace_write_errors"] == 1
    assert session.status == "ready"


def test_terminal_task_status_and_close_sync(tmp_path):
    """Terminal transitions record final status, and close() synchronizes last row if needed."""
    trace_file = tmp_path / "terminal.jsonl"
    session = _create_mock_session(trace_path=str(trace_file))

    # Step 1: Intermediate step
    session._log_trace_step(
        actor="S1",
        action={"id": "1", "kind": "click", "label": "Click button"},
        operation="CLICK",
        task_status=None,
        step_no=1,
    )
    row1 = json.loads(trace_file.read_text(encoding="utf-8").strip().splitlines()[0])
    assert row1["task_status"] is None

    # Step 2: Terminal S2 COMPLETE_TASK transition
    session.status = "done"
    session._log_trace_step(
        actor="S2",
        action={"id": "complete_task", "label": "COMPLETE_TASK", "kind": "complete_task"},
        operation="DONE",
        task_status="done",
        step_no=2,
    )
    lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row2 = json.loads(lines[1])
    assert row2["task_status"] == "done"

    # Close session and verify integrity
    session.close()
    lines_after_close = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines_after_close) == 2
    assert json.loads(lines_after_close[1])["task_status"] == "done"
