"""Unit tests for Milestone M3 (K3 fix(G9/G10)):
Removal of hardcoded keyword lists, supervisor-driven constraint extraction,
error shielding on extraction failure, context prompt inclusion,
and CLICK gate escalation to System 2 on form submissions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from reflex.agent import ReflexSession
from reflex.config import CalibrationThresholds, ReflexConfig
from reflex.context import DualTierContext
from reflex.gate import evaluate_decision
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation


def _make_fake_browser(page: dict[str, Any] | None = None) -> MagicMock:
    mock_b = MagicMock()
    if page is None:
        page = {
            "url": "https://example.com/survey",
            "title": "Anket Formu",
            "text": "Lütfen formu doldurunuz.",
            "actions": [
                {
                    "id": "btn_submit",
                    "node": 10,
                    "kind": "click",
                    "role": "button",
                    "label": "Formu Gönder",
                    "submits_form": True,
                },
                {
                    "id": "btn_next",
                    "node": 11,
                    "kind": "click",
                    "role": "button",
                    "label": "Sonraki Soru",
                    "submits_form": False,
                },
                {
                    "id": "inp_name",
                    "node": 12,
                    "kind": "type_text",
                    "role": "textbox",
                    "label": "Adınız",
                    "submits_form": False,
                },
            ],
        }
    mock_b.observe.return_value = page
    mock_b.fresh.return_value = True
    return mock_b


def test_keyword_list_removed_from_agent_source():
    """Verify that lines 78-81 keyword list is removed from reflex/agent.py."""
    agent_path = Path(__file__).resolve().parent.parent / "reflex" / "agent.py"
    content = agent_path.read_text(encoding="utf-8")

    # The exact hardcoded tuple or keyword list must not exist in agent.py
    forbidden_keywords = (
        '("kafana göre", "rastgele", "kendi yorumuna", "istediğin", "random", "arbitrary")',
        "('kafana göre', 'rastgele', 'kendi yorumuna', 'istediğin', 'random', 'arbitrary')",
    )
    for kw_tuple in forbidden_keywords:
        assert kw_tuple not in content, f"Hardcoded keyword tuple found in agent.py: {kw_tuple}"


def test_session_init_extracts_constraints_and_arbitrary_choices_via_supervisor():
    """Session init calls supervisor interpret_user_message and sets ctx constraints and allow_arbitrary_choices."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig(supervisor_model="qwen/qwen3.8-omni-flash", allow_model_fallback=False)

    fake_interp = UserMessageInterpretation(
        intent="CONTINUE_OR_STEER",
        allow_arbitrary_choices=True,
        constraints=["do not submit form", "skip question 3"],
        model="qwen/qwen3.8-omni-flash",
    )

    with (
        patch("reflex.agent.Browser", return_value=fake_b),
        patch("reflex.agent.interpret_user_message", return_value=fake_interp) as mock_interp,
    ):
        goal_text = "Fill the survey with sample data and do not submit"
        session = ReflexSession(
            url="https://example.com/survey",
            goal=goal_text,
            config=cfg,
        )

    mock_interp.assert_called_once()
    kwargs = mock_interp.call_args.kwargs
    assert kwargs["user_message"] == goal_text
    assert kwargs["supervisor_model"] == "qwen/qwen3.8-omni-flash"
    assert kwargs["allow_model_fallback"] is False
    assert kwargs["pending_fact_key"] is None

    assert session.status == "ready"
    assert session.last_error is None
    assert session.ctx.allow_arbitrary_choices is True
    assert session.ctx.constraints == ["do not submit form", "skip question 3"]


def test_session_init_does_not_rely_on_keywords_when_supervisor_disallows():
    """Even if goal contains keywords like 'rastgele', supervisor determines allow_arbitrary_choices."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig()

    fake_strict_interp = UserMessageInterpretation(
        intent="CONTINUE_OR_STEER",
        allow_arbitrary_choices=False,
        constraints=["must verify each answer"],
    )

    with (
        patch("reflex.agent.Browser", return_value=fake_b),
        patch("reflex.agent.interpret_user_message", return_value=fake_strict_interp),
    ):
        session = ReflexSession(
            url="https://example.com/survey",
            goal="rastgele değil kesinlikle doğrulanmış bilgi gir",
            config=cfg,
        )

    # Supervisor returned allow_arbitrary_choices=False, so keywords must NOT override it
    assert session.ctx.allow_arbitrary_choices is False
    assert session.ctx.constraints == ["must verify each answer"]


def test_session_start_failure_sets_status_error_and_last_error():
    """If supervisor call during __init__ fails, session sets status='error' and last_error with shield."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig()

    extraction_err = RuntimeError("Supervisor connection timeout: 504 Gateway Timeout")

    with (
        patch("reflex.agent.Browser", return_value=fake_b),
        patch("reflex.agent.interpret_user_message", side_effect=extraction_err),
    ):
        session = ReflexSession(
            url="https://example.com/survey",
            goal="Fill survey",
            config=cfg,
        )

    assert session.status == "error"
    assert session.last_error == f"Goal constraint extraction failed: {extraction_err}"

    # Step immediately returns error snapshot without attempting browser actions
    snap = session.step()
    assert snap["status"] == "error"
    assert snap["last_error"] == f"Goal constraint extraction failed: {extraction_err}"
    assert len(session.history) == 0


def test_context_includes_constraints_in_s1_and_s2_prompts():
    """DualTierContext correctly compiles constraints into compile_system1_goal, build_supervisor_context_block, and snapshot_dict."""
    ctx = DualTierContext(
        primary_goal="Fill application form",
        constraints=["do not submit", "leave phone number blank"],
        allow_arbitrary_choices=True,
    )

    # 1. compile_system1_goal
    s1_goal = ctx.compile_system1_goal()
    assert "Constraints: do not submit; leave phone number blank" in s1_goal

    # 2. build_supervisor_context_block
    s2_ctx = ctx.build_supervisor_context_block([])
    assert "Active Constraints: do not submit; leave phone number blank" in s2_ctx

    # 3. snapshot_dict
    snap_dict = ctx.snapshot_dict()
    assert snap_dict["constraints"] == ["do not submit", "leave phone number blank"]
    assert snap_dict["allow_arbitrary_choices"] is True

    # 4. Empty constraints omits constraints lines
    ctx_empty = DualTierContext(primary_goal="Normal goal")
    assert "Constraints:" not in ctx_empty.compile_system1_goal()
    assert "Active Constraints:" not in ctx_empty.build_supervisor_context_block([])
    assert ctx_empty.snapshot_dict()["constraints"] == []
    assert ctx_empty.snapshot_dict()["allow_arbitrary_choices"] is False


def test_mock_llm_goal_formu_doldur_ama_gonderme_escalates_to_s2():
    """Test mock LLM: goal 'formu doldur ama gönderme' -> constraints populated -> CLICK on submits_form escalates to S2 with constraint_check."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig()

    # Supervisor extracts negative submission constraint
    interp = UserMessageInterpretation(
        intent="CONTINUE_OR_STEER",
        allow_arbitrary_choices=False,
        constraints=["formu gönderme", "do not submit form"],
    )

    with (
        patch("reflex.agent.Browser", return_value=fake_b),
        patch("reflex.agent.interpret_user_message", return_value=interp),
    ):
        session = ReflexSession(
            url="https://example.com/survey",
            goal="formu doldur ama gönderme",
            config=cfg,
        )

    assert "formu gönderme" in session.ctx.constraints
    assert "do not submit form" in session.ctx.constraints

    # System 1 proposes clicking the submit button ("btn_submit" with submits_form=True)
    s1_decision = {
        "operation": "CLICK",
        "target": "btn_submit",
        "choice": "btn_submit",
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 15,
        "model": "turnike-4b",
    }

    # Supervisor S2 handles escalation and overrides
    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="WAIT",
        target=None,
        reasoning="Form submission prevented by active constraint: formu gönderme",
        model="qwen/qwen3.8-omni-flash",
        latency_ms=80,
        cost_usd=0.0002,
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision) as mock_resolve,
    ):
        snap = session.step()

    mock_resolve.assert_called_once()
    kwargs = mock_resolve.call_args.kwargs
    assert kwargs["trigger_type"] == "constraint_check"
    assert "submits form while constraints are active" in kwargs["escalation_reason"]
    assert "formu gönderme" in kwargs["escalation_reason"]

    # Verify escalation was logged in telemetry
    assert len(session.escalations) == 1
    assert session.escalations[0]["trigger_type"] == "constraint_check"


def test_press_enter_on_textbox_escalates_under_active_constraints():
    """Press enter on an input element also triggers constraint_check escalation when constraints are active."""
    thresholds = CalibrationThresholds()
    textbox_action = {
        "id": "inp_name",
        "kind": "type_text",
        "role": "textbox",
        "label": "Adınız",
        "submits_form": False,
    }
    decision = {
        "operation": "PRESS_ENTER",
        "target": "inp_name",
        "confidence": 0.90,
        "target_confidence": 0.90,
    }

    # Under constraints, pressing enter on a form field could submit the form
    eval_res = evaluate_decision(
        decision,
        thresholds,
        constraints=["do not submit form"],
        action=textbox_action,
    )
    assert eval_res.should_escalate is True
    assert eval_res.trigger_type == "constraint_check"
    assert "submits form while constraints are active" in eval_res.reason


def test_unconstrained_form_submit_does_not_trigger_constraint_check():
    """When no constraints exist, clicking a submit button does NOT trigger constraint_check escalation."""
    thresholds = CalibrationThresholds()
    submit_action = {
        "id": "btn_submit",
        "kind": "click",
        "role": "button",
        "label": "Formu Gönder",
        "submits_form": True,
    }
    decision = {
        "operation": "CLICK",
        "target": "btn_submit",
        "confidence": 0.95,
        "target_confidence": 0.95,
    }

    eval_res = evaluate_decision(
        decision,
        thresholds,
        constraints=[],
        action=submit_action,
    )
    assert eval_res.trigger_type != "constraint_check"
