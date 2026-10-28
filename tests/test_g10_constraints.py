"""Tests for Task G10: General Mechanism for Submission Constraints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from reflex.agent import ReflexSession
from reflex.config import CalibrationThresholds, ReflexConfig
from reflex.context import DualTierContext
from reflex.gate import evaluate_decision
from reflex.supervisor import UserMessageInterpretation


def test_no_hardcoded_submit_negation_in_agent_and_system1():
    """Neither agent.py nor system1.py should contain hardcoded gönderme / do not submit string checks."""
    agent_src = (Path(__file__).resolve().parent.parent / "reflex" / "agent.py").read_text(encoding="utf-8")
    s1_src = (Path(__file__).resolve().parent.parent / "reflex" / "system1.py").read_text(encoding="utf-8")

    forbidden_patterns = [
        "gönderme",
        "göndermeden",
        "formu gönderme",
        "without submitting",
        "do not submit",
        "don't submit",
    ]
    for pattern in forbidden_patterns:
        assert pattern.lower() not in agent_src.lower(), f"Found '{pattern}' in agent.py"
        assert pattern.lower() not in s1_src.lower(), f"Found '{pattern}' in system1.py"


def test_snapshot_js_contains_form_submitter_detection():
    """snapshot.js must detect form submitters and tag actions with submits_form."""
    snapshot_src = (Path(__file__).resolve().parent.parent / "reflex" / "browser" / "snapshot.js").read_text(encoding="utf-8")
    assert "isFormSubmitter" in snapshot_src
    assert "submits_form" in snapshot_src


def test_ctx_constraints_initialization_and_update():
    """DualTierContext initializes constraints from caller or user message interpretation."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com/survey", "actions": []}
        mock_browser_cls.return_value = mock_browser

        fake_default_interp = UserMessageInterpretation(constraints=[])
        with patch("reflex.agent.interpret_user_message", return_value=fake_default_interp):
            # Normal goal without constraints
            sess_normal = ReflexSession("https://example.com/survey", "Fill the survey")
            assert sess_normal.ctx.constraints == []

            # Goal with submission constraint passed via structured constraints
            sess_constrained = ReflexSession(
                "https://example.com/survey",
                "Fill the survey",
                constraints=["do not submit form"],
            )
            assert "do not submit form" in sess_constrained.ctx.constraints

        # User steers with constraints via reply_user
        fake_interp = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            constraints=["never click submit button"],
        )
        with patch("reflex.agent.interpret_user_message", return_value=fake_interp):
            sess_normal.reply_user("Leave form without final submission")
            assert "never click submit button" in sess_normal.ctx.constraints


def test_evaluate_decision_escalates_on_submits_form_when_constraints_active():
    """evaluate_decision triggers constraint_check escalation when action submits form under active constraints."""
    thresholds = CalibrationThresholds()
    submit_action = {
        "id": "1",
        "kind": "click",
        "role": "button",
        "label": "Formu Gönder",
        "submits_form": True,
    }
    non_submit_action = {
        "id": "2",
        "kind": "click",
        "role": "button",
        "label": "Sonraki Soru",
        "submits_form": False,
    }
    decision = {
        "operation": "CLICK",
        "target": "1",
        "choice": "1",
        "confidence": 0.95,
        "target_confidence": 0.95,
    }

    # 1. Under active constraints, submit action must escalate with constraint_check
    eval_constrained = evaluate_decision(
        decision,
        thresholds,
        constraints=["do not submit form"],
        action=submit_action,
    )
    assert eval_constrained.should_escalate is True
    assert eval_constrained.trigger_type == "constraint_check"
    assert "submits form while constraints are active" in eval_constrained.reason

    # 2. Non-submit action under active constraints should NOT escalate with constraint_check
    eval_non_submit = evaluate_decision(
        decision,
        thresholds,
        constraints=["do not submit form"],
        action=non_submit_action,
    )
    assert eval_non_submit.trigger_type != "constraint_check"

    # 3. Submit action with NO constraints should NOT escalate with constraint_check
    eval_no_constraints = evaluate_decision(
        decision,
        thresholds,
        constraints=[],
        action=submit_action,
    )
    assert eval_no_constraints.trigger_type != "constraint_check"


def test_end_to_end_step_escalates_on_constrained_submission():
    """ReflexSession.step() escalates to System 2 with constraint_check when S1 chooses a form submission."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_page = {
            "url": "https://example.com/form",
            "actions": [
                {"id": "btn_submit", "node": 1, "kind": "click", "role": "button", "label": "Submit", "submits_form": True},
            ],
            "title": "Form Page",
            "text": "Please submit",
        }
        mock_browser.observe.return_value = mock_page
        mock_browser_cls.return_value = mock_browser

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
            session = ReflexSession("https://example.com/form", "Fill fields", constraints=["do not submit form"])
        assert len(session.ctx.constraints) > 0

        # Mock choose_action proposing the submit action
        s1_choice = {
            "operation": "CLICK",
            "target": "btn_submit",
            "choice": "btn_submit",
            "confidence": 0.90,
            "target_confidence": 0.90,
            "latency_ms": 20,
            "model": "turnike-mock",
        }
        with patch("reflex.agent.choose_action", return_value=s1_choice):
            with patch("reflex.agent.resolve_escalation") as mock_resolve:
                fake_sup_decision = MagicMock()
                fake_sup_decision.action_type = "OVERRIDE_ACTION"
                fake_sup_decision.operation = "WAIT"
                fake_sup_decision.target = None
                fake_sup_decision.verified_fact_key = None
                fake_sup_decision.verified_fact_value = None
                fake_sup_decision.tactical_hint = None
                fake_sup_decision.target_url = None
                fake_sup_decision.model = "sup-mock"
                fake_sup_decision.latency_ms = 100
                fake_sup_decision.cost_usd = 0.0
                mock_resolve.return_value = fake_sup_decision

                session.step()

                # Verify resolve_escalation was called with trigger_type="constraint_check"
                mock_resolve.assert_called_once()
                assert mock_resolve.call_args.kwargs["trigger_type"] == "constraint_check"
