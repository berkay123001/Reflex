"""Tests for Task G9: Agent Cleanup & Structured Arbitrary Choices."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from reflex.agent import ReflexSession
from reflex.config import CalibrationThresholds, ReflexConfig
from reflex.context import DualTierContext
from reflex.gate import evaluate_decision
from reflex.supervisor import UserMessageInterpretation


def test_no_books_toscrape_hardcoded_redirect():
    """books.toscrape.com must not be silently redirected to google.com."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://books.toscrape.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
            session = ReflexSession(
                url="https://books.toscrape.com",
                goal="Find cheap books",
                config=ReflexConfig(),
            )
        assert session.url == "https://books.toscrape.com"
        mock_browser_cls.assert_called_once_with("https://books.toscrape.com", browser_language="tr-TR")


def test_ctx_allow_arbitrary_choices_default_and_goal_init():
    """DualTierContext has allow_arbitrary_choices=False by default, or initialized from goal."""
    ctx_default = DualTierContext(primary_goal="Strict goal")
    assert ctx_default.allow_arbitrary_choices is False

    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation(allow_arbitrary_choices=False)):
            session_strict = ReflexSession(url="https://example.com", goal="Fill form accurately")
        assert session_strict.ctx.allow_arbitrary_choices is False

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation(allow_arbitrary_choices=True)):
            session_arb = ReflexSession(url="https://example.com", goal="Anketi kafana göre doldur")
        assert session_arb.ctx.allow_arbitrary_choices is True


def test_reply_user_populates_allow_arbitrary_choices():
    """reply_user sets allow_arbitrary_choices on ctx when supervisor interpretation indicates it."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
            session = ReflexSession(url="https://example.com", goal="Initial strict goal")
        assert session.ctx.allow_arbitrary_choices is False

        fake_interp = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            updated_subgoal="Fill with sample answers",
            allow_arbitrary_choices=True,
        )
        with patch("reflex.agent.interpret_user_message", return_value=fake_interp):
            session.reply_user("istediğin gibi rastgele doldur")
            assert session.ctx.allow_arbitrary_choices is True


def test_gate_evaluates_allow_arbitrary_choices():
    """evaluate_decision relaxes target and margin thresholds when allow_arbitrary_choices is True."""
    thresholds = CalibrationThresholds()
    decision = {
        "operation": "CLICK",
        "target": "3",
        "confidence": 0.40,
        "target_confidence": 0.25,  # Below default target_min_conf (0.32), above relaxed (0.22)
        "target_probabilities": {"3": 0.25, "4": 0.245},  # margin = 0.005, below default target_min_margin (0.04)
    }

    # Strict (allow_arbitrary_choices=False) should escalate
    eval_strict = evaluate_decision(
        decision,
        thresholds,
        allow_arbitrary_choices=False,
    )
    assert eval_strict.should_escalate is True
    assert eval_strict.trigger_type in ("low_target_conf", "ambiguous_target", "low_op_conf")

    # Relaxed (allow_arbitrary_choices=True) should not escalate on target margin
    decision_relaxed = {
        "operation": "CLICK",
        "target": "3",
        "confidence": 0.55,  # Above relaxed op_min for CLICK (0.34) and joint_conf (0.55*0.25=0.1375 > 0.13)
        "target_confidence": 0.25,  # Above relaxed target_min_conf (0.22)
        "target_probabilities": {"3": 0.25, "4": 0.23},  # margin = 0.02, above relaxed (0.01)
    }
    eval_relaxed = evaluate_decision(
        decision_relaxed,
        thresholds,
        allow_arbitrary_choices=True,
    )
    assert eval_relaxed.should_escalate is False


def test_agent_py_source_code_free_of_hardcoded_heuristics():
    """Ensure forbidden strings are not present in reflex/agent.py."""
    agent_path = Path(__file__).resolve().parent.parent / "reflex" / "agent.py"
    source = agent_path.read_text(encoding="utf-8")

    forbidden = [
        "books.toscrape.com",
        "ranking_logic",
        "is_ordering_puzzle",
        "[DOLU:",
        "[dolu:",
        "google ana sayfasına git",
    ]
    for pattern in forbidden:
        assert pattern not in source, f"Found forbidden heuristic pattern '{pattern}' in reflex/agent.py"
