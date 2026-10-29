"""Evaluation baseline `s2_every_step`: System 1 is never queried and System 2 decides every step."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig
from reflex.supervisor import SupervisorDecision


def _session(config: ReflexConfig) -> ReflexSession:
    with (
        patch("reflex.agent.Browser") as browser_cls,
        patch("reflex.agent.interpret_user_message") as interp,
    ):
        browser = MagicMock()
        browser.observe.return_value = {
            "url": "https://example.com/",
            "title": "Example",
            "text": "Example page",
            "actions": [{"id": "1", "kind": "click", "label": "More", "node": 1}],
            "fingerprint": "fp",
        }
        browser.fresh.return_value = True
        browser_cls.return_value = browser
        result = MagicMock()
        result.model_dump.return_value = {}
        result.allow_arbitrary_choices = True  # arbitrary-choice clicks have their own fixed gate rules
        result.constraints = []
        result.plan = None
        interp.return_value = result
        session = ReflexSession(url="https://example.com/", goal="Read the page", config=config)
        session.browser = browser
        return session


def test_s2_every_step_skips_system1_and_escalates():
    session = _session(ReflexConfig(s2_every_step=True, screenshots=False))
    sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Read", summary_for_user="Done.")
    with (
        patch("reflex.agent.choose_action") as s1,
        patch("reflex.agent.resolve_escalation", return_value=sup) as s2,
    ):
        snap = session.step()
    s1.assert_not_called()
    assert s2.call_args.kwargs["trigger_type"] == "s2_every_step"
    assert snap["status"] == "done"


def test_default_hybrid_still_queries_system1():
    session = _session(ReflexConfig(screenshots=False))
    s1_decision = {
        "operation": "CLICK", "target": "1", "choice": "1", "confidence": 0.95, "target_confidence": 0.95,
        "target_probabilities": {"1": 0.95}, "latency_ms": 10, "model": "test", "usage": {},
    }
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision) as s1,
        patch("reflex.agent.resolve_escalation") as s2,
    ):
        session.step()
    s1.assert_called_once()
    s2.assert_not_called()
