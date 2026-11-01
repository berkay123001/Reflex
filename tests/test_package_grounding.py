"""Action-package grounding: System 1 answers only the target question for the package step's operation.

Also covers actor attribution (step 1 is the Supervisor's decision) and package completion accounting.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig
from reflex.supervisor import ActionPackageItem, SupervisorDecision
from reflex.system1 import build_system1_request, ground_target
from reflex.verifier import VerificationResult


def _resp(answers: dict[str, Any]) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"answers": answers, "model": "turnike-test", "usage": {"cost": 0.0}}
    r.raise_for_status.return_value = None
    return r


QUIZ_PAGE = {
    "url": "https://example.com/quiz",
    "title": "Quiz",
    "text": "Q1 Q2",
    "actions": [
        {"id": "a1", "node": 101, "kind": "click", "role": "radio", "label": "Q1 → Agree"},
        {"id": "a2", "node": 102, "kind": "click", "role": "radio", "label": "Q2 → Disagree"},
        {"id": "a3", "node": 103, "kind": "fill", "role": "textbox", "label": "Comment"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ],
}


def test_ground_target_sends_only_the_operation_target_question() -> None:
    cfg = ReflexConfig(decision_backend="local_turnike")
    post = MagicMock(return_value=_resp({"click_target": {"choice": "2", "probabilities": {"1": 0.1, "2": 0.9}}}))
    with patch("reflex.system1._HTTP_CLIENT.post", post):
        out = ground_target(QUIZ_PAGE, 'Question 2: choose "Disagree"', "CLICK", [], cfg)

    body = post.call_args.kwargs["json"]
    assert list(body["questions"]) == ["click_target"]
    assert body["questions"]["click_target"]["instructions"]["goal"] == 'Question 2: choose "Disagree"'
    # Everything except the question set is the regular System 1 request.
    full, _, _ = build_system1_request(QUIZ_PAGE, 'Question 2: choose "Disagree"', [], cfg)
    assert body["state"] == full["state"]
    assert body["questions"]["click_target"] == full["questions"]["click_target"]
    assert out["target"] == "2"
    assert out["choice"] == "a2"
    assert out["target_confidence"] == 0.9
    assert out["raw_answers"]["click_target"]["choice"] == "2"


def test_ground_target_without_candidates_skips_the_model_call() -> None:
    cfg = ReflexConfig(decision_backend="local_turnike")
    post = MagicMock()
    with patch("reflex.system1._HTTP_CLIENT.post", post):
        out = ground_target(QUIZ_PAGE, "Pick the colour", "SELECT", [], cfg)
    post.assert_not_called()
    assert out["target"] is None


def test_ground_target_request_is_json_serialisable_and_regular_request_unchanged() -> None:
    cfg = ReflexConfig(decision_backend="local_turnike")
    before = json.dumps(build_system1_request(QUIZ_PAGE, "goal", [], cfg)[0], sort_keys=True)
    post = MagicMock(return_value=_resp({"type_text_target": {"choice": "3", "probabilities": {"3": 1.0}}}))
    with patch("reflex.system1._HTTP_CLIENT.post", post):
        ground_target(QUIZ_PAGE, "Comment field: type a note", "TYPE_TEXT", [], cfg)
    assert json.dumps(build_system1_request(QUIZ_PAGE, "goal", [], cfg)[0], sort_keys=True) == before


# ---------------------------------------------------------------------------
# Executor accounting
# ---------------------------------------------------------------------------


def _browser(page: dict[str, Any]) -> MagicMock:
    b = MagicMock()
    b.observe.return_value = page
    b.fresh.return_value = True
    b.act.return_value = {"rect": {"x": 1, "y": 1, "width": 10, "height": 10}}
    return b


def _interp() -> MagicMock:
    return MagicMock(
        intent="TASK_START",
        direct_reply=None,
        constraints=[],
        allow_arbitrary_choices=True,
        budget_sources=None,
        model="test-supervisor",
        plan=None,
    )


def _verify(url_changed: bool = False) -> VerificationResult:
    return VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=url_changed,
        element_state_changed=True,
        effective_progress=True,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="ok",
    )


def _two_step_package() -> SupervisorDecision:
    return SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="two answers",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction='Question 1: choose "Agree"'),
            ActionPackageItem(operation="CLICK", target="2", instruction='Question 2: choose "Disagree"'),
        ],
    )


def _grounded(idx: str) -> dict[str, Any]:
    return {"target": idx, "raw_answers": {"click_target": {"choice": idx}}, "target_confidence": 0.9}


def test_step_one_is_attributed_to_s2_and_later_steps_to_s1() -> None:
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)
    with (
        patch("reflex.agent.Browser", return_value=_browser(QUIZ_PAGE)),
        patch("reflex.agent.interpret_user_message", return_value=_interp()),
        patch("reflex.agent.resolve_escalation", return_value=_two_step_package()),
        patch("reflex.agent.ground_target", side_effect=[_grounded("1"), _grounded("2")]),
        patch("reflex.agent.verify_step", return_value=_verify()),
    ):
        s = ReflexSession(url=QUIZ_PAGE["url"], goal="Answer the quiz", config=cfg)
        s.step()
        snap = s.step()
    assert [h["actor"] for h in s.history] == ["S2", "S1"]
    assert s.history[0]["package"]["agreed"] is True
    assert snap["metrics"]["s1_steps"] == 1
    assert snap["metrics"]["packages_completed"] == 1
    assert snap["metrics"]["package_aborts"] == {}


def test_url_change_on_last_step_completes_the_package() -> None:
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)
    result_page = {**QUIZ_PAGE, "url": "https://example.com/result"}
    browser = _browser(QUIZ_PAGE)
    followup = SupervisorDecision(action_type="OVERRIDE_ACTION", operation="SCROLL_DOWN", reasoning="read result")
    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[_two_step_package(), followup]) as sup,
        patch("reflex.agent.ground_target", side_effect=[_grounded("1"), _grounded("2")]),
        patch("reflex.agent.verify_step", side_effect=[_verify(), _verify(url_changed=True), _verify()]),
    ):
        s = ReflexSession(url=QUIZ_PAGE["url"], goal="Answer the quiz", config=cfg)
        s.step()

        def _submit(action, before_page, text=None):
            browser.observe.return_value = result_page
            return {"rect": {"x": 1, "y": 1}}

        browser.act.side_effect = _submit
        s.step()
        assert s.active_package is None
        assert s.packages_completed == 1
        assert dict(s.package_aborts) == {}
        assert s.pending_escalation_reason is None
        browser.act.side_effect = None
        s.step()
    assert sup.call_args_list[1].kwargs["trigger_type"] == "s2_every_step"


def test_target_outside_s1_window_aborts_without_asking_s1() -> None:
    actions = [
        {"id": f"b{i}", "node": 200 + i, "kind": "click", "role": "button", "label": f"Item {i}"}
        for i in range(1, 71)
    ]
    page = {"url": "https://example.com/long", "title": "", "text": "", "actions": actions}
    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="two picks",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Item 1"),
            ActionPackageItem(operation="CLICK", target="65", instruction="Click Item 65"),
        ],
    )
    fallback = SupervisorDecision(action_type="OVERRIDE_ACTION", operation="CLICK", target="65", reasoning="direct")
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)
    grounding = MagicMock(side_effect=[_grounded("1")])
    with (
        patch("reflex.agent.Browser", return_value=_browser(page)),
        patch("reflex.agent.interpret_user_message", return_value=_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision, fallback]) as sup,
        patch("reflex.agent.ground_target", grounding),
        patch("reflex.agent.verify_step", return_value=_verify()),
    ):
        s = ReflexSession(url=page["url"], goal="Pick items", config=cfg)
        s.step()
        s.step()
    assert grounding.call_count == 1  # only the step-1 agreement probe
    assert s.package_aborts["package_target_out_of_window"] == 1
    assert sup.call_args_list[1].kwargs["trigger_type"] == "package_target_out_of_window"
