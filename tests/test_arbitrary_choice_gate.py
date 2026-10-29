"""Arbitrary radio choices stay with System 1 when the group itself is the decision."""

from __future__ import annotations

from typing import Any

from reflex.config import CalibrationThresholds
from reflex.gate import evaluate_decision


def _radios(group: str, *ids: str, checked: str = "false") -> list[dict[str, Any]]:
    return [
        {"id": item_id, "role": "radio", "group": group, "checked": checked, "kind": "click"}
        for item_id in ids
    ]


def _click(choice: str, confidence: float, probs: dict[str, float]) -> dict[str, Any]:
    top = max(probs, key=probs.get)
    return {
        "operation": "CLICK",
        "target": "1",
        "choice": choice,
        "confidence": confidence,
        "target_confidence": probs[top],
        "target_probabilities": probs,
    }


def test_unanswered_radio_group_is_one_choice_when_arbitrary() -> None:
    actions = _radios("q1", "a", "b", "c", "d")
    decision = _click("a", 0.7, {"a": 0.16, "b": 0.15, "c": 0.14, "d": 0.13})
    strict = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=False,
        action=actions[0],
        page_actions=actions,
    )
    pooled = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=True,
        action=actions[0],
        page_actions=actions,
    )
    assert strict.should_escalate is True
    assert strict.trigger_type == "low_target_conf"
    assert pooled.should_escalate is False
    assert pooled.trigger_type not in {"low_target_conf", "ambiguous_target"}
    assert pooled.target_confidence == 0.58


def test_close_options_in_one_group_are_not_ambiguous() -> None:
    actions = _radios("q1", "a", "b") + [
        {"id": "later", "role": "button", "kind": "click", "label": "Next"},
    ]
    decision = _click("a", 0.9, {"a": 0.40, "b": 0.395, "later": 0.05})
    pooled = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=True,
        action=actions[0],
        page_actions=actions,
    )
    assert pooled.should_escalate is False
    assert pooled.trigger_type != "ambiguous_target"
    assert pooled.target_confidence == 0.795


def test_different_groups_and_answered_groups_still_escalate() -> None:
    split = _radios("q1", "a") + _radios("q2", "b")
    decision = _click("a", 0.9, {"a": 0.40, "b": 0.395})
    different = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=True,
        action=split[0],
        page_actions=split,
    )
    answered = _radios("q1", "a", "b", checked="true")
    answered[1]["checked"] = "false"
    settled = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=True,
        action=answered[0],
        page_actions=answered,
    )
    assert different.should_escalate is True
    assert different.trigger_type == "ambiguous_target"
    assert settled.should_escalate is True
    assert settled.trigger_type == "ambiguous_target"


def test_arbitrary_flag_off_does_not_pool_the_group() -> None:
    actions = _radios("q1", "a", "b")
    decision = _click("a", 0.8, {"a": 0.18, "b": 0.17})
    result = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=False,
        action=actions[0],
        page_actions=actions,
    )
    assert result.should_escalate is True
    assert result.trigger_type == "low_target_conf"
    assert result.target_confidence == 0.18
