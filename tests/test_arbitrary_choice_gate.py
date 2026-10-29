"""Arbitrary radio choices stay with System 1 when the group itself is the decision."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx

from reflex.config import CalibrationThresholds, ReflexConfig
from reflex.gate import evaluate_decision
from reflex.system1 import choose_action


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


def _loose_question(option_count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "id": f"e{index}",
            "node": index,
            "kind": "click",
            "role": "radio",
            "label": f"Seçenek {index}",
            "group": "q1",
            "group_label": "Soru",
            "form_key": "loose",
            "checked": "false",
        }
        for index in range(1, option_count + 1)
    ]


def _choose(actions: list[dict[str, Any]], indexes: dict[str, float]) -> dict[str, Any]:
    """Return the decision choose_action builds from an index-keyed S1 answer."""
    payload = {
        "model": "jev-test",
        "answers": {
            "operation": {
                "choice": "CLICK",
                "probabilities": {"CLICK": 0.9, "DONE": 0.05, "BLOCKED": 0.05},
            },
            "click_target": {"choice": "1", "probabilities": indexes},
        },
        "usage": {},
    }

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://127.0.0.1/s1"))

    page = {"url": "https://example.test/quiz", "title": "Quiz", "text": "Soru", "actions": actions}
    with patch("reflex.system1._HTTP_CLIENT.post", post):
        return choose_action(page, "Rastgele çöz", [], ReflexConfig(decision_backend="local_turnike"))


def test_real_s1_indexes_pool_one_loose_question() -> None:
    actions = _loose_question(5)
    decision = _choose(actions, {str(index): 0.2 for index in range(1, 6)})
    assert set(decision["target_probabilities"]) == {"1", "2", "3", "4", "5"}
    assert set(decision["probabilities"]) == {"e1", "e2", "e3", "e4", "e5"}
    chosen = next(action for action in actions if action["id"] == decision["choice"])
    pooled = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=True,
        action=chosen,
        page_actions=actions,
    )
    strict = evaluate_decision(
        decision,
        CalibrationThresholds(),
        allow_arbitrary_choices=False,
        action=chosen,
        page_actions=actions,
    )
    assert pooled.should_escalate is False
    assert pooled.trigger_type not in {"low_target_conf", "ambiguous_target"}
    assert pooled.target_confidence == 1.0
    assert strict.should_escalate is True
    assert strict.trigger_type == "low_target_conf"


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
