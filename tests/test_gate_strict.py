"""Strict gate (`ReflexConfig.s1_strict`): System 1 keeps only confident clicks/typing."""

from __future__ import annotations

from reflex.config import CalibrationThresholds
from reflex.gate import evaluate_decision


def _decision(operation: str, confidence: float, target: str | None = None, target_conf: float = 0.9) -> dict:
    return {
        "operation": operation,
        "confidence": confidence,
        "target": target,
        "choice": target,
        "target_confidence": target_conf if target else None,
        "target_probabilities": {target: target_conf, "x": 1 - target_conf} if target else {},
    }


def test_strict_sends_scrolling_to_system2():
    scroll = _decision("SCROLL_DOWN", 0.95)
    assert not evaluate_decision(scroll, CalibrationThresholds()).should_escalate
    strict = evaluate_decision(scroll, CalibrationThresholds(), strict=True)
    assert strict.should_escalate
    assert strict.trigger_type == "strict_scroll"


def test_strict_removes_arbitrary_choice_relaxation():
    # 0.38 passes the relaxed arbitrary-choice CLICK floor (0.34) but not the regular CLICK threshold (0.42).
    click = _decision("CLICK", 0.38, target="7", target_conf=0.9)
    relaxed = evaluate_decision(click, CalibrationThresholds(), allow_arbitrary_choices=True)
    assert not relaxed.should_escalate
    strict = evaluate_decision(click, CalibrationThresholds(), allow_arbitrary_choices=True, strict=True)
    assert strict.should_escalate
    assert strict.trigger_type == "low_op_conf"


def test_strict_keeps_confident_click():
    click = _decision("CLICK", 0.95, target="3", target_conf=0.95)
    assert not evaluate_decision(click, CalibrationThresholds(), strict=True).should_escalate
