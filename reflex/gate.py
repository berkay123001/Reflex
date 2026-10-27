"""Calibrated Multi-Metric Escalation Gate for System 1 (Jev / Turnike) decisions.

Evaluates:
1. Explicit BLOCKED signal from System 1 -> escalate instead of aborting.
2. Premature DONE on step 0 or below `done_min_conf` -> suppress or escalate.
3. Per-operation confidence threshold (`op_min_conf[operation]`).
4. Target confidence threshold (`target_min_conf`) and top-1 vs top-2 margin (`target_min_margin`).
5. Joint confidence (`P(operation) * P(target | operation) >= joint_min_conf`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .config import CalibrationThresholds


class GateEvaluation(BaseModel):
    """Result of evaluating a System 1 decision before execution."""

    should_escalate: bool = False
    reason: str = "ok"
    trigger_type: str = "none"  # "none" | "blocked" | "low_op_conf" | "low_target_conf" | "ambiguous_target" | "low_joint_conf" | "verify_done"
    op_confidence: float = 1.0
    target_confidence: float = 1.0
    joint_confidence: float = 1.0
    target_margin: float = 1.0
    second_best_target: str | None = None


def compute_margin(probabilities: dict[str, float]) -> tuple[float, str | None]:
    """Return (top1 - top2 probability margin, second_best_key)."""
    if not probabilities:
        return 1.0, None
    sorted_items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_items) == 1:
        return 1.0, None
    top1_prob = float(sorted_items[0][1])
    top2_key, top2_prob = sorted_items[1][0], float(sorted_items[1][1])
    return round(max(0.0, top1_prob - top2_prob), 4), top2_key


def evaluate_decision(
    decision: dict[str, Any],
    thresholds: CalibrationThresholds,
    *,
    step_count: int = 0,
    verify_done_with_supervisor: bool = True,
    allow_subjective: bool = False,
) -> GateEvaluation:
    """Decide whether a System 1 prediction is safe to execute directly or should escalate to System 2."""
    operation = str(decision.get("operation", ""))
    op_conf = float(decision.get("confidence", 0.0) or 0.0)
    target_probs = decision.get("target_probabilities") or {}
    target_conf = float(decision.get("target_confidence") if decision.get("target_confidence") is not None else 1.0)
    joint_conf = round(op_conf * target_conf, 4)
    target_margin, second_best = compute_margin(target_probs)

    # 1. Explicit BLOCKED -> Escalate to Supervisor (never abort directly!)
    if operation == "BLOCKED":
        return GateEvaluation(
            should_escalate=True,
            reason=f"System 1 returned BLOCKED (p={op_conf:.2f})",
            trigger_type="blocked",
            op_confidence=op_conf,
            target_confidence=target_conf,
            joint_confidence=joint_conf,
            target_margin=target_margin,
            second_best_target=second_best,
        )

    # 2. DONE handling: prevent premature DONE on step 0 or low-confidence DONE
    if operation == "DONE":
        done_min = thresholds.op_min_conf.get("DONE", 0.72)
        if step_count == 0 or op_conf < done_min:
            return GateEvaluation(
                should_escalate=True,
                reason=f"Premature or low-confidence DONE (step={step_count}, p={op_conf:.2f} < {done_min:.2f})",
                trigger_type="verify_done",
                op_confidence=op_conf,
                target_confidence=1.0,
                joint_confidence=op_conf,
                target_margin=1.0,
            )
        if verify_done_with_supervisor:
            return GateEvaluation(
                should_escalate=True,
                reason=f"Verifying DONE completion with Supervisor (p={op_conf:.2f})",
                trigger_type="verify_done",
                op_confidence=op_conf,
                target_confidence=1.0,
                joint_confidence=op_conf,
                target_margin=1.0,
            )
        return GateEvaluation(
            should_escalate=False,
            op_confidence=op_conf,
            target_confidence=1.0,
            joint_confidence=op_conf,
            target_margin=1.0,
        )

    # 3. Operation-specific confidence threshold (never block exploratory SCROLL/WAIT on low_op_conf)
    if operation not in {"SCROLL_DOWN", "SCROLL_UP", "WAIT"}:
        op_min = thresholds.op_min_conf.get(operation, 0.42)
        if allow_subjective and operation == "CLICK":
            op_min = min(op_min, 0.34)
        if op_conf < op_min:
            return GateEvaluation(
                should_escalate=True,
                reason=f"Low {operation} confidence ({op_conf:.2f} < {op_min:.2f})",
                trigger_type="low_op_conf",
                op_confidence=op_conf,
                target_confidence=target_conf,
                joint_confidence=joint_conf,
                target_margin=target_margin,
                second_best_target=second_best,
            )

    # 4. Target confidence & ambiguity margin check (when an element target is selected)
    if decision.get("target") is not None:
        eff_target_min = 0.22 if allow_subjective else thresholds.target_min_conf
        eff_margin_min = 0.01 if allow_subjective else thresholds.target_min_margin
        eff_joint_min = 0.13 if allow_subjective else thresholds.joint_min_conf
        if target_conf < eff_target_min:
            return GateEvaluation(
                should_escalate=True,
                reason=f"Low target [{decision.get('target')}] confidence ({target_conf:.2f} < {eff_target_min:.2f})",
                trigger_type="low_target_conf",
                op_confidence=op_conf,
                target_confidence=target_conf,
                joint_confidence=joint_conf,
                target_margin=target_margin,
                second_best_target=second_best,
            )
        if len(target_probs) > 1 and target_margin < eff_margin_min:
            return GateEvaluation(
                should_escalate=True,
                reason=(
                    f"Ambiguous target [{decision.get('target')}] vs [{second_best}] "
                    f"(margin {target_margin:.2f} < {eff_margin_min:.2f})"
                ),
                trigger_type="ambiguous_target",
                op_confidence=op_conf,
                target_confidence=target_conf,
                joint_confidence=joint_conf,
                target_margin=target_margin,
                second_best_target=second_best,
            )
        if joint_conf < eff_joint_min:
            return GateEvaluation(
                should_escalate=True,
                reason=f"Low joint confidence ({joint_conf:.2f} < {eff_joint_min:.2f})",
                trigger_type="low_joint_conf",
                op_confidence=op_conf,
                target_confidence=target_conf,
                joint_confidence=joint_conf,
                target_margin=target_margin,
                second_best_target=second_best,
            )

    return GateEvaluation(
        should_escalate=False,
        op_confidence=op_conf,
        target_confidence=target_conf,
        joint_confidence=joint_conf,
        target_margin=target_margin,
        second_best_target=second_best,
    )
