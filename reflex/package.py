"""Action package executor, element pre-conversion, S1 instruction compiler, and matching engine for Reflex G10."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

MAX_PACKAGE_ACTIONS = 8
SUPPORTED_PACKAGE_OPERATIONS = {"CLICK", "TYPE_TEXT", "SELECT"}


@dataclass(frozen=True)
class ExpectedElement:
    """Pre-converted element attributes captured when S2 creates the package."""

    node: int | None
    label: str
    role: str
    operation: str
    checked: str | None = None
    value: str | None = None


@dataclass
class PackageStep:
    """A single action step within an action package."""

    step_index: int  # 1-indexed (1..N)
    operation: str
    target: str  # Page action index, e.g. "4" or "2:1"
    text: str | None
    instruction: str
    expected: ExpectedElement


class ActionPackage:
    """Tracks execution state across multi-step action packages."""

    def __init__(self, id: str, initial_url: str, steps: list[PackageStep]) -> None:
        self.id = id
        self.initial_url = initial_url
        self.steps = steps
        self.current_step_index = 1  # 1..len(steps)
        self.aborted = False
        self.abort_reason: str | None = None
        self.abort_trigger: str | None = None

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def is_finished(self) -> bool:
        return self.aborted or self.current_step_index > self.total_steps

    @property
    def current_step(self) -> PackageStep | None:
        if 1 <= self.current_step_index <= self.total_steps:
            return self.steps[self.current_step_index - 1]
        return None

    def remaining_instructions(self) -> list[str]:
        return [s.instruction for s in self.steps[self.current_step_index - 1 :]]


def compile_system1_instruction(instruction: str) -> str:
    """Compile a compact goal string for System 1 from a package instruction.

    Enforces maximum 15 words. NOTE: compile_system1_goal in reflex/context.py MUST NOT be modified!
    """
    clean = (instruction or "").strip()
    words = clean.split()
    if len(words) > 15:
        return " ".join(words[:15])
    return clean


def normalize_label(label: str) -> str:
    """Normalize element label for fuzzy fallback matching across re-renders."""
    clean = re.sub(r"\s+", " ", str(label or "")).strip().casefold()
    # Strip dropdown arrow suffixes if present
    return clean.split(" → ")[0].strip()


def normalize_role(role: str) -> str:
    """Normalize element role."""
    return str(role or "").strip().casefold()


def create_action_package(
    raw_actions: list[Any],
    page: dict[str, Any],
    targets: dict[str, dict[str, Any]],
) -> ActionPackage | None:
    """Validate and build an ActionPackage from S2 raw actions and current page targets."""
    if not isinstance(raw_actions, list) or not raw_actions:
        return None

    valid_steps: list[PackageStep] = []
    for raw in raw_actions[:MAX_PACKAGE_ACTIONS]:
        if hasattr(raw, "get"):
            op = str(raw.get("operation") or "").upper().strip()
            target_idx = str(raw.get("target") or "").strip().strip("[]").strip()
            instruction = str(raw.get("instruction") or "").strip()
            text_val = raw.get("text")
        elif hasattr(raw, "operation"):
            op = str(getattr(raw, "operation") or "").upper().strip()
            target_idx = str(getattr(raw, "target") or "").strip().strip("[]").strip()
            instruction = str(getattr(raw, "instruction") or "").strip()
            text_val = getattr(raw, "text", None)
        else:
            continue

        if op == "FILL":
            op = "TYPE_TEXT"
        if op not in SUPPORTED_PACKAGE_OPERATIONS:
            continue
        if not target_idx or op not in targets or target_idx not in targets[op]:
            continue
        if not instruction:
            continue

        target_act = targets[op][target_idx]
        text_str = str(text_val).strip() if (op in {"TYPE_TEXT", "SELECT"} and text_val is not None) else None

        expected = ExpectedElement(
            node=target_act.get("node"),
            label=str(target_act.get("label") or ""),
            role=str(target_act.get("role") or ""),
            operation=op,
            checked=str(target_act.get("checked")) if target_act.get("checked") is not None else None,
            value=str(target_act.get("value")) if target_act.get("value") is not None else None,
        )

        step = PackageStep(
            step_index=len(valid_steps) + 1,
            operation=op,
            target=target_idx,
            text=text_str,
            instruction=instruction,
            expected=expected,
        )
        valid_steps.append(step)

    if not valid_steps:
        return None

    pkg_id = f"pkg_{uuid.uuid4().hex[:8]}"
    return ActionPackage(id=pkg_id, initial_url=str(page.get("url") or ""), steps=valid_steps)


def extract_s1_target_index(raw_answers: dict[str, Any], operation: str) -> str | None:
    """Extract S1's chosen target index for the specified operation from raw_answers."""
    if not isinstance(raw_answers, dict):
        return None
    target_key = f"{operation.lower()}_target"
    ans = raw_answers.get(target_key)
    if ans is None:
        return None
    if isinstance(ans, (str, int)):
        return str(ans).strip()
    if isinstance(ans, dict):
        choice = ans.get("choice")
        if choice is not None:
            return str(choice).strip()
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and probs:
            return str(max(probs, key=probs.get)).strip()
    return None


def match_step_element(
    expected: ExpectedElement,
    s1_target_idx: str | None,
    current_targets: dict[str, dict[str, Any]],
    current_page_actions: list[dict[str, Any]],
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Execute the G10 decision tree matching S1's chosen element against the expected element.

    Returns: (matched, abort_trigger, chosen_action_dict)
    """
    op = expected.operation
    op_targets = current_targets.get(op, {})
    if not s1_target_idx or s1_target_idx not in op_targets:
        return False, "package_mismatch", None

    chosen_act = op_targets[s1_target_idx]
    chosen_node = chosen_act.get("node")

    # Check if expected node is present on the current page
    expected_node_present = (
        expected.node is not None
        and any(a.get("node") == expected.node for a in current_page_actions if a.get("node") is not None)
    )

    if expected_node_present:
        if chosen_node == expected.node:
            return True, None, chosen_act
        return False, "package_mismatch", None

    # Expected node missing (re-render): compare normalized label and role
    if (
        normalize_label(chosen_act.get("label", "")) == normalize_label(expected.label)
        and normalize_role(chosen_act.get("role", "")) == normalize_role(expected.role)
    ):
        return True, None, chosen_act

    return False, "package_target_missing", None


def format_package_abort_reason(
    step_num: int,
    total_steps: int,
    trigger: str,
    detail: str,
    discarded_instructions: list[str],
) -> str:
    """Format the structured escalation reason when an action package aborts."""
    instr_summary = (
        f"[{', '.join(repr(s) for s in discarded_instructions)}]"
        if discarded_instructions
        else "[]"
    )
    return (
        f"Action package aborted at step {step_num}/{total_steps} ({trigger}: {detail}). "
        f"Discarded instructions: {instr_summary}"
    )
