"""Unit tests for Reflex G10 Milestone 2 (R1 & R2): Action Package & System Prompt Rules.

Verifies:
1. Data Model (ActionPackageItem, MAX_PACKAGE_ACTIONS, SupervisorDecision.actions)
2. Action Parsing & Normalization (casing, targets, dict access)
3. Capping at 8 items
4. Discarding invalid items (operation, missing target, empty instruction, non-dict, element presence, text clearing)
5. Instruction 15-word limit
6. Top-level autofill from actions[0] and backward compatibility for empty/missing actions
7. Action type gate (OVERRIDE_ACTION only; COMPLETE_TASK, ASK_USER cleared)
8. System prompt caching determinism and neutral, domain-agnostic wording
"""

from __future__ import annotations

import pytest

from reflex.supervisor import (
    MAX_PACKAGE_ACTIONS,
    PACKAGE_ESCALATION_SYSTEM_PROMPT,
    PACKAGE_OPERATIONS,
    ActionPackageItem,
    ESCALATION_SYSTEM_PROMPT,
    SupervisorDecision,
    get_escalation_system_prompt,
    parse_package_actions,
    parse_supervisor_decision,
)
from tests.test_g8_supervisor_prompts import FORBIDDEN_PATTERNS


# ============================================================================
# 1. Data Model Tests
# ============================================================================


def test_supervisor_decision_actions_default_empty() -> None:
    """SupervisorDecision has actions defaulting to empty list, preserving legacy defaults."""
    decision = SupervisorDecision()
    assert decision.actions == []
    assert decision.action_type == "OVERRIDE_ACTION"
    assert decision.operation is None
    assert decision.target is None
    assert decision.text is None


def test_action_package_item_attributes_and_mapping() -> None:
    """ActionPackageItem supports attribute access, dict mapping access, and serialization."""
    item = ActionPackageItem(
        operation="CLICK",
        target="4",
        text=None,
        instruction='Question 3: choose "Agree"',
    )
    # Attribute access
    assert item.operation == "CLICK"
    assert item.target == "4"
    assert item.text is None
    assert item.instruction == 'Question 3: choose "Agree"'

    # Mapping / dict access
    assert item["operation"] == "CLICK"
    assert item["target"] == "4"
    assert item["text"] is None
    assert item["instruction"] == 'Question 3: choose "Agree"'

    # .get() method
    assert item.get("operation") == "CLICK"
    assert item.get("text") is None
    assert item.get("nonexistent", "fallback") == "fallback"

    # 'in' operator
    assert "operation" in item
    assert "target" in item
    assert "text" in item
    assert "instruction" in item

    # KeyError on invalid key
    with pytest.raises(KeyError):
        _ = item["nonexistent_field"]

    # Serialization
    dumped = item.model_dump()
    assert dumped == {
        "operation": "CLICK",
        "target": "4",
        "text": None,
        "instruction": 'Question 3: choose "Agree"',
    }


# ============================================================================
# 2. Actions Reading & Normalization Tests
# ============================================================================


def test_parse_package_actions_reads_valid_actions() -> None:
    """Parses valid CLICK, TYPE_TEXT, and SELECT actions into ActionPackageItem instances."""
    raw = [
        {"operation": "CLICK", "target": "1", "instruction": "Select question 1"},
        {"operation": "TYPE_TEXT", "target": "2", "text": "test@example.com", "instruction": "Enter user email"},
        {"operation": "SELECT", "target": "3:2", "instruction": "Choose Option 2"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 3
    assert actions[0].operation == "CLICK"
    assert actions[0].target == "1"
    assert actions[0].text is None
    assert actions[0].instruction == "Select question 1"

    assert actions[1].operation == "TYPE_TEXT"
    assert actions[1].target == "2"
    assert actions[1].text == "test@example.com"
    assert actions[1].instruction == "Enter user email"

    assert actions[2].operation == "SELECT"
    assert actions[2].target == "3:2"
    assert actions[2].text is None
    assert actions[2].instruction == "Choose Option 2"


def test_parse_package_actions_normalizes_casing() -> None:
    """Operation names are normalized to uppercase, and 'fill' is mapped to 'TYPE_TEXT'."""
    raw = [
        {"operation": "click", "target": "1", "instruction": "click one"},
        {"operation": "fill", "target": "2", "text": "hello", "instruction": "fill two"},
        {"operation": "type_text", "target": "3", "text": "world", "instruction": "type three"},
        {"operation": "select", "target": "4", "instruction": "select four"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 4
    assert actions[0].operation == "CLICK"
    assert actions[1].operation == "TYPE_TEXT"
    assert actions[2].operation == "TYPE_TEXT"
    assert actions[3].operation == "SELECT"


def test_parse_package_actions_normalizes_targets() -> None:
    """Numeric, string, bracketed, and compound targets are normalized to clean strings."""
    raw = [
        {"operation": "CLICK", "target": 4, "instruction": "target four"},
        {"operation": "CLICK", "target": "[5]", "instruction": "target five bracketed"},
        {"operation": "CLICK", "target": "  6  ", "instruction": "target six padded"},
        {"operation": "SELECT", "target": "2:1", "instruction": "target select option"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 4
    assert actions[0].target == "4"
    assert actions[1].target == "5"
    assert actions[2].target == "6"
    assert actions[3].target == "2:1"


def test_parse_package_actions_bracketed_whitespace_targets() -> None:
    """Bracketed targets with inner/outer whitespace are normalized and matched against valid_targets."""
    raw = [
        {"operation": "CLICK", "target": "[ 4 ]", "instruction": "click target four"},
        {"operation": "SELECT", "target": "[ 10:4 ]", "instruction": "select option four"},
        {"operation": "CLICK", "target": "  [  5  ]  ", "instruction": "padded bracketed target"},
    ]
    actions = parse_package_actions(raw, valid_targets={"4", "10", "5"})
    assert len(actions) == 3
    assert actions[0].target == "4"
    assert actions[1].target == "10:4"
    assert actions[2].target == "5"

    # Empty bracket whitespace targets are discarded
    raw_empty = [
        {"operation": "CLICK", "target": "[  ]", "instruction": "empty brackets"},
        {"operation": "CLICK", "target": "  [   ]  ", "instruction": "spaced empty brackets"},
        {"operation": "CLICK", "target": "[]", "instruction": "pure empty brackets"},
    ]
    assert parse_package_actions(raw_empty) == []
    assert parse_package_actions(raw_empty, valid_targets={"1", "2"}) == []


# ============================================================================
# 3. Capping at MAX_PACKAGE_ACTIONS (8) Tests
# ============================================================================


def test_parse_package_actions_caps_at_8() -> None:
    """More than 8 valid actions are strictly capped at MAX_PACKAGE_ACTIONS (8)."""
    raw = [
        {"operation": "CLICK", "target": str(i), "instruction": f"Step {i}"}
        for i in range(1, 13)
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == MAX_PACKAGE_ACTIONS == 8
    assert [a.target for a in actions] == ["1", "2", "3", "4", "5", "6", "7", "8"]


def test_parse_package_actions_caps_at_8_after_discarding_invalid() -> None:
    """Invalid items are discarded first, and the remaining valid items are capped at 8."""
    raw = [
        {"operation": "UNKNOWN_OP", "target": "bad1", "instruction": "bad"},
        {"operation": "CLICK", "target": "1", "instruction": "Step 1"},
        {"operation": "CLICK", "target": None, "instruction": "missing target"},
        {"operation": "CLICK", "target": "2", "instruction": "Step 2"},
        {"operation": "CLICK", "target": "3", "instruction": "Step 3"},
        {"operation": "CLICK", "target": "4", "instruction": "Step 4"},
        {"operation": "CLICK", "target": "5", "instruction": ""},  # empty instruction
        {"operation": "CLICK", "target": "5", "instruction": "Step 5"},
        {"operation": "CLICK", "target": "6", "instruction": "Step 6"},
        {"operation": "CLICK", "target": "7", "instruction": "Step 7"},
        {"operation": "CLICK", "target": "8", "instruction": "Step 8"},
        {"operation": "CLICK", "target": "9", "instruction": "Step 9"},
        {"operation": "CLICK", "target": "10", "instruction": "Step 10"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 8
    assert [a.target for a in actions] == ["1", "2", "3", "4", "5", "6", "7", "8"]


# ============================================================================
# 4. Discarding Invalid Items Tests
# ============================================================================


def test_parse_package_actions_discards_unknown_operations() -> None:
    """Operations outside PACKAGE_OPERATIONS (CLICK, TYPE_TEXT, SELECT) are discarded."""
    raw = [
        {"operation": "SCROLL_DOWN", "target": "1", "instruction": "scroll down"},
        {"operation": "WAIT", "target": "1", "instruction": "wait"},
        {"operation": "GOTO_URL", "target": "1", "instruction": "navigate"},
        {"operation": "PRESS_ESCAPE", "target": "1", "instruction": "escape"},
        {"operation": "CLICK", "target": "2", "instruction": "valid click"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 1
    assert actions[0].operation == "CLICK"
    assert actions[0].target == "2"


def test_parse_package_actions_discards_missing_target() -> None:
    """Items with missing, empty, or null targets are dropped."""
    raw = [
        {"operation": "CLICK", "target": None, "instruction": "None target"},
        {"operation": "CLICK", "target": "", "instruction": "empty string target"},
        {"operation": "CLICK", "target": "   ", "instruction": "whitespace target"},
        {"operation": "CLICK", "target": "null", "instruction": "literal null target"},
        {"operation": "CLICK", "target": "none", "instruction": "literal none target"},
        {"operation": "CLICK", "target": "3", "instruction": "valid target"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 1
    assert actions[0].target == "3"


def test_parse_package_actions_discards_empty_instruction() -> None:
    """Items with missing or empty instructions are dropped."""
    raw = [
        {"operation": "CLICK", "target": "1", "instruction": None},
        {"operation": "CLICK", "target": "2", "instruction": ""},
        {"operation": "CLICK", "target": "3", "instruction": "   \n\t  "},
        {"operation": "CLICK", "target": "4", "instruction": 123},
        {"operation": "CLICK", "target": "5", "instruction": "valid instruction"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 1
    assert actions[0].target == "5"


def test_parse_package_actions_discards_non_dict_items() -> None:
    """Non-dict items in raw_actions are safely ignored."""
    raw = [
        "not a dict",
        123,
        None,
        ["sublist"],
        {"operation": "CLICK", "target": "1", "instruction": "valid item"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 1
    assert actions[0].target == "1"


def test_parse_package_actions_validates_against_dom_elements() -> None:
    """When valid_targets is provided, items targeting absent indices are discarded."""
    valid_targets = {"1", "2", "3"}
    raw = [
        {"operation": "CLICK", "target": "1", "instruction": "valid target 1"},
        {"operation": "SELECT", "target": "2:1", "instruction": "valid option target (base 2)"},
        {"operation": "CLICK", "target": "99", "instruction": "absent target 99"},
        {"operation": "CLICK", "target": "3", "instruction": "valid target 3"},
    ]
    actions = parse_package_actions(raw, valid_targets=valid_targets)
    assert len(actions) == 3
    assert actions[0].target == "1"
    assert actions[1].target == "2:1"
    assert actions[2].target == "3"


def test_parse_package_actions_clears_text_for_click_and_select() -> None:
    """Non-TYPE_TEXT actions have their text field strictly forced to None."""
    raw = [
        {"operation": "CLICK", "target": "1", "text": "click text", "instruction": "click"},
        {"operation": "SELECT", "target": "2", "text": "select text", "instruction": "select"},
        {"operation": "TYPE_TEXT", "target": "3", "text": "type text", "instruction": "type"},
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 3
    assert actions[0].text is None
    assert actions[1].text is None
    assert actions[2].text == "type text"


# ============================================================================
# 5. Word Limit Tests
# ============================================================================


def test_parse_package_actions_truncates_instruction_to_15_words() -> None:
    """Instructions exceeding 15 words are truncated to exactly 15 words."""
    long_instruction = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    raw = [
        {"operation": "CLICK", "target": "1", "instruction": long_instruction}
    ]
    actions = parse_package_actions(raw)
    assert len(actions) == 1
    words = actions[0].instruction.split()
    assert len(words) == 15
    assert actions[0].instruction == "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"


# ============================================================================
# 6. Top-Level Autofill & Backward Compatibility Tests
# ============================================================================


def test_parse_decision_autofills_toplevel_from_first_action() -> None:
    """Top-level operation, target, and text are auto-filled from actions[0] when actions is present."""
    raw_json = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "SCROLL_DOWN",  # S2 outputted something else top-level
        "target": "999",
        "text": "ignored text",
        "actions": [
            {"operation": "CLICK", "target": "4", "instruction": "Click checkbox"},
            {"operation": "TYPE_TEXT", "target": "5", "text": "admin", "instruction": "Type username"},
        ],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.operation == "CLICK"
    assert decision.target == "4"
    assert decision.text is None
    assert len(decision.actions) == 2
    assert decision.actions[0].target == "4"
    assert decision.actions[1].target == "5"


def test_parse_decision_empty_actions_leaves_toplevel_untouched() -> None:
    """When actions is an empty list, top-level operation, target, and text are preserved."""
    raw_json = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "SCROLL_DOWN",
        "target": "5",
        "text": None,
        "actions": [],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.operation == "SCROLL_DOWN"
    assert decision.target == "5"
    assert decision.text is None
    assert decision.actions == []


def test_parse_decision_missing_actions_leaves_toplevel_untouched() -> None:
    """When actions key is omitted, behavior is byte-for-byte identical to legacy execution."""
    raw_json = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "TYPE_TEXT",
        "target": "3",
        "text": "hello world",
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.operation == "TYPE_TEXT"
    assert decision.target == "3"
    assert decision.text == "hello world"
    assert decision.actions == []


def test_parse_decision_all_invalid_actions_leaves_toplevel_untouched() -> None:
    """When all items in actions are invalid, actions becomes empty and top-level fields are kept."""
    raw_json = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "CLICK",
        "target": "10",
        "text": None,
        "actions": [
            {"operation": "INVALID_OP", "target": "1", "instruction": "test"},
            {"operation": "CLICK", "target": None, "instruction": "test"},
        ],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.actions == []
    assert decision.operation == "CLICK"
    assert decision.target == "10"


def test_parse_decision_bracketed_whitespace_toplevel_target() -> None:
    """Top-level target in decision with bracketed inner whitespace is properly stripped or set to None."""
    raw_json = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "CLICK",
        "target": "[ 4 ]",
        "actions": [],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.target == "4"

    raw_json_compound = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "SELECT",
        "target": "[ 10:4 ]",
        "actions": [],
    }
    decision_compound = parse_supervisor_decision(raw_json_compound)
    assert decision_compound.target == "10:4"

    raw_json_empty = {
        "action_type": "OVERRIDE_ACTION",
        "operation": "CLICK",
        "target": "[  ]",
        "actions": [],
    }
    decision_empty = parse_supervisor_decision(raw_json_empty)
    assert decision_empty.target is None


# ============================================================================
# 7. Action Type Gate Tests
# ============================================================================


def test_parse_decision_actions_ignored_for_complete_task() -> None:
    """actions is strictly cleared to empty list when action_type is COMPLETE_TASK."""
    raw_json = {
        "action_type": "COMPLETE_TASK",
        "summary_for_user": "Task completed successfully",
        "actions": [
            {"operation": "CLICK", "target": "1", "instruction": "should be ignored"}
        ],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.action_type == "COMPLETE_TASK"
    assert decision.actions == []
    assert decision.summary_for_user == "Task completed successfully"


def test_parse_decision_actions_ignored_for_ask_user() -> None:
    """actions is strictly cleared to empty list when action_type is ASK_USER."""
    raw_json = {
        "action_type": "ASK_USER",
        "question_for_user": "Which option do you prefer?",
        "actions": [
            {"operation": "CLICK", "target": "1", "instruction": "should be ignored"}
        ],
    }
    decision = parse_supervisor_decision(raw_json)
    assert decision.action_type == "ASK_USER"
    assert decision.actions == []
    assert decision.question_for_user == "Which option do you prefer?"


# ============================================================================
# 8. System Prompt & Caching Determinism Tests
# ============================================================================


def test_system_prompt_unchanged_in_non_package_mode() -> None:
    """get_escalation_system_prompt returns ESCALATION_SYSTEM_PROMPT byte-for-byte in non-package mode."""
    assert get_escalation_system_prompt(package_mode=False) == ESCALATION_SYSTEM_PROMPT
    assert get_escalation_system_prompt() == ESCALATION_SYSTEM_PROMPT


def test_package_system_prompt_contains_package_rules() -> None:
    """PACKAGE_ESCALATION_SYSTEM_PROMPT includes the 4 core rules and actions schema."""
    prompt = PACKAGE_ESCALATION_SYSTEM_PROMPT
    assert "ACTION PACKAGE RULE (BATCH INDEPENDENT ACTIONS)" in prompt
    assert "2 to 8 items" in prompt
    assert "final item" in prompt
    assert "actions" in prompt
    assert '"operation": "CLICK" | "TYPE_TEXT" | "SELECT"' in prompt
    assert '"instruction": "max 15 words describing the element with visible label and context"' in prompt


def test_package_system_prompt_determinism() -> None:
    """PACKAGE_ESCALATION_SYSTEM_PROMPT is static and deterministic across repeated calls."""
    prompt1 = get_escalation_system_prompt(package_mode=True)
    prompt2 = get_escalation_system_prompt(package_mode=True)
    assert prompt1 is prompt2
    assert prompt1 == PACKAGE_ESCALATION_SYSTEM_PROMPT
    assert prompt1.encode("utf-8") == prompt2.encode("utf-8")


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_package_system_prompt_free_of_hardcoded_heuristics(pattern: str) -> None:
    """PACKAGE_ESCALATION_SYSTEM_PROMPT must not contain any scenario-specific names or heuristics."""
    assert pattern.lower() not in PACKAGE_ESCALATION_SYSTEM_PROMPT.lower()


# ============================================================================
# 9. R3 Action Package Executor & S1 Matching Tests
# ============================================================================

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig, run_mode_of
from reflex.package import (
    ActionPackage,
    ExpectedElement,
    PackageStep,
    compile_system1_instruction,
    create_action_package,
    extract_s1_target_index,
    format_package_abort_reason,
    match_step_element,
    normalize_label,
    normalize_role,
)
from reflex.system1 import build_system1_request
from reflex.verifier import VerificationResult


def _make_mock_browser(page: dict[str, Any]) -> MagicMock:
    mock_b = MagicMock()
    mock_b.observe.return_value = page
    mock_b.fresh.return_value = True
    mock_b.act.return_value = {"rect": {"x": 10, "y": 10, "width": 50, "height": 20}}
    return mock_b


def _make_mock_interp() -> MagicMock:
    return MagicMock(
        intent="TASK_START",
        direct_reply=None,
        constraints=[],
        allow_arbitrary_choices=True,
        budget_sources=None,
        model="test-supervisor",
        plan=None,
    )


def _ok_verify() -> VerificationResult:
    return VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=False,
        element_state_changed=True,
        effective_progress=True,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="Progress verified",
    )


def test_executor_4_step_all_agree_single_s2_call() -> None:
    """A 4-step package where S1 agrees on all steps executes completely with only 1 S2 call."""
    page = {
        "url": "https://example.com/quiz",
        "title": "Quiz",
        "text": "Quiz questions",
        "actions": [
            {"id": "q1_opt1", "node": 101, "kind": "click", "role": "radio", "label": "Option A"},
            {"id": "q2_opt2", "node": 102, "kind": "click", "role": "radio", "label": "Option B"},
            {"id": "q3_text", "node": 103, "kind": "fill", "role": "textbox", "label": "Notes"},
            {"id": "btn_sub", "node": 104, "kind": "click", "role": "button", "label": "Submit"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Batching 4 quiz inputs",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Select Option A"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Select Option B"),
            ActionPackageItem(operation="TYPE_TEXT", target="3", text="Looks good", instruction="Type Notes"),
            ActionPackageItem(operation="CLICK", target="4", instruction="Submit Quiz"),
        ],
    )

    s1_responses = [
        {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9, "model": "turnike"},
        {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.85, "model": "turnike"},
        {"raw_answers": {"type_text_target": {"choice": "3"}}, "confidence": 0.95, "model": "turnike"},
        {"raw_answers": {"click_target": {"choice": "4"}}, "confidence": 0.99, "model": "turnike"},
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=s1_responses),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Complete quiz", config=cfg)

        # Step 1: Initial escalation to S2, which plans package and runs Step 1
        snap1 = session.step()
        assert len(session.history) == 1
        assert session.history[0]["actor"] == "S2"  # step 1 is the Supervisor's own action
        assert session.history[0]["package"]["step"] == 1
        assert session.history[0]["package"]["of"] == 4
        assert session.history[0]["package"]["agreed"] is True
        assert session.active_package is not None
        assert session.active_package.current_step_index == 2

        # Step 2: Executes package step 2
        snap2 = session.step()
        assert len(session.history) == 2
        assert session.history[1]["actor"] == "S1"
        assert session.history[1]["package"]["step"] == 2
        assert session.history[1]["package"]["agreed"] is True

        # Step 3: Executes package step 3
        snap3 = session.step()
        assert len(session.history) == 3
        assert session.history[2]["actor"] == "S1"
        assert session.history[2]["package"]["step"] == 3
        assert session.history[2]["text"] == "Looks good"
        assert session.history[2]["package"]["agreed"] is True

        # Step 4: Executes package step 4
        snap4 = session.step()
        assert len(session.history) == 4
        assert session.history[3]["actor"] == "S1"
        assert session.history[3]["package"]["step"] == 4
        assert session.history[3]["package"]["agreed"] is True

        # Package finished
        assert session.active_package is None
        assert mock_sup.call_count == 1
        metrics = snap4["metrics"]
        assert metrics["s2_calls"] == 1
        assert metrics["packages_started"] == 1
        assert metrics["package_steps_executed"] == 4
        assert metrics["package_aborts"] == {}
        assert metrics["s1_package_agreement"]["step_1"] == 1.0
        assert metrics["s1_package_agreement"]["step_k"] == 1.0


def test_executor_step2_mismatch_aborts_and_escalates_to_s2() -> None:
    """When S1 picks a different element on step 2, package aborts and escalates with package_mismatch and discarded instructions."""
    page = {
        "url": "https://example.com/quiz",
        "title": "Quiz",
        "text": "Quiz questions",
        "actions": [
            {"id": "q1_opt1", "node": 101, "kind": "click", "role": "radio", "label": "Option A"},
            {"id": "q2_opt2", "node": 102, "kind": "click", "role": "radio", "label": "Option B"},
            {"id": "q2_opt3", "node": 105, "kind": "click", "role": "radio", "label": "Option C"},
            {"id": "btn_sub", "node": 104, "kind": "click", "role": "button", "label": "Submit"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Batching 3 quiz inputs",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Select Option A"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Select Option B"),
            ActionPackageItem(operation="CLICK", target="4", instruction="Submit Quiz"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="3",
        reasoning="Single override after mismatch",
        actions=[],
    )

    s1_responses = [
        {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
        {"raw_answers": {"click_target": {"choice": "3"}}, "confidence": 0.8},
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=s1_responses),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Complete quiz", config=cfg)

        # Step 1: S2 packages, runs step 1
        session.step()
        assert len(session.history) == 1
        assert session.active_package is not None

        # Step 2: S1 selects target 3 instead of target 2 -> mismatch abort!
        session.step()
        assert mock_sup.call_count == 2
        second_call_kwargs = mock_sup.call_args_list[1].kwargs
        assert second_call_kwargs["trigger_type"] == "package_mismatch"
        reason = second_call_kwargs["escalation_reason"]
        assert "package_mismatch" in reason
        assert "Select Option B" in reason
        assert "Submit Quiz" in reason
        assert session.active_package is None
        assert session.package_aborts["package_mismatch"] == 1
        assert session.s1_pkg_stepk_total == 1
        assert session.s1_pkg_stepk_agreed == 0


def test_executor_missing_node_re_render_label_match_executes() -> None:
    """When expected node is missing due to DOM re-render, matching normalized label/role executes successfully."""
    page_step1 = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_agree", "node": 101, "kind": "click", "role": "button", "label": "I Agree"},
            {"id": "btn_next", "node": 102, "kind": "click", "role": "button", "label": "Proceed → Next"},
        ],
    }
    # On step 2, page re-renders: node 102 disappeared, node 202 is now present with label "Proceed"
    page_step2 = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_agree", "node": 101, "kind": "click", "role": "button", "label": "I Agree"},
            {"id": "btn_next_new", "node": 202, "kind": "click", "role": "button", "label": "Proceed"},
        ],
    }
    mock_b = _make_mock_browser(page_step1)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Agree and proceed",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click I Agree"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Proceed"),
        ],
    )

    s1_responses = [
        {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
        {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.85},
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=s1_responses),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/form", goal="Agree and proceed", config=cfg)
        session.step()

        # Update page to re-rendered DOM before step 2
        mock_b.observe.return_value = page_step2
        session.step()

        assert len(session.history) == 2
        assert session.history[1]["actor"] == "S1"
        assert session.history[1]["node"] == 202
        assert session.history[1]["package"]["agreed"] is True
        assert session.package_aborts["package_target_missing"] == 0
        assert mock_sup.call_count == 1


def test_executor_missing_node_label_mismatch_aborts() -> None:
    """When expected node is missing and S1 selects an element with different label, package aborts with package_target_missing."""
    page_step1 = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Step 1"},
            {"id": "btn_agree", "node": 102, "kind": "click", "role": "button", "label": "I Agree"},
        ],
    }
    page_step2 = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Step 1"},
            {"id": "btn_cancel", "node": 202, "kind": "click", "role": "button", "label": "Cancel"},
        ],
    }
    mock_b = _make_mock_browser(page_step1)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2-step package",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Step 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click I Agree"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Fallback after abort",
        actions=[],
    )

    s1_responses = [
        {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
        {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.85},
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=s1_responses),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/form", goal="Form", config=cfg)
        session.step()

        # DOM re-renders with "Cancel" instead of "I Agree"
        mock_b.observe.return_value = page_step2
        session.step()

        assert mock_sup.call_count == 2
        second_kwargs = mock_sup.call_args_list[1].kwargs
        assert second_kwargs["trigger_type"] == "package_target_missing"
        assert session.package_aborts["package_target_missing"] == 1
        assert "package_target_missing" in second_kwargs["escalation_reason"]


def test_executor_url_change_aborts_package() -> None:
    """When an action causes a URL change, the action package aborts immediately and escalates to S2."""
    page_step1 = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Next"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Next 2"},
        ],
    }
    page_step2 = {
        "url": "https://example.com/result",  # URL changed!
        "actions": [
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Next 2"},
        ],
    }
    mock_b = _make_mock_browser(page_step1)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 steps",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Next"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Next 2"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Escalated after URL change",
        actions=[],
    )

    s1_resp = {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9}

    url_change_verify = VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=True,
        element_state_changed=False,
        effective_progress=True,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="Navigated to new URL",
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", return_value=s1_resp),
        patch("reflex.agent.verify_step", return_value=url_change_verify),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)

        def _nav_act(action, before_page, text=None):
            mock_b.observe.return_value = page_step2
            return {"rect": {"x": 10, "y": 10}}

        mock_b.act.side_effect = _nav_act
        session.step()

        # Step 1 executed, but URL changed to /result
        assert session.active_package is None
        assert session.package_aborts["url_changed"] == 1
        assert session.pending_escalation_reason is not None
        assert session.pending_escalation_reason[0] == "url_changed"

        # Next step escalates to S2
        session.step()
        assert mock_sup.call_count == 2
        assert mock_sup.call_args_list[1].kwargs["trigger_type"] == "url_changed"


def test_executor_no_progress_aborts_package() -> None:
    """When verification detects no progress, the package aborts immediately."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 steps",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="After no progress",
        actions=[],
    )

    s1_resp = {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9}

    fake_verify = VerificationResult(
        executed_ok=True,
        page_changed=False,
        url_changed=False,
        element_state_changed=False,
        effective_progress=False,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="No change detected",
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", return_value=s1_resp),
        patch("reflex.agent.verify_step", return_value=fake_verify),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()

        # Step 1 saw effective_progress=False, so package aborted post-step
        assert session.active_package is None
        assert session.package_aborts["no_progress"] == 1
        assert session.pending_escalation_reason[0] == "no_progress"

        # Next step escalates
        session.step()
        assert mock_sup.call_count == 2
        assert mock_sup.call_args_list[1].kwargs["trigger_type"] == "no_progress"


def test_executor_s1_error_shields_session_from_crash() -> None:
    """When System 1 throws an error, the action package aborts, escalates cleanly to S2, and session never crashes."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 steps",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="2",
        reasoning="Fallback after S1 crash",
        actions=[],
    )

    s1_effects = [
        {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
        RuntimeError("Turnike 503 Server Unavailable"),
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=s1_effects),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()

        # Step 2 encounters S1 error: must NOT crash session status to 'error'!
        session.step()
        assert session.status == "ready"
        assert session.s1_errors == 1
        assert session.package_aborts["s1_error"] == 1
        assert mock_sup.call_count == 2
        assert mock_sup.call_args_list[1].kwargs["trigger_type"] == "s1_error"


def test_executor_type_text_uses_s2_preset_text() -> None:
    """TYPE_TEXT actions in a package use the preset text from S2 without calling generate_field_text."""
    page = {
        "url": "https://example.com/signup",
        "actions": [
            {"id": "inp_user", "node": 101, "kind": "fill", "role": "textbox", "label": "Username"},
            {"id": "inp_email", "node": 102, "kind": "fill", "role": "textbox", "label": "Email"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="TYPE_TEXT",
        target="1",
        text="berkay123",
        reasoning="Fill registration form",
        actions=[
            ActionPackageItem(operation="TYPE_TEXT", target="1", text="berkay123", instruction="Type Username"),
            ActionPackageItem(operation="TYPE_TEXT", target="2", text="user@example.test", instruction="Type Email"),
        ],
    )

    s1_responses = [
        {"raw_answers": {"type_text_target": {"choice": "1"}}, "confidence": 0.9},
        {"raw_answers": {"type_text_target": {"choice": "2"}}, "confidence": 0.9},
    ]

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", side_effect=s1_responses),
        patch("reflex.agent.generate_field_text") as mock_field_text,
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/signup", goal="Register", config=cfg)

        # Step 1: uses "berkay123"
        session.step()
        assert mock_b.act.call_args_list[0].kwargs["text"] == "berkay123"

        # Step 2: uses "user@example.test"
        session.step()
        assert mock_b.act.call_args_list[1].kwargs["text"] == "user@example.test"

        # generate_field_text was never called because text was supplied by S2 package
        assert mock_field_text.call_count == 0
        assert session.history[0]["text"] == "berkay123"
        assert session.history[1]["text"] == "user@example.test"


def test_run_mode_package_and_s1_request_isolation() -> None:
    """Verifies package mode identity, legacy mode invariance, and byte-for-byte S1 request isolation."""
    # 1. run_mode_of returns package for s2_action_packages=True
    assert run_mode_of(s2_every_step=False, s1_strict=False, s2_action_packages=True) == "package"
    # Legacy modes unchanged
    assert run_mode_of(s2_every_step=True, s1_strict=False) == "s2_only"
    assert run_mode_of(s2_every_step=False, s1_strict=True) == "strict"
    assert run_mode_of(s2_every_step=False, s1_strict=False) == "hybrid"

    # 2. compile_system1_instruction strictly caps to 15 words
    long_inst = "This is a very detailed instruction with more than fifteen words in order to test truncation limit"
    compiled = compile_system1_instruction(long_inst)
    assert len(compiled.split()) == 15
    assert compiled == "This is a very detailed instruction with more than fifteen words in order to test"

    # 3. Non-package S1 request payload is byte-for-byte identical
    page = {
        "url": "https://example.com/books",
        "actions": [
            {"id": "btn_search", "node": 10, "kind": "click", "role": "button", "label": "Search"},
        ],
    }
    body1, targets1, controls1 = build_system1_request(
        page=page,
        compiled_goal="Find books",
        history=[],
    )
    body2, targets2, controls2 = build_system1_request(
        page=page,
        compiled_goal="Find books",
        history=[],
    )
    assert body1 == body2
    assert targets1 == targets2
    assert controls1 == controls2


