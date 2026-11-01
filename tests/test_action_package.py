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
