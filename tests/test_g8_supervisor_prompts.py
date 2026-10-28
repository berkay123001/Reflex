"""Tests for Task G8: Supervisor Prompts Cleanup."""

from __future__ import annotations

import pytest

from reflex.supervisor import (
    ESCALATION_SYSTEM_PROMPT,
    FIELD_TEXT_SYSTEM_PROMPT,
    USER_MESSAGE_SYSTEM_PROMPT,
)


FORBIDDEN_PATTERNS = [
    "Özal Yıldırım",
    "ozalyildirim",
    "abs.firat.edu.tr",
    "scholar.google.com",
    "HARD RULE D",
    "made with",
    "[DOLU:",
]

USER_MESSAGE_FORBIDDEN = [
    "gemini",
    "yaş",
    "bölüm",
    "sınıf",
    "cinsiyet",
    "department",
    "gender",
]

SCHEMA_KEYWORDS = [
    "intent",
    "direct_reply",
    "new_goal",
    "updated_subgoal",
    "tactical_hint",
    "target_url",
    "fact_value",
    "allow_arbitrary_choices",
    "constraints",
    "CHAT_ONLY",
    "NEW_GOAL",
    "CONTINUE_OR_STEER",
    "PROVIDE_FACT",
]

INSTRUCTION_KEYWORDS = [
    "switch_task",
    "clarify",
    "answer_question",
]


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_field_text_prompt_free_of_hardcoded_heuristics(pattern):
    """FIELD_TEXT_SYSTEM_PROMPT must not contain any scenario-specific names or heuristics."""
    assert pattern.lower() not in FIELD_TEXT_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize("pattern", FORBIDDEN_PATTERNS)
def test_escalation_prompt_free_of_hardcoded_heuristics(pattern):
    """ESCALATION_SYSTEM_PROMPT must not contain any scenario-specific names, domains, or game rules."""
    assert pattern.lower() not in ESCALATION_SYSTEM_PROMPT.lower()


def test_general_principles_preserved_in_prompts():
    """Verify that domain-agnostic principles are maintained."""
    # General search rules
    assert "SHORT, CLEAN, and ENTITY-FOCUSED" in FIELD_TEXT_SYSTEM_PROMPT
    assert "HARD RULE A (NEVER RE-CLICK REJECTED LINKS)" in ESCALATION_SYSTEM_PROMPT
    assert "HARD RULE B (SCROLL ON SEARCH RESULTS)" in ESCALATION_SYSTEM_PROMPT
    assert "HARD RULE C (SHORT SEARCH QUERIES)" in ESCALATION_SYSTEM_PROMPT
    # 3-step stateful reasoning
    assert "evaluation_previous_goal" in ESCALATION_SYSTEM_PROMPT
    assert "memory" in ESCALATION_SYSTEM_PROMPT
    assert "next_goal" in ESCALATION_SYSTEM_PROMPT


@pytest.mark.parametrize("pattern", USER_MESSAGE_FORBIDDEN)
def test_user_message_prompt_free_of_specific_heuristics(pattern):
    """USER_MESSAGE_SYSTEM_PROMPT must not contain site-specific names or demographic enumerations."""
    assert pattern.lower() not in USER_MESSAGE_SYSTEM_PROMPT.lower()


@pytest.mark.parametrize("keyword", SCHEMA_KEYWORDS)
def test_user_message_prompt_retains_schema_keywords(keyword):
    """USER_MESSAGE_SYSTEM_PROMPT must retain all required output JSON schema keys and intent constants."""
    assert keyword in USER_MESSAGE_SYSTEM_PROMPT


@pytest.mark.parametrize("instruction", INSTRUCTION_KEYWORDS)
def test_user_message_prompt_retains_instruction_keywords(instruction):
    """USER_MESSAGE_SYSTEM_PROMPT must retain instructions and conceptual labels for task switching, clarification, and answering questions."""
    assert instruction in USER_MESSAGE_SYSTEM_PROMPT


def test_user_message_prompt_neutral_generic_examples():
    """Verify generic task-switching example and neutral dummy/sample wording."""
    assert "görevi değiştir ve güncel gelişmeleri araştır" in USER_MESSAGE_SYSTEM_PROMPT
    assert "sample inputs, preferences, or profile fields" in USER_MESSAGE_SYSTEM_PROMPT
