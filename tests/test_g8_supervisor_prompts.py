"""Tests for Task G8: Supervisor Prompts Cleanup."""

from __future__ import annotations

import pytest

from reflex.supervisor import ESCALATION_SYSTEM_PROMPT, FIELD_TEXT_SYSTEM_PROMPT


FORBIDDEN_PATTERNS = [
    "Özal Yıldırım",
    "ozalyildirim",
    "abs.firat.edu.tr",
    "scholar.google.com",
    "HARD RULE D",
    "made with",
    "[DOLU:",
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
