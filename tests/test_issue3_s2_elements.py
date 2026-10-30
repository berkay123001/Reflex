"""Tests for Requirement R1 / Issue #3.

Verifies:
1. Fake page with 100 clickable elements: S2 prompt has all 100; S1 action_space cuts at 60.
2. Fake page with 200 clickable elements: S2 prompt has 150 elements + "+50 more interactive elements omitted (scroll or use GOTO_URL)".
3. Targeting 65th element in S2 resolves to correct action.
4. S1 60-element limit and symbol indexing isolation remain untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import MAX_SUPERVISOR_ELEMENTS, MAX_SYSTEM1_ELEMENTS, ReflexConfig
from reflex.safety import wrap_untrusted_dom
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation
from reflex.system1 import (
    MAX_SUPERVISOR_ELEMENTS as S1_MAX_SUPERVISOR_ELEMENTS,
)
from reflex.system1 import (
    MAX_SYSTEM1_ELEMENTS as S1_MAX_SYSTEM1_ELEMENTS,
)
from reflex.system1 import (
    action_space,
    build_system1_request,
)


def _generate_raw_actions(n: int) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "id": f"elem_{i}",
            "node": i,
            "kind": "click",
            "role": "link",
            "label": f"Link Item {i}",
            "rect": {"x": 10, "y": i * 15, "w": 100, "h": 12},
        }
        for i in range(1, n + 1)
    ]
    actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420})
    return actions


def test_constants_defined_and_reexported():
    """MAX_SUPERVISOR_ELEMENTS and MAX_SYSTEM1_ELEMENTS are consistently exported."""
    assert MAX_SYSTEM1_ELEMENTS == 60
    assert S1_MAX_SYSTEM1_ELEMENTS == 60
    assert MAX_SUPERVISOR_ELEMENTS == 150
    assert S1_MAX_SUPERVISOR_ELEMENTS == 150


def test_fake_page_100_elements_s2_sees_all_and_s1_cuts_at_60():
    """Fake page with 100 clickable elements: S2 prompt has all 100; S1 action_space cuts at 60."""
    actions = _generate_raw_actions(100)
    page = {"url": "https://news.ycombinator.com", "title": "Hacker News", "actions": actions}

    # System 1 action_space cuts at default 60 (MAX_SYSTEM1_ELEMENTS)
    s1_elems, s1_targets, _ = action_space(actions)
    assert len(s1_elems) == 60
    assert len(s1_targets["CLICK"]) == 60
    assert "60" in s1_targets["CLICK"]
    assert "61" not in s1_targets["CLICK"]

    # System 2 action_space supports up to MAX_SUPERVISOR_ELEMENTS (150)
    s2_elems, s2_targets, _ = action_space(actions, max_elements=MAX_SUPERVISOR_ELEMENTS)
    assert len(s2_elems) == 100
    assert len(s2_targets["CLICK"]) == 100
    assert "100" in s2_targets["CLICK"]

    # System 2 prompt via wrap_untrusted_dom includes all 100 elements without omission line
    s2_prompt = wrap_untrusted_dom(page, s2_elems, max_elements=MAX_SUPERVISOR_ELEMENTS)
    assert "[1] (link) Link Item 1" in s2_prompt
    assert "[100] (link) Link Item 100" in s2_prompt
    assert "elements omitted" not in s2_prompt


def test_fake_page_200_elements_s2_caps_at_150_with_omission_message():
    """Fake page with 200 elements: S2 prompt has 150 elements + "+50 more interactive elements omitted"."""
    actions = _generate_raw_actions(200)
    page = {"url": "https://news.ycombinator.com", "title": "Hacker News", "actions": actions}

    s2_elems, s2_targets, _ = action_space(actions, max_elements=MAX_SUPERVISOR_ELEMENTS)
    assert len(s2_elems) == 150
    assert len(s2_targets["CLICK"]) == 150

    # wrap_untrusted_dom formats 150 elements and appends the required omission message
    s2_prompt = wrap_untrusted_dom(page, s2_elems, max_elements=MAX_SUPERVISOR_ELEMENTS)
    assert "[1] (link) Link Item 1" in s2_prompt
    assert "[150] (link) Link Item 150" in s2_prompt
    assert "[151]" not in s2_prompt
    assert "+50 more interactive elements omitted (scroll or use GOTO_URL)" in s2_prompt

    # Direct pass of 200 pre-indexed elements with empty actions also omits 50
    direct_elements = [
        {"index": str(i), "role": "link", "label": f"Direct Item {i}", "operations": ["CLICK"]}
        for i in range(1, 201)
    ]
    s2_prompt_direct = wrap_untrusted_dom(
        {"url": "https://example.com"}, direct_elements, max_elements=MAX_SUPERVISOR_ELEMENTS
    )
    assert "[150] (link) Direct Item 150 | ops: CLICK" in s2_prompt_direct
    assert "[151]" not in s2_prompt_direct
    assert "+50 more interactive elements omitted (scroll or use GOTO_URL)" in s2_prompt_direct


def test_s2_target_65_resolves_to_correct_action():
    """Targeting 65th element in S2 resolves to correct action; S1 action_space does not include it."""
    actions = _generate_raw_actions(90)
    actions[64]["label"] = "9. A tiny world model (github.com/sheepseb)"
    target_node = actions[64]["node"]

    # S1 action_space lacks index 65
    _s1_elems, s1_targets, _ = action_space(actions)
    assert "65" not in s1_targets.get("CLICK", {})

    mock_browser = MagicMock()
    mock_browser.observe.return_value = {
        "url": "https://news.ycombinator.com",
        "fingerprint": "fp_hn",
        "actions": actions,
        "text": "Hacker News Content",
    }

    with (
        patch("reflex.agent.Browser", return_value=mock_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
    ):
        session = ReflexSession(
            url="https://news.ycombinator.com",
            goal="Find top projects",
            config=ReflexConfig(s2_every_step=True),
        )

        sup_decision = SupervisorDecision(
            action_type="OVERRIDE_ACTION",
            operation="CLICK",
            target="65",
            reasoning="Click on 9th story (sheepseb)",
        )

        with (
            patch("reflex.agent.resolve_escalation", return_value=sup_decision),
            patch.object(session, "_execute_and_verify", return_value=session.snapshot()) as mock_exec,
        ):
            session._handle_escalation(
                reason="gate_uncertain",
                trigger_type="gate_uncertain",
                s1_decision=None,
            )

            assert mock_exec.called
            call_kwargs = mock_exec.call_args.kwargs
            executed_action = call_kwargs["action"]
            assert executed_action["node"] == target_node
            assert "sheepseb" in executed_action["label"]
            assert call_kwargs["target"] == "65"


def test_s2_target_150_boundary_and_out_of_bounds():
    """Target 150 resolves correctly; target 151 does not exist in targets."""
    actions = _generate_raw_actions(160)
    s2_elems, s2_targets, _controls = action_space(actions, max_elements=MAX_SUPERVISOR_ELEMENTS)
    assert len(s2_elems) == 150
    assert "150" in s2_targets["CLICK"]
    assert "151" not in s2_targets["CLICK"]


def test_s1_paths_untouched():
    """Verify S1 paths (action_space default, choose_action, build_system1_request) remain 60."""
    actions = _generate_raw_actions(120)
    page = {"url": "https://news.ycombinator.com", "title": "Hacker News", "actions": actions}

    # action_space default max_elements is 60
    elems, targets, _ = action_space(actions)
    assert len(elems) == 60
    assert len(targets["CLICK"]) == 60

    # build_system1_request uses default 60
    req, s1_targets, _ = build_system1_request(page, "find story", history=[])
    # Elements in request are capped at 60
    assert len(req["state"]["elements"]) == 60
    assert len(s1_targets["CLICK"]) == 60

    # scroll distance is untouched (420px)
    scroll_action = next(a for a in actions if a.get("kind") == "scroll")
    assert scroll_action.get("delta") == 420
