"""Unit tests for Task G12: System 1 radio & SERP cleanup."""
from pathlib import Path
from reflex.system1 import action_space, choose_action
from reflex.config import ReflexConfig


SYSTEM1_PATH = Path(__file__).resolve().parent.parent / "reflex" / "system1.py"


def test_forbidden_heuristics_removed_from_system1():
    """system1.py must not contain answered_radio_groups, [DOLU:, is_serp, or SCROLL_UP suppression."""
    content = SYSTEM1_PATH.read_text(encoding="utf-8")

    forbidden = [
        "answered_radio_groups",
        "[DOLU:",
        "is_serp",
        'controls.pop("SCROLL_UP"',
        "google ana sayfasına git",
    ]
    for pat in forbidden:
        assert pat not in content, f"Forbidden heuristic '{pat}' found in system1.py"


def test_radio_only_checked_option_excluded():
    """Only already-checked radio options are excluded from CLICK targets; sibling options remain clickable."""
    actions = [
        # Question 1: Option A checked, Option B unchecked
        {"id": "r1", "node": 1, "kind": "click", "role": "radio", "label": "Q1 → Option A", "checked": "true"},
        {"id": "r2", "node": 2, "kind": "click", "role": "radio", "label": "Q1 → Option B", "checked": "false"},
        # Question 2: Option C unchecked, Option D unchecked
        {"id": "r3", "node": 3, "kind": "click", "role": "radio", "label": "Q2 → Option C", "checked": "false"},
        {"id": "r4", "node": 4, "kind": "click", "role": "radio", "label": "Q2 → Option D", "checked": "false"},
    ]

    elements, targets, controls = action_space(actions)

    # All 4 elements should be present in elements list for context
    assert len(elements) == 4

    click_targets = targets.get("CLICK", {})
    # r1 (checked=true) must NOT be in CLICK targets
    assert not any(act["id"] == "r1" for act in click_targets.values())

    # r2 (sibling of r1 in Q1, but unchecked) MUST be in CLICK targets
    assert any(act["id"] == "r2" for act in click_targets.values())

    # r3 and r4 must both be in CLICK targets
    assert any(act["id"] == "r3" for act in click_targets.values())
    assert any(act["id"] == "r4" for act in click_targets.values())


def test_scroll_up_not_artificially_suppressed():
    """SCROLL_UP is preserved in controls even after a successful SCROLL_DOWN step."""
    actions = [
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
        {"id": "scroll_up", "kind": "scroll", "label": "Scroll up"},
    ]
    elements, targets, controls = action_space(actions)
    assert "SCROLL_UP" in controls
    assert "SCROLL_DOWN" in controls


def test_serp_targets_not_deleted():
    """Search box and controls are not artificially removed on SERP pages."""
    actions = [
        {"id": "inp1", "node": 10, "kind": "fill", "role": "searchbox", "label": "Search", "value": "cats"},
        {"id": "btn1", "node": 11, "kind": "click", "role": "button", "label": "Clear"},
    ]
    page = {
        "url": "https://www.google.com/search?q=cats",
        "actions": actions,
    }

    elements, targets, controls = action_space(actions)
    assert "TYPE_TEXT" in targets
    assert "CLICK" in targets
    assert any(act["id"] == "inp1" for act in targets["TYPE_TEXT"].values())
    assert any(act["id"] == "btn1" for act in targets["CLICK"].values())
