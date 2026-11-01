"""Unit tests for Reflex G11 Milestone M2 (R2): Package Executor Offscreen & Already Satisfied.

Verifies:
1. Browser.scroll_to_node method (instant center alignment evaluate, return types, error handling)
2. ExpectedElement & create_action_package initial_checked and desired_checked state capture:
   - radio -> desired_checked=True
   - checkbox / switch -> desired_checked=not initial_checked
   - buttons / text -> desired_checked=None
3. ActionPackage.consecutive_no_progress field and PackageStep property accessors
4. Already-satisfied skipping in _execute_package_step:
   - Radio already checked -> skipped without S1 call or click
   - Checkbox in desired state -> skipped
   - Element in page_controls already satisfied -> skipped without scroll
   - package_steps_skipped metric and history record
5. Offscreen target handling (k >= 2):
   - Element missing from actions but in page_controls -> scroll_to_node -> re-observe -> S1 grounding
   - package_offscreen_scrolls metric
   - Target still missing after scroll -> aborts with package_target_missing
   - Unsettled page after scroll -> aborts with page_unsettled
6. no_progress tolerance in _finish_package_step:
   - 1st no_progress tolerated when next target is in actions
   - 1st no_progress tolerated when next target is in page_controls
   - 2 consecutive no_progress aborts package
   - 1st no_progress on missing next target aborts
   - 1st no_progress on last step aborts
   - Successful progress step resets consecutive_no_progress counter
7. Preserved abort rules (url_changed, loop, oscillation, s1_error)
8. Session metrics snapshot serialization
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.browser.cdp import Browser
from reflex.config import ReflexConfig
from reflex.package import (
    ActionPackage,
    ExpectedElement,
    PackageStep,
    create_action_package,
    parse_checked_state,
)
from reflex.supervisor import ActionPackageItem, SupervisorDecision
from reflex.verifier import VerificationResult

# ============================================================================
# Helpers
# ============================================================================


def _make_mock_browser(page: dict[str, Any]) -> MagicMock:
    mock_b = MagicMock()
    mock_b.observe.return_value = page
    mock_b.fresh.return_value = True
    mock_b.act.return_value = {"rect": {"x": 10, "y": 10, "width": 50, "height": 20}}
    mock_b.scroll_to_node.return_value = True
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


def _no_progress_verify() -> VerificationResult:
    return VerificationResult(
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


# ============================================================================
# 1. Browser.scroll_to_node Unit Tests
# ============================================================================


def test_scroll_to_node_valid_node_returns_true() -> None:
    """scroll_to_node executes instant center scrollIntoView and returns True on success."""
    browser = Browser.__new__(Browser)
    browser.evaluate = MagicMock(return_value=True)

    result = browser.scroll_to_node(42)

    assert result is True
    browser.evaluate.assert_called_once()
    expr = browser.evaluate.call_args[0][0]
    assert "window.__jevFast?.nodes?.get(42)" in expr
    assert "behavior: 'instant'" in expr
    assert "block: 'center'" in expr


def test_scroll_to_node_missing_or_disconnected_returns_false() -> None:
    """scroll_to_node returns False when node is missing or disconnected in DOM."""
    browser = Browser.__new__(Browser)
    browser.evaluate = MagicMock(return_value=False)

    result = browser.scroll_to_node(999)

    assert result is False


def test_scroll_to_node_invalid_argument_types() -> None:
    """scroll_to_node returns False without evaluation when node is not an int."""
    browser = Browser.__new__(Browser)
    browser.evaluate = MagicMock()

    assert browser.scroll_to_node("42") is False  # type: ignore[arg-type]
    assert browser.scroll_to_node(None) is False  # type: ignore[arg-type]
    assert browser.scroll_to_node(3.14) is False  # type: ignore[arg-type]
    browser.evaluate.assert_not_called()


def test_scroll_to_node_exception_shielding() -> None:
    """scroll_to_node catches evaluation exceptions and returns False cleanly."""
    browser = Browser.__new__(Browser)
    browser.evaluate = MagicMock(side_effect=RuntimeError("CDP connection failed"))

    result = browser.scroll_to_node(10)
    assert result is False


# ============================================================================
# 2. Checked State Parsing & Package Creation Tests
# ============================================================================


def test_parse_checked_state_matrix() -> None:
    """parse_checked_state parses booleans, strings, numbers, and nulls."""
    assert parse_checked_state(True) is True
    assert parse_checked_state(False) is False
    assert parse_checked_state("true") is True
    assert parse_checked_state("True") is True
    assert parse_checked_state("false") is False
    assert parse_checked_state("False") is False
    assert parse_checked_state("1") is True
    assert parse_checked_state("0") is False
    assert parse_checked_state(1) is True
    assert parse_checked_state(0) is False
    assert parse_checked_state(None) is None
    assert parse_checked_state("") is None
    assert parse_checked_state("mixed") is None


def test_create_action_package_radio_desired_state() -> None:
    """CLICK targets with role 'radio' always have desired_checked=True."""
    targets = {
        "CLICK": {
            "1": {"id": "1", "node": 101, "kind": "click", "role": "radio", "label": "Option A", "checked": "false"},
            "2": {"id": "2", "node": 102, "kind": "click", "role": "menuitemradio", "label": "Option B", "checked": "true"},
        }
    }
    raw_actions = [
        {"operation": "CLICK", "target": "1", "instruction": "Select Option A"},
        {"operation": "CLICK", "target": "2", "instruction": "Select Option B"},
    ]
    pkg = create_action_package(raw_actions, {"url": "https://example.com"}, targets)
    assert pkg is not None
    assert pkg.consecutive_no_progress == 0

    step1 = pkg.steps[0]
    assert step1.initial_checked is False
    assert step1.desired_checked is True
    assert step1.expected.desired_checked is True

    step2 = pkg.steps[1]
    assert step2.initial_checked is True
    assert step2.desired_checked is True


def test_create_action_package_checkbox_toggle_desired_state() -> None:
    """CLICK targets with role 'checkbox' or 'switch' desire the opposite of their initial state."""
    targets = {
        "CLICK": {
            "1": {"id": "1", "node": 201, "kind": "click", "role": "checkbox", "label": "Terms", "checked": "false"},
            "2": {"id": "2", "node": 202, "kind": "click", "role": "switch", "label": "Dark Mode", "checked": "true"},
            "3": {"id": "3", "node": 203, "kind": "click", "role": "checkbox", "label": "Newsletter"},  # unchecked, no checked attr
        }
    }
    raw_actions = [
        {"operation": "CLICK", "target": "1", "instruction": "Agree Terms"},
        {"operation": "CLICK", "target": "2", "instruction": "Toggle Dark Mode"},
        {"operation": "CLICK", "target": "3", "instruction": "Subscribe Newsletter"},
    ]
    pkg = create_action_package(raw_actions, {"url": "https://example.com"}, targets)
    assert pkg is not None

    # Step 1: initially False -> desires True
    assert pkg.steps[0].initial_checked is False
    assert pkg.steps[0].desired_checked is True

    # Step 2: initially True -> desires False
    assert pkg.steps[1].initial_checked is True
    assert pkg.steps[1].desired_checked is False

    # Step 3: initially None -> desires True
    assert pkg.steps[2].initial_checked is None
    assert pkg.steps[2].desired_checked is True


def test_create_action_package_button_has_none_desired_checked() -> None:
    """Non-choice CLICK targets (buttons, links) have desired_checked=None."""
    targets = {
        "CLICK": {
            "1": {"id": "1", "node": 301, "kind": "click", "role": "button", "label": "Submit"},
            "2": {"id": "2", "node": 302, "kind": "click", "role": "link", "label": "Next"},
        }
    }
    raw_actions = [
        {"operation": "CLICK", "target": "1", "instruction": "Click Submit"},
        {"operation": "CLICK", "target": "2", "instruction": "Click Next"},
    ]
    pkg = create_action_package(raw_actions, {"url": "https://example.com"}, targets)
    assert pkg is not None
    assert pkg.steps[0].desired_checked is None
    assert pkg.steps[1].desired_checked is None


def test_package_step_properties_delegate_to_expected() -> None:
    """PackageStep.initial_checked and desired_checked delegate to ExpectedElement."""
    expected = ExpectedElement(
        node=1,
        label="Radio",
        role="radio",
        operation="CLICK",
        initial_checked=False,
        desired_checked=True,
    )
    step = PackageStep(
        step_index=1,
        operation="CLICK",
        target="1",
        text=None,
        instruction="Click Radio",
        expected=expected,
    )
    assert step.initial_checked is False
    assert step.desired_checked is True


def test_action_package_consecutive_no_progress_initial_zero() -> None:
    """ActionPackage initializes consecutive_no_progress to 0."""
    pkg = ActionPackage(id="pkg_test", initial_url="https://example.com", steps=[])
    assert pkg.consecutive_no_progress == 0


# ============================================================================
# 3. Already Satisfied State Skipping Tests
# ============================================================================


def test_executor_skips_already_satisfied_radio_step() -> None:
    """When a package step targets an already-checked radio, skip click and do not invoke S1 or S2."""
    page = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Start"},
            {"id": "rad_1", "node": 102, "kind": "click", "role": "radio", "label": "Option 1", "checked": "true"},
            {"id": "btn_sub", "node": 103, "kind": "click", "role": "button", "label": "Submit"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Fill form",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Start"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Select Option 1"),
            ActionPackageItem(operation="CLICK", target="3", instruction="Click Submit"),
        ],
    )
    s1_resp = {"raw_answers": {"click_target": {"choice": "3"}}, "confidence": 0.95}

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision) as mock_sup,
        patch("reflex.agent.ground_target", return_value=s1_resp) as mock_ground,
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/form", goal="Form", config=cfg)

        # Step 1: Supervisor executes btn_1
        session.step()
        assert session.packages_started == 1
        assert session.package_steps_executed == 1
        assert session.package_steps_skipped == 0
        assert mock_sup.call_count == 1

        # Step 2: rad_1 is already checked ("true"). It must be skipped!
        session.step()
        assert session.package_steps_skipped == 1
        assert session.package_steps_executed == 1  # not incremented on skip

        # Verify history entry
        skip_record = session.history[-1]
        assert skip_record["actor"] == "package"
        assert skip_record["package"]["skipped"] == "already_satisfied"
        assert skip_record["package"]["step_index"] == 2
        assert skip_record["action"]["node"] == 102

        # Step 3: btn_sub executes normally
        session.step()
        assert session.package_steps_executed == 2
        assert session.packages_completed == 1
        assert session.active_package is None
        assert mock_sup.call_count == 1  # No escalation to S2 occurred!

        # ground_target was called for step 1 probe and step 3 execution, NEVER for step 2!
        assert mock_ground.call_count == 2
        assert mock_ground.call_args_list[1].args[1] == "Click Submit"


def test_executor_skips_already_satisfied_checkbox_step() -> None:
    """When a checkbox is in the desired toggled state, skip click."""
    # Checkbox initially unchecked (checked="false"). S2 package wants to toggle it (desired=True).
    # But after step 1 re-render, the checkbox is already checked.
    page_initial = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Expand"},
            {"id": "chk_1", "node": 102, "kind": "click", "role": "checkbox", "label": "Accept Terms", "checked": "false"},
        ],
    }
    page_after_step1 = {
        "url": "https://example.com/form",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Expand"},
            {"id": "chk_1", "node": 102, "kind": "click", "role": "checkbox", "label": "Accept Terms", "checked": "true"},
        ],
    }
    mock_b = _make_mock_browser(page_initial)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Agree terms",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Expand"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Check Accept Terms"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/form", goal="Form", config=cfg)
        session.step()

        # Update browser observe to return page_after_step1 where chk_1 is already checked
        mock_b.observe.return_value = page_after_step1

        # Step 2: chk_1 is already in desired state (True) -> skipped!
        session.step()
        assert session.package_steps_skipped == 1
        assert session.packages_completed == 1
        assert session.active_package is None
        assert session.history[-1]["package"]["skipped"] == "already_satisfied"


def test_executor_skips_already_satisfied_target_in_page_controls() -> None:
    """When an offscreen target in page_controls is already satisfied, skip without scrolling."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Start"},
        ],
        "page_controls": [
            {"node": 202, "kind": "click", "role": "radio", "label": "Question 2 Choice A", "checked": "true"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Start"),
            ActionPackageItem(operation="CLICK", target="f1", instruction="Select Choice A"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()  # Step 1

        # Step 2: f1 is in page_controls with checked="true". Desired for radio is True.
        # It must skip without calling scroll_to_node!
        session.step()
        assert session.package_steps_skipped == 1
        assert session.package_offscreen_scrolls == 0
        mock_b.scroll_to_node.assert_not_called()
        assert session.packages_completed == 1


# ============================================================================
# 4. Offscreen Target Scrolling Tests (k >= 2)
# ============================================================================


def test_executor_offscreen_target_scrolls_and_executes() -> None:
    """When expected target is missing from actions but in page_controls, scroll, re-observe, and ground."""
    # Step 1: on screen
    # Step 2: node 202 is offscreen initially (in page_controls, not in actions)
    page_step1 = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Question 1"},
        ],
        "page_controls": [
            {"node": 202, "kind": "click", "role": "radio", "label": "Question 2 Choice B", "checked": "false"},
        ],
    }
    # After scrolling to node 202, node 202 is now visible in actions!
    page_after_scroll = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "rad_q2", "node": 202, "kind": "click", "role": "radio", "label": "Question 2 Choice B", "checked": "false"},
        ],
        "page_controls": [],
    }

    mock_b = _make_mock_browser(page_step1)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Q1"),
            ActionPackageItem(operation="CLICK", target="f1", instruction="Select Q2 Choice B"),
        ],
    )
    s1_resp = {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.92}

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value=s1_resp),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()  # Step 1: Supervisor executes Q1 (btn_1)

        # When scroll_to_node is called, scrolling brings node 202 into actions for subsequent observe
        def on_scroll(node: int) -> bool:
            mock_b.observe.return_value = page_after_scroll
            return True
        mock_b.scroll_to_node.side_effect = on_scroll

        # Step 2: triggers scroll_to_node(202) -> re-observes -> S1 grounds rad_q2 -> executes
        session.step()

        mock_b.scroll_to_node.assert_called_once_with(202)
        assert session.package_offscreen_scrolls == 1
        assert session.package_steps_executed == 2
        assert session.packages_completed == 1
        assert session.active_package is None

        # Verify metric in snapshot
        metrics = session.snapshot()["metrics"]
        assert metrics["package_offscreen_scrolls"] == 1
        assert metrics["package_steps_skipped"] == 0


def test_executor_offscreen_target_still_missing_after_scroll_aborts() -> None:
    """When target remains missing from actions after scroll_to_node, abort package with package_target_missing."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Question 1"},
        ],
        "page_controls": [
            {"node": 202, "kind": "click", "role": "radio", "label": "Question 2 Choice B", "checked": "false"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Q1"),
            ActionPackageItem(operation="CLICK", target="f1", instruction="Select Q2 Choice B"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Recovery after missing target",
        actions=[],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=_ok_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()  # Step 1

        # Step 2: scroll_to_node called, but observe still returns page without node 202 in actions!
        session.step()

        mock_b.scroll_to_node.assert_called_once_with(202)
        assert session.package_offscreen_scrolls == 1
        assert session.active_package is None
        assert session.package_aborts["package_target_missing"] == 1
        assert mock_sup.call_count == 2
        assert mock_sup.call_args_list[1].kwargs["trigger_type"] == "package_target_missing"


# ============================================================================
# 5. no_progress Tolerance Tests
# ============================================================================


def test_executor_tolerates_single_no_progress_when_next_target_in_actions() -> None:
    """1st no_progress is tolerated and advances package when next target is present in actions."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 choices",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )
    s1_resp = {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.9}

    # Step 1 has no_progress; Step 2 has progress
    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision) as mock_sup,
        patch("reflex.agent.ground_target", return_value=s1_resp),
        patch("reflex.agent.verify_step", side_effect=[_no_progress_verify(), _ok_verify()]),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)

        # Step 1 saw no_progress, but Choice 2 is on page: tolerated!
        session.step()
        assert session.active_package is not None
        assert session.active_package.consecutive_no_progress == 1
        assert session.active_package.current_step_index == 2
        assert mock_sup.call_count == 1  # No escalation!

        # Step 2 succeeds with progress: resets consecutive_no_progress and completes package!
        session.step()
        assert session.packages_completed == 1
        assert session.active_package is None
        assert session.package_aborts["no_progress"] == 0


def test_executor_tolerates_single_no_progress_when_next_target_in_page_controls() -> None:
    """1st no_progress is tolerated when next target is offscreen in page_controls."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
        ],
        "page_controls": [
            {"node": 202, "kind": "click", "role": "radio", "label": "Choice 2", "checked": "false"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="f1", instruction="Select Choice 2"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=_no_progress_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()

        # Step 1 had no_progress, next target 202 is in page_controls -> tolerated!
        assert session.active_package is not None
        assert session.active_package.consecutive_no_progress == 1
        assert session.active_package.current_step_index == 2


def test_executor_two_consecutive_no_progress_aborts() -> None:
    """Two consecutive no_progress steps abort the package."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
            {"id": "btn_3", "node": 103, "kind": "click", "role": "button", "label": "Choice 3"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision1 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="3 choices",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
            ActionPackageItem(operation="CLICK", target="3", instruction="Click Choice 3"),
        ],
    )
    decision2 = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="After 2 no progress",
        actions=[],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", side_effect=[decision1, decision2]) as mock_sup,
        patch("reflex.agent.ground_target", side_effect=[
            {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
            {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.9},
        ]),
        patch("reflex.agent.verify_step", return_value=_no_progress_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)

        # Step 1: no_progress (tolerated, count=1)
        session.step()
        assert session.active_package is not None
        assert session.active_package.consecutive_no_progress == 1

        # Step 2: 2nd consecutive no_progress -> aborts package!
        session.step()
        assert session.active_package is None
        assert session.package_aborts["no_progress"] == 1
        assert session.pending_escalation_reason[0] == "no_progress"

        # Step 3: escalates to S2
        session.step()
        assert mock_sup.call_count == 2
        assert mock_sup.call_args_list[1].kwargs["trigger_type"] == "no_progress"


def test_executor_no_progress_aborts_immediately_if_next_target_missing() -> None:
    """When a step produces no_progress and the next target is missing from the page, abort immediately."""
    page_step1 = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page_step1)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 choices",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=_no_progress_verify()),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)

        # Before step 1 completes, update observe to return page without node 102
        mock_b.observe.return_value = {
            "url": "https://example.com/quiz",
            "actions": [{"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"}],
            "page_controls": [],
        }
        session.step()

        # Step 1 saw no_progress, and next target (node 102) is NOT on page -> aborts immediately!
        assert session.active_package is None
        assert session.package_aborts["no_progress"] == 1


def test_executor_no_progress_on_last_step_aborts() -> None:
    """When the final step of a package produces no_progress, abort with no_progress."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="2 choices",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", side_effect=[
            {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
            {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.9},
        ]),
        patch("reflex.agent.verify_step", side_effect=[_ok_verify(), _no_progress_verify()]),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()  # Step 1: progress ok
        assert session.active_package.consecutive_no_progress == 0

        # Step 2: final step produces no_progress -> aborts package!
        session.step()
        assert session.active_package is None
        assert session.package_aborts["no_progress"] == 1


def test_executor_progress_resets_consecutive_no_progress() -> None:
    """A successful step with progress resets consecutive_no_progress to 0."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
            {"id": "btn_3", "node": 103, "kind": "click", "role": "button", "label": "Choice 3"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="3 choices",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
            ActionPackageItem(operation="CLICK", target="3", instruction="Click Choice 3"),
        ],
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", side_effect=[
            {"raw_answers": {"click_target": {"choice": "1"}}, "confidence": 0.9},
            {"raw_answers": {"click_target": {"choice": "2"}}, "confidence": 0.9},
            {"raw_answers": {"click_target": {"choice": "3"}}, "confidence": 0.9},
        ]),
        # Step 1: no progress (tolerated); Step 2: progress ok (resets); Step 3: progress ok (completes)
        patch("reflex.agent.verify_step", side_effect=[_no_progress_verify(), _ok_verify(), _ok_verify()]),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)

        session.step()  # Step 1: no progress -> tolerated, counter=1
        assert session.active_package.consecutive_no_progress == 1

        session.step()  # Step 2: progress -> counter reset to 0!
        assert session.active_package.consecutive_no_progress == 0

        session.step()  # Step 3: completes
        assert session.packages_completed == 1
        assert session.active_package is None


# ============================================================================
# 6. Preserved Strict Abort Rules Tests
# ============================================================================


def test_executor_strict_url_change_aborts_immediately() -> None:
    """URL change still aborts immediately at step 1 or step k."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )
    url_changed_verify = VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=True,
        element_state_changed=False,
        effective_progress=True,
        loop_detected=False,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="Navigated",
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=url_changed_verify),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()

        # Step 1 caused url_changed on non-last step -> aborts immediately!
        assert session.active_package is None
        assert session.package_aborts["url_changed"] == 1
        assert session.pending_escalation_reason[0] == "url_changed"


def test_executor_strict_loop_and_oscillation_abort() -> None:
    """Loop and oscillation detection abort immediately without tolerance."""
    page = {
        "url": "https://example.com/quiz",
        "actions": [
            {"id": "btn_1", "node": 101, "kind": "click", "role": "button", "label": "Choice 1"},
            {"id": "btn_2", "node": 102, "kind": "click", "role": "button", "label": "Choice 2"},
        ],
    }
    mock_b = _make_mock_browser(page)
    cfg = ReflexConfig(s2_action_packages=True, max_steps=10)

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Quiz",
        actions=[
            ActionPackageItem(operation="CLICK", target="1", instruction="Click Choice 1"),
            ActionPackageItem(operation="CLICK", target="2", instruction="Click Choice 2"),
        ],
    )
    loop_verify = VerificationResult(
        executed_ok=True,
        page_changed=True,
        url_changed=False,
        element_state_changed=False,
        effective_progress=True,
        loop_detected=True,
        oscillation_detected=False,
        scroll_budget_exceeded=False,
        reason="Loop",
    )

    with (
        patch("reflex.agent.Browser", return_value=mock_b),
        patch("reflex.agent.interpret_user_message", return_value=_make_mock_interp()),
        patch("reflex.agent.resolve_escalation", return_value=decision),
        patch("reflex.agent.ground_target", return_value={"raw_answers": {}, "confidence": 0.9}),
        patch("reflex.agent.verify_step", return_value=loop_verify),
    ):
        session = ReflexSession(url="https://example.com/quiz", goal="Quiz", config=cfg)
        session.step()

        assert session.active_package is None
        assert session.package_aborts["loop_detected"] == 1


# ============================================================================
# 7. Metrics Exporter Snapshot Verification
# ============================================================================


def test_metrics_snapshot_contains_r2_counters() -> None:
    """session.snapshot()['metrics'] includes package_steps_skipped and package_offscreen_scrolls."""
    cfg = ReflexConfig(s2_action_packages=True)
    session = ReflexSession(url="https://example.com", goal="Test", config=cfg)

    session.package_steps_skipped = 5
    session.package_offscreen_scrolls = 3

    metrics = session.snapshot()["metrics"]
    assert metrics["package_steps_skipped"] == 5
    assert metrics["package_offscreen_scrolls"] == 3
