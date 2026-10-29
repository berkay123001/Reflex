"""Adversarial Empirical Stress Test Suite for Milestone M3 (Progress Ledger & Milestone Triggers).

Tests:
1. URL & domain normalization fuzzing, rapid domain changes, malformed/schemeless URLs,
   ports, case variations, subdomains, special browser schemes, IPv6, null bytes, IDN.
2. Step counter boundary transitions: exactly 11, 12, 13 steps, S2 intervening at step 5,
   S2 intervening at step 12, S2 actor execution reset.
3. Subgoal completion variations: empty subgoals, 1 subgoal, N subgoals, all subgoals completed,
   signal sources (S1 DONE, verifier details, page criteria).
4. Stuck counter oscillation: stuck -> not stuck -> stuck -> stuck -> not stuck,
   stuck >= 3 recovery hint application, tactical hint clearing on effective progress.
5. Multiple concurrent milestone triggers, simple task immunity, noop escalation suppression.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.planner import (
    Subgoal,
    SubgoalResult,
    TaskLedger,
    _extract_domain,
    parse_task_ledger,
)
from reflex.supervisor import SupervisorDecision
from reflex.verifier import VerificationResult

# ==============================================================================
# Helper Fixtures & Builders
# ==============================================================================


def _build_test_agent(
    plan: TaskLedger | None = None,
    initial_url: str = "https://www.example.com",
) -> ReflexAgent:
    """Instantiate a ReflexAgent with mocked browser and message interpreter."""
    with (
        patch("reflex.agent.Browser") as mock_browser_cls,
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_browser = MagicMock()
        mock_browser_cls.return_value = mock_browser
        mock_browser.observe.return_value = {
            "url": initial_url,
            "title": "Example Page",
            "text": "Example text content",
            "full_text": "Example text content for testing",
            "headings": ["Example Heading"],
            "actions": [
                {"id": "act_1", "kind": "click", "label": "Next", "node": 1, "rect": [10, 10, 50, 20]},
                {"id": "act_2", "kind": "type", "label": "Search", "node": 2, "rect": [10, 40, 100, 20]},
            ],
            "fingerprint": "fp_init",
        }
        mock_browser.fresh.return_value = True

        mock_interp_res = MagicMock()
        mock_interp_res.model_dump.return_value = {}
        mock_interp_res.allow_arbitrary_choices = False
        mock_interp_res.constraints = []
        mock_interp_res.plan = plan
        mock_interp.return_value = mock_interp_res

        config = ReflexConfig(max_steps=50, screenshots=False)
        agent = ReflexAgent("Test adversarial goal", initial_url, config=config)
        agent.browser = mock_browser
        return agent


def _multi_subgoal_plan(count: int = 3) -> TaskLedger:
    """Generate a valid TaskLedger with N subgoals."""
    subgoals = [
        {
            "id": f"sg_{i}",
            "text": f"Subgoal {i} description",
            "done_when": f"Criteria for subgoal {i}",
            "status": "active" if i == 1 else "pending",
            "results": [],
        }
        for i in range(1, count + 1)
    ]
    raw = {
        "needs_plan": True,
        "finish_criteria": "All subgoals completed",
        "budget": {"sources": 2, "max_steps": 30},
        "subgoals": subgoals,
    }
    return parse_task_ledger(raw)


# ==============================================================================
# 1. URL & Domain Fuzzing & Rapid Domain Changes
# ==============================================================================


@pytest.mark.parametrize(
    ("url_input", "expected_domain"),
    [
        # Standard URLs
        ("https://www.example.com", "example.com"),
        ("http://example.com/path/to/page", "example.com"),
        ("https://sub.domain.co.uk", "sub.domain.co.uk"),
        ("https://www.sub.domain.co.uk", "sub.domain.co.uk"),
        # Subdomains
        ("https://api.example.com", "api.example.com"),
        ("https://auth.api.example.com", "auth.api.example.com"),
        ("https://www.api.example.com", "api.example.com"),
        # Case variations
        ("HTTPS://WWW.EXAMPLE.COM/PATH", "example.com"),
        ("http://ExAmPlE.cOm:8080/Foo", "example.com"),
        ("hTTps://WwW.SuB.ExAmPlE.Com", "sub.example.com"),
        # Port variations (same domain should be extracted)
        ("http://example.com:80", "example.com"),
        ("https://example.com:443", "example.com"),
        ("http://example.com:8080", "example.com"),
        ("http://example.com:9000/api", "example.com"),
        # Schemeless URLs
        ("example.com", "example.com"),
        ("www.example.com", "example.com"),
        ("example.com/wiki/test", "example.com"),
        ("api.service.io:3000/v1/health", "api.service.io"),
        # Protocol-relative URLs
        ("//example.com/test", "example.com"),
        ("//www.example.com/test", "example.com"),
        # Special browser pseudo-schemes -> must return empty string
        ("about:blank", ""),
        ("about:config", ""),
        ("javascript:void(0)", ""),
        ("javascript:alert(1)", ""),
        ("chrome://settings", ""),
        ("data:text/html,<html></html>", ""),
        ("blob:https://example.com/uuid-1234", ""),
        ("file:///tmp/test.html", ""),
        # Malformed / empty / degenerate inputs
        ("", ""),
        ("   ", ""),
        (None, ""),
        ("http://", ""),
        ("https://", ""),
        ("://", ""),
        ("///", ""),
        ("/relative/path/only", ""),
        ("?query=param", ""),
        ("#hash_only", ""),
        # IPv4 and IPv6
        ("http://127.0.0.1:8000/test", "127.0.0.1"),
        ("http://[::1]:8080/test", "::1"),
        # Malformed IPv6 (must NOT raise uncaught exception)
        ("http://[invalid-ipv6/path", "http"),
        # Query parameters containing nested URLs
        ("https://example.com/redirect?target=https://malicious.org", "example.com"),
        # Whitespace padding
        ("   https://www.example.com/path   ", "example.com"),
    ],
)
def test_extract_domain_fuzzing(url_input: Any, expected_domain: str):
    """_extract_domain must handle malformed, special, schemeless, and diverse URLs robustly."""
    domain = _extract_domain(url_input)
    assert domain == expected_domain, f"Failed for input {url_input!r}: expected {expected_domain!r}, got {domain!r}"


def test_rapid_domain_changes_sequence():
    """Verify sequence of rapid domain changes correctly sets or suppresses site_changed milestone."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan, initial_url="https://www.google.com/search?q=test")

    action = {"id": "act_1", "kind": "click", "label": "Navigate", "node": 1}
    act_res = {"success": True}

    url_transitions = [
        # (before, after, expect_trigger, reason_substr)
        ("https://www.google.com/search?q=1", "https://www.google.com/search?q=2", False, None),
        ("https://www.google.com/search?q=2", "https://en.wikipedia.org/wiki/Ada", True, "Site changed from google.com to en.wikipedia.org"),
        ("https://en.wikipedia.org/wiki/Ada", "https://en.wikipedia.org/wiki/Ada#Early_life", False, None),
        ("https://en.wikipedia.org/wiki/Ada", "about:blank", False, None),  # browser internal -> empty domain -> no trigger
        ("about:blank", "https://en.wikipedia.org/wiki/Babbage", False, None),  # before empty -> no trigger
        ("https://en.wikipedia.org/wiki/Babbage", "https://api.wikipedia.org/v1", True, "Site changed from en.wikipedia.org to api.wikipedia.org"),
        ("https://api.wikipedia.org/v1", "http://api.wikipedia.org:8080/v2", False, None),  # port variation, same domain -> no trigger
        ("http://api.wikipedia.org:8080/v2", "HTTPS://API.WIKIPEDIA.ORG/v3", False, None),  # case variation, same domain -> no trigger
    ]

    for before_url, after_url, expect_trigger, reason_substr in url_transitions:
        agent.pending_escalation_reason = None
        before_page = {"url": before_url, "fingerprint": "fp_b", "actions": [action]}
        after_page = {"url": after_url, "fingerprint": "fp_a", "actions": [action]}
        verification = VerificationResult(
            executed_ok=True,
            effective_progress=True,
            page_changed=True,
            url_changed=before_url != after_url,
            reason="Navigated",
        )

        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=after_page),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = before_page
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target="Next",
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=50,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )

        if expect_trigger:
            assert agent.pending_escalation_reason is not None, f"Expected milestone trigger for {before_url} -> {after_url}"
            trig_type, reason = agent.pending_escalation_reason
            assert trig_type == "progress_milestone"
            assert reason_substr in reason
        else:
            assert agent.pending_escalation_reason is None, f"Unexpected trigger for {before_url} -> {after_url}: {agent.pending_escalation_reason}"


# ==============================================================================
# 2. S1 Step Counter Boundaries (11, 12, 13 steps & S2 Interventions)
# ==============================================================================


def test_s1_step_counter_boundaries_11_12_13():
    """Verify strict boundary behavior at 11, 12, and 13 steps."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    action = {"id": "act_1", "kind": "click", "label": "Button", "node": 1}
    act_res = {"success": True}
    page_state = {"url": "https://www.example.com/test", "fingerprint": "fp_1", "actions": [action]}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="Progress")

    # Run 11 consecutive S1 steps
    for step_i in range(1, 12):
        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=page_state),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = page_state
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=10,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )
        assert agent._s1_steps_since_s2 == step_i
        if step_i < 12:
            assert agent.pending_escalation_reason is None

    # After 11 steps, counter is 11, pending_escalation_reason is None
    assert agent._s1_steps_since_s2 == 11
    assert agent.pending_escalation_reason is None

    # Step 12 execution
    with (
        patch.object(agent.browser, "act", return_value=act_res),
        patch.object(agent.browser, "observe", return_value=page_state),
        patch("reflex.agent.verify_step", return_value=verification),
    ):
        agent.page = page_state
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.5,
            latency_ms=10,
            model_name="Turnike-Web",
            usage={},
            preset_text=None,
            escalation_note=None,
            step_t0=0.0,
            s1_ms=10,
            s2_ms=0,
            pre_obs_ms=5,
        )

    # Exactly at 12 steps: counter is 12 and pending_escalation_reason is set
    assert agent._s1_steps_since_s2 == 12
    assert agent.pending_escalation_reason is not None
    trig, reason = agent.pending_escalation_reason
    assert trig == "progress_milestone"
    assert "12 consecutive S1 steps without S2" in reason

    # Calling step() now resolves escalation via S2 and resets counter to 0
    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
    )
    with (
        patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision),
        patch.object(agent.browser, "observe", return_value=page_state),
    ):
        snap = agent.step()
        assert snap["status"] == "ready"
        assert agent._s1_steps_since_s2 == 0
        assert agent.pending_escalation_reason is None


def test_s1_step_counter_fail_safe_at_13_steps():
    """Verify that if _s1_steps_since_s2 is >= 12 when step() starts, step() intercepts immediately."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    # Artificially set counter to 13 steps without pending_escalation_reason
    agent._s1_steps_since_s2 = 13
    agent.pending_escalation_reason = None

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision) as mock_resolve:
        snap = agent.step()
        assert snap["status"] == "ready"
        assert mock_resolve.called
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["trigger_type"] == "progress_milestone"
        assert "12 consecutive S1 steps without S2" in call_kwargs["escalation_reason"]
        assert agent._s1_steps_since_s2 == 0


def test_s1_step_counter_s2_intervening_at_step_5():
    """When S2 intervenes at step 5, counter resets to 0 and requires 12 new S1 steps to trigger."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    page_state = {"url": "https://www.example.com/test", "fingerprint": "fp_1", "actions": [action]}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="OK")

    # Run 5 S1 steps
    for _ in range(5):
        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=page_state),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = page_state
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=10,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )
    assert agent._s1_steps_since_s2 == 5

    # S2 intervenes at step 5
    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        agent._handle_escalation(
            trigger_type="test_trigger",
            reason="Intervention at step 5",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )

    # Counter must be reset to 0
    assert agent._s1_steps_since_s2 == 0

    # Run 11 more S1 steps (session step 6 through 16): no milestone trigger
    for _ in range(11):
        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=page_state),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = page_state
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=10,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )
    assert agent._s1_steps_since_s2 == 11
    assert agent.pending_escalation_reason is None

    # Step 12 since S2 (session step 17)
    with (
        patch.object(agent.browser, "act", return_value=act_res),
        patch.object(agent.browser, "observe", return_value=page_state),
        patch("reflex.agent.verify_step", return_value=verification),
    ):
        agent.page = page_state
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.5,
            latency_ms=10,
            model_name="Turnike-Web",
            usage={},
            preset_text=None,
            escalation_note=None,
            step_t0=0.0,
            s1_ms=10,
            s2_ms=0,
            pre_obs_ms=5,
        )
    assert agent._s1_steps_since_s2 == 12
    assert agent.pending_escalation_reason is not None
    assert agent.pending_escalation_reason[0] == "progress_milestone"


def test_s1_step_counter_s2_intervening_at_step_12():
    """When S2 intervenes at step 12 via an external reason, counter resets cleanly."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    page_state = {"url": "https://www.example.com/test", "fingerprint": "fp_1", "actions": [action]}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="OK")

    # Run 11 steps
    for _ in range(11):
        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=page_state),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = page_state
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=10,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )
    assert agent._s1_steps_since_s2 == 11

    # At step 12, S2 intervenes directly (e.g. gate escalation)
    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        agent._handle_escalation(
            trigger_type="gate_escalation",
            reason="Escalated at step 12",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
    assert agent._s1_steps_since_s2 == 0


def test_s2_actor_execution_resets_step_counter():
    """When _execute_and_verify executes with actor='S2', counter resets to 0."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)
    agent._s1_steps_since_s2 = 8

    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    page_state = {"url": "https://www.example.com/test", "fingerprint": "fp_1", "actions": [action]}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="OK")

    with (
        patch.object(agent.browser, "act", return_value=act_res),
        patch.object(agent.browser, "observe", return_value=page_state),
        patch("reflex.agent.verify_step", return_value=verification),
    ):
        agent.page = page_state
        agent._execute_and_verify(
            action=action,
            actor="S2",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.5,
            latency_ms=10,
            model_name="Supervisor",
            usage={},
            preset_text=None,
            escalation_note="S2 manual override",
            step_t0=0.0,
            s1_ms=0,
            s2_ms=100,
            pre_obs_ms=5,
        )

    assert agent._s1_steps_since_s2 == 0


# ==============================================================================
# 3. Subgoal Completion Matrix (Empty, 1 Subgoal, N Subgoals, All Completed)
# ==============================================================================


def test_subgoal_completion_empty_subgoals():
    """Verify behavior when TaskLedger has empty subgoals list."""
    # 1. Raw TaskLedger with strictly empty subgoals
    raw_plan = TaskLedger(needs_plan=True, subgoals=[])
    agent = _build_test_agent(plan=raw_plan)

    assert agent.ctx.plan.get_active_subgoal() is None
    assert agent.ctx.plan.advance_subgoal() is None
    assert agent.ctx.plan.subgoals == []

    # Verification helper returns False without throwing
    res = agent._is_subgoal_completion_signaled(None, None, {})
    assert res is False

    # Progress check with subgoal_done=True on empty subgoals does not crash
    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=True,
        next_subgoal=None,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        snap = agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Progress check on empty subgoals",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
        assert snap["status"] == "ready"

    # 2. Parsed TaskLedger with empty subgoals list heals into a default subgoal
    parsed_plan = parse_task_ledger({"needs_plan": True, "subgoals": []})
    agent2 = _build_test_agent(plan=parsed_plan)
    assert agent2.ctx.plan.get_active_subgoal() is not None
    assert agent2.ctx.plan.get_active_subgoal().id == "sg_1"


def test_subgoal_completion_single_subgoal():
    """Verify lifecycle of a task with exactly 1 subgoal."""
    plan = parse_task_ledger(
        {
            "needs_plan": True,
            "subgoals": [
                {"id": "sg_1", "text": "Only subgoal", "done_when": "done", "status": "active", "results": []}
            ],
        }
    )
    agent = _build_test_agent(plan=plan)

    assert agent.ctx.plan.get_active_subgoal().id == "sg_1"

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=True,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Completed only subgoal",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )

    # Subgoal is marked completed, no active/pending remain
    assert agent.ctx.plan.subgoals[0].status == "completed"
    assert agent.ctx.plan.get_active_subgoal() is None

    # Subsequent progress check with subgoal_done=True does not crash
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        snap = agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subsequent check",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
        assert snap["status"] == "ready"
        assert agent.ctx.plan.subgoals[0].status == "completed"


def test_subgoal_completion_n_subgoals_sequence():
    """Verify sequential advancement across N subgoals (1 -> 2 -> 3 -> all complete)."""
    plan = _multi_subgoal_plan(count=3)
    agent = _build_test_agent(plan=plan)

    assert agent.ctx.plan.subgoals[0].status == "active"
    assert agent.ctx.plan.subgoals[1].status == "pending"
    assert agent.ctx.plan.subgoals[2].status == "pending"

    # 1st Milestone: Subgoal 1 completes -> Subgoal 2 becomes active
    mock_s2_1 = SupervisorDecision(action_type="OVERRIDE_ACTION", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_1):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal 1 finished",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
    assert agent.ctx.plan.subgoals[0].status == "completed"
    assert agent.ctx.plan.subgoals[1].status == "active"
    assert agent.ctx.plan.subgoals[2].status == "pending"
    assert agent.ctx.active_subgoal == "Subgoal 2 description"

    # 2nd Milestone: Subgoal 2 completes -> Subgoal 3 becomes active
    mock_s2_2 = SupervisorDecision(action_type="OVERRIDE_ACTION", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_2):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal 2 finished",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
    assert agent.ctx.plan.subgoals[0].status == "completed"
    assert agent.ctx.plan.subgoals[1].status == "completed"
    assert agent.ctx.plan.subgoals[2].status == "active"
    assert agent.ctx.active_subgoal == "Subgoal 3 description"

    # 3rd Milestone: Subgoal 3 completes -> all subgoals completed
    mock_s2_3 = SupervisorDecision(action_type="OVERRIDE_ACTION", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_3):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal 3 finished",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )
    assert all(sg.status == "completed" for sg in agent.ctx.plan.subgoals)
    assert agent.ctx.plan.get_active_subgoal() is None


# ==============================================================================
# 4. Stuck Counter Oscillation & Recovery Guidance
# ==============================================================================


def test_stuck_counter_oscillation_matrix():
    """Verify stuck counter increment, reset, and threshold recovery guidance:
    stuck -> not stuck -> stuck -> stuck -> not stuck -> stuck*3 -> reset on progress.
    """
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    def _call_s2(stuck: bool, tactical_hint: str | None = None):
        mock_decision = SupervisorDecision(
            action_type="OVERRIDE_ACTION",
            subgoal_done=False,
            stuck=stuck,
            tactical_hint=tactical_hint,
        )
        with patch("reflex.agent.resolve_escalation", return_value=mock_decision):
            return agent._handle_escalation(
                trigger_type="progress_milestone",
                reason="Stuck check test",
                s1_decision=None,
                step_t0=0.0,
                s1_ms=0,
                pre_obs_ms=0,
            )

    # Initial state
    assert agent._stuck_counter == 0
    assert agent.ctx.tactical_hint == ""

    # 1. stuck = True -> counter = 1
    _call_s2(stuck=True)
    assert agent._stuck_counter == 1
    assert agent.ctx.tactical_hint == ""

    # 2. stuck = False (not stuck) -> counter resets to 0
    _call_s2(stuck=False)
    assert agent._stuck_counter == 0
    assert agent.ctx.tactical_hint == ""

    # 3. stuck = True -> counter = 1
    _call_s2(stuck=True)
    assert agent._stuck_counter == 1
    assert agent.ctx.tactical_hint == ""

    # 4. stuck = True -> counter = 2
    _call_s2(stuck=True)
    assert agent._stuck_counter == 2
    assert agent.ctx.tactical_hint == ""

    # 5. stuck = False (not stuck) -> counter resets to 0
    _call_s2(stuck=False)
    assert agent._stuck_counter == 0
    assert agent.ctx.tactical_hint == ""

    # 6. stuck = True -> counter = 1
    _call_s2(stuck=True)
    assert agent._stuck_counter == 1

    # 7. stuck = True -> counter = 2
    _call_s2(stuck=True)
    assert agent._stuck_counter == 2
    assert agent.ctx.tactical_hint == ""

    # 8. stuck = True -> counter = 3 (Threshold reached! Recovery hint applied)
    _call_s2(stuck=True, tactical_hint="Try clicking search filter button")
    assert agent._stuck_counter == 3
    assert agent.ctx.tactical_hint == "Try clicking search filter button"

    # 9. stuck = True -> counter = 4 (Counter keeps incrementing, hint retained)
    _call_s2(stuck=True, tactical_hint="Alternative recovery suggestion")
    assert agent._stuck_counter == 4
    assert agent.ctx.tactical_hint == "Alternative recovery suggestion"

    # 10. S1 takes an action that makes effective progress -> clears tactical_hint
    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    page_state = {"url": "https://www.example.com/test", "fingerprint": "fp_prog", "actions": [action]}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="OK")

    with (
        patch.object(agent.browser, "act", return_value=act_res),
        patch.object(agent.browser, "observe", return_value=page_state),
        patch("reflex.agent.verify_step", return_value=verification),
    ):
        agent.page = page_state
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.5,
            latency_ms=10,
            model_name="Turnike-Web",
            usage={},
            preset_text=None,
            escalation_note=None,
            step_t0=0.0,
            s1_ms=10,
            s2_ms=0,
            pre_obs_ms=5,
        )
    assert agent.ctx.tactical_hint == ""

    # 11. S2 runs with stuck=False -> resets counter to 0
    _call_s2(stuck=False)
    assert agent._stuck_counter == 0

    # Snapshot metrics reflect stuck_counter
    snap = agent.snapshot()
    assert snap["metrics"]["stuck_counter"] == 0


# ==============================================================================
# 5. Concurrent Triggers & Priority
# ==============================================================================


def test_concurrent_milestone_triggers_priority():
    """When site change, 12 S1 steps, and subgoal completion occur at the same step,
    site_changed takes precedence and only one clean milestone reason is scheduled.
    """
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)
    agent._s1_steps_since_s2 = 11  # Will become 12 on this step

    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    before_page = {"url": "https://www.google.com/search?q=test", "fingerprint": "fp_b", "actions": [action]}
    after_page = {
        "url": "https://en.wikipedia.org/wiki/Test",
        "fingerprint": "fp_a",
        "actions": [action],
        "text": "Criteria for subgoal 1 found here",
    }
    verification = VerificationResult(
        executed_ok=True,
        effective_progress=True,
        page_changed=True,
        url_changed=True,
        reason="Navigated to new site with criteria",
    )

    with (
        patch.object(agent.browser, "act", return_value=act_res),
        patch.object(agent.browser, "observe", return_value=after_page),
        patch("reflex.agent.verify_step", return_value=verification),
    ):
        agent.page = before_page
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.5,
            latency_ms=10,
            model_name="Turnike-Web",
            usage={},
            preset_text=None,
            escalation_note=None,
            step_t0=0.0,
            s1_ms=10,
            s2_ms=0,
            pre_obs_ms=5,
        )

    # Site change is prioritized
    assert agent.pending_escalation_reason is not None
    trig, reason = agent.pending_escalation_reason
    assert trig == "progress_milestone"
    assert "Site changed from google.com to en.wikipedia.org" in reason


# ==============================================================================
# 6. Simple Tasks Immunity
# ==============================================================================


def test_simple_task_needs_plan_false_complete_immunity():
    """Verify simple tasks (needs_plan=False) are completely immune to all milestone triggers."""
    plan = parse_task_ledger({"needs_plan": False, "subgoals": []})
    agent = _build_test_agent(plan=plan)

    action = {"id": "act_1", "kind": "click", "label": "Next", "node": 1}
    act_res = {"success": True}
    verification = VerificationResult(executed_ok=True, effective_progress=True, page_changed=True, reason="OK")

    # Run 15 consecutive S1 steps with site changes on each step
    for step_i in range(1, 16):
        before_page = {"url": f"https://site{step_i}.com", "fingerprint": "fp_b", "actions": [action]}
        after_page = {"url": f"https://site{step_i+1}.com", "fingerprint": "fp_a", "actions": [action]}

        with (
            patch.object(agent.browser, "act", return_value=act_res),
            patch.object(agent.browser, "observe", return_value=after_page),
            patch("reflex.agent.verify_step", return_value=verification),
        ):
            agent.page = before_page
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.5,
                latency_ms=10,
                model_name="Turnike-Web",
                usage={},
                preset_text=None,
                escalation_note=None,
                step_t0=0.0,
                s1_ms=10,
                s2_ms=0,
                pre_obs_ms=5,
            )
        # Pending escalation must NEVER be set for progress_milestone
        assert agent.pending_escalation_reason is None

    # Step() with needs_plan=False must NOT trigger line 386 12-step check
    assert agent._s1_steps_since_s2 == 15
    mock_s1_decision = {
        "choice": "act_1",
        "operation": "DONE",
        "target": None,
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 15,
        "model": "Turnike-Web",
    }
    with (
        patch("reflex.agent.choose_action", return_value=mock_s1_decision),
        patch("reflex.agent.evaluate_decision", return_value=MagicMock(should_escalate=False, joint_confidence=0.95, target_margin=0.5)),
    ):
        snap = agent.step()
        # S1 DONE must directly finish simple task
        assert snap["status"] == "done"
        assert agent.status == "done"


# ==============================================================================
# 7. Noop Escalation Suppression for Progress Milestone
# ==============================================================================


def test_progress_milestone_noop_suppression():
    """Verify that when progress_milestone resolves without an override action,
    it does not increment _noop_escalation_counts and leaves agent in 'ready' status.
    """
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    # Initial noop counts
    assert agent._noop_escalation_counts == {}

    # S2 returns OVERRIDE_ACTION with empty operation (noop)
    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation=None,
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        snap = agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="12 consecutive S1 steps without S2 (12 steps)",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )

    # Must be ready and NOT counted as noop escalation
    assert snap["status"] == "ready"
    assert agent.status == "ready"
    assert agent._noop_escalation_counts == {}


# ==============================================================================
# 8. S1 DONE Interception & Gate verify_done Routing
# ==============================================================================


def test_s1_done_operation_routing_on_planned_task():
    """System 1 emitting DONE operation on a planned task routes to progress_milestone."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    mock_s1_decision = {
        "choice": "act_1",
        "operation": "DONE",
        "target": None,
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 15,
        "model": "Turnike-Web",
    }
    mock_gate_eval = MagicMock(should_escalate=False, joint_confidence=0.95, target_margin=0.5)

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=True,
        stuck=False,
    )

    with (
        patch("reflex.agent.choose_action", return_value=mock_s1_decision),
        patch("reflex.agent.evaluate_decision", return_value=mock_gate_eval),
        patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision) as mock_resolve,
    ):
        snap = agent.step()
        assert snap["status"] == "ready"
        assert agent.status == "ready"
        assert mock_resolve.called
        assert mock_resolve.call_args.kwargs["trigger_type"] == "progress_milestone"
        assert "System 1 signaled subgoal completion (DONE operation)" in mock_resolve.call_args.kwargs["escalation_reason"]


def test_gate_verify_done_routing_on_planned_task():
    """Gate escalation with trigger_type='verify_done' routes to progress_milestone on planned tasks."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    mock_s1_decision = {
        "choice": "act_1",
        "operation": "CLICK",
        "target": "Next",
        "confidence": 0.6,
        "target_confidence": 0.6,
        "latency_ms": 15,
        "model": "Turnike-Web",
    }
    mock_gate_eval = MagicMock(
        should_escalate=True,
        trigger_type="verify_done",
        reason="Verification gate triggered for DONE",
        joint_confidence=0.6,
        target_margin=0.1,
    )

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=True,
        stuck=False,
    )

    with (
        patch("reflex.agent.choose_action", return_value=mock_s1_decision),
        patch("reflex.agent.evaluate_decision", return_value=mock_gate_eval),
        patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision) as mock_resolve,
    ):
        snap = agent.step()
        assert snap["status"] == "ready"
        assert mock_resolve.called
        assert mock_resolve.call_args.kwargs["trigger_type"] == "progress_milestone"
        assert "System 1 signaled subgoal completion (DONE verification)" in mock_resolve.call_args.kwargs["escalation_reason"]


# ==============================================================================
# 9. Subgoal Completion Signal Robustness & Edge Cases
# ==============================================================================


@pytest.mark.parametrize(
    ("active_sg", "verification", "page", "expected_result"),
    [
        # None active subgoal -> False
        (None, None, {}, False),
        # Verification with subgoal_done attr
        (Subgoal(id="1", text="sg1"), MagicMock(subgoal_done=True, details={}), {}, True),
        # Verification with details['subgoal_done'] = True
        (Subgoal(id="1", text="sg1"), MagicMock(subgoal_done=False, details={"subgoal_done": True}), {}, True),
        # Verification with details not a dict (e.g. None or str)
        (Subgoal(id="1", text="sg1"), MagicMock(subgoal_done=False, details="invalid"), {}, False),
        # Page text matching done_when (case-insensitive)
        (
            Subgoal(id="1", text="sg1", done_when="Birth date recorded"),
            None,
            {"text": "birth date recorded in 1815"},
            True,
        ),
        # Page full_text matching done_when
        (
            Subgoal(id="1", text="sg1", done_when="Found Charles"),
            None,
            {"text": "", "full_text": "we found charles babbage here"},
            True,
        ),
        # Page headings matching done_when
        (
            Subgoal(id="1", text="sg1", done_when="Early Life"),
            None,
            {"text": "", "full_text": "", "headings": ["1. Background", "2. Early Life & Work"]},
            True,
        ),
        # Page with None / missing values
        (
            Subgoal(id="1", text="sg1", done_when="Target string"),
            None,
            {"text": None, "full_text": None, "headings": None},
            False,
        ),
        # Empty done_when string
        (
            Subgoal(id="1", text="sg1", done_when=""),
            None,
            {"text": "Some non-empty text", "headings": ["Header"]},
            False,
        ),
    ],
)
def test_is_subgoal_completion_signaled_edge_cases(
    active_sg: Subgoal | None,
    verification: Any,
    page: dict[str, Any],
    expected_result: bool,
):
    """_is_subgoal_completion_signaled must handle missing, non-dict, and varied inputs safely."""
    agent = _build_test_agent()
    result = agent._is_subgoal_completion_signaled(active_sg, verification, page)
    assert result is expected_result


# ==============================================================================
# 10. Malformed Results & Candidates Handling in Progress Check
# ==============================================================================


def test_malformed_extracted_data_in_escalation():
    """Supervisor returning malformed or partially empty results/candidates must not crash."""
    plan = _multi_subgoal_plan(2)
    agent = _build_test_agent(plan=plan)

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
        extracted_results=[
            {"fact": None, "source_url": "https://example.com"},  # None fact
            {"fact": "Valid fact 1", "source_url": None},  # None URL
            {"fact": "<div class='dom'>Raw HTML string</div>"},  # Raw DOM should be rejected
            {"fact": "Valid fact 2", "source_url": "https://example.com/2"},
        ],
        extracted_candidates=[
            {"item": None},  # None item
            {"item": "   "},  # Empty item
            {"item": "Valid Item A", "attributes": "not_a_dict"},  # Non-dict attributes
            {"item": "Valid Item B", "attributes": {"price": "100"}},
        ],
        raw={
            "new_results": [
                "Direct string fact from raw",
            ]
        },
    )

    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        snap = agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Testing malformed data",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )

    assert snap["status"] == "ready"
    active_sg = agent.ctx.plan.get_active_subgoal()
    # At most 3 facts added, raw DOM rejected, invalid facts ignored
    assert len(active_sg.results) == 2
    assert any(r.fact == "Valid fact 1" for r in active_sg.results)
    assert any(r.fact == "Valid fact 2" for r in active_sg.results)

    # Valid candidates added
    assert any(c.item == "Valid Item A" for c in agent.ctx.plan.candidates)
    assert any(c.item == "Valid Item B" for c in agent.ctx.plan.candidates)


# ==============================================================================
# 11. 2500-Character Summarization during Progress Milestone
# ==============================================================================


def test_summarization_triggered_during_progress_milestone():
    """When completed subgoals exceed 2500 chars, summarization preserves sources and updates memory."""
    plan = _multi_subgoal_plan(2)
    # Add large text to completed subgoal
    large_fact_1 = "Fact " * 300 + " from Source A"  # ~2100 chars
    large_fact_2 = "Fact " * 200 + " from Source B"  # ~1400 chars
    plan.subgoals[0].status = "completed"
    plan.subgoals[0].results.append(
        SubgoalResult(fact=large_fact_1, source_url="https://source-a.com/data", step=1)
    )
    plan.subgoals[0].results.append(
        SubgoalResult(fact=large_fact_2, source_url="https://source-b.com/data", step=2)
    )
    assert plan.total_results_chars() > 2500

    agent = _build_test_agent(plan=plan)

    mock_s2_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=mock_s2_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Milestone check with large completed results",
            s1_decision=None,
            step_t0=0.0,
            s1_ms=0,
            pre_obs_ms=0,
        )

    # Summarization occurred
    assert plan.total_results_chars() < 2500
    # Both sources are strictly preserved
    sources = {r.source_url for r in plan.subgoals[0].results}
    assert "https://source-a.com/data" in sources
    assert "https://source-b.com/data" in sources
    # Working memory was refreshed with formatted plan
    assert "source-a.com" in agent.ctx.working_memory
    assert "source-b.com" in agent.ctx.working_memory
