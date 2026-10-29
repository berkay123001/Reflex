"""Comprehensive Unit and Integration Test Suite for Milestone M3 (Progress Ledger & Milestone Triggers).

Covers:
- Site change milestone trigger (_extract_domain(before) != _extract_domain(after))
- Same-site navigation does not trigger milestone
- 12 consecutive S1 steps without S2 triggers progress milestone
- S1 step counter increments on S1 and resets on S2 run/escalation
- Subgoal completion signal from S1 (DONE operation)
- Subgoal completion signal from verifier (details.subgoal_done)
- Subgoal completion signal from observed page criteria (done_when in text/headings)
- Subgoal advancement and active_subgoal update on subgoal_done=True
- All subgoals completed handling
- Stuck counter increment on stuck=True, recovery hint at >= 3, and reset on stuck=False
- Simple tasks (needs_plan=False) bypass all milestone triggers
- Progress milestone extracts results and candidates additively
- Progress milestone without override action leaves agent ready (no false noop escalation)
- SupervisorDecision progress check schema fields (subgoal_done, next_subgoal, stuck)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.planner import TaskLedger, parse_task_ledger
from reflex.supervisor import SupervisorDecision
from reflex.verifier import VerificationResult


def _create_mock_agent(plan: TaskLedger | None = None) -> ReflexAgent:
    """Helper to instantiate ReflexAgent with browser and network mocked out."""
    with (
        patch("reflex.agent.Browser") as mock_browser_cls,
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_browser = MagicMock()
        mock_browser_cls.return_value = mock_browser
        mock_browser.observe.return_value = {
            "url": "https://www.google.com/search?q=test",
            "title": "Google Search",
            "text": "Search results",
            "full_text": "Search results for test",
            "headings": ["Google"],
            "actions": [],
            "fingerprint": "fp_init",
        }
        mock_browser.fresh.return_value = True

        mock_interp_res = MagicMock()
        mock_interp_res.model_dump.return_value = {}
        mock_interp_res.allow_arbitrary_choices = False
        mock_interp_res.constraints = []
        mock_interp_res.plan = plan
        mock_interp.return_value = mock_interp_res

        config = ReflexConfig(max_steps=30, screenshots=False)
        agent = ReflexAgent("Test goal", "https://www.google.com/search?q=test", config=config)
        agent.browser = mock_browser
        return agent


def _sample_planned_task() -> TaskLedger:
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect dates from Wikipedia",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace lifespans",
                "done_when": "Ada Lovelace dates recorded",
                "status": "active",
                "results": [],
            },
            {
                "id": "sg_2",
                "text": "Find Charles Babbage lifespans",
                "done_when": "Charles Babbage dates recorded",
                "status": "pending",
                "results": [],
            },
        ],
    }
    return parse_task_ledger(raw)


# ==============================================================================
# 1. Site Change Trigger Tests
# ==============================================================================


def test_site_change_triggers_progress_milestone():
    """Site change (_extract_domain(before) != _extract_domain(after)) schedules progress milestone."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    before_page = {
        "url": "https://www.google.com/search?q=ada+lovelace",
        "fingerprint": "fp_1",
        "actions": [{"id": "1", "kind": "click", "label": "Wikipedia link", "role": "link", "node": 1}],
    }
    after_page = {
        "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "fingerprint": "fp_2",
        "actions": [],
        "text": "Ada Lovelace was an English mathematician",
        "headings": ["Ada Lovelace"],
    }
    agent.browser.observe.return_value = after_page
    agent.browser.act.return_value = {}

    action = {"id": "1", "kind": "click", "label": "Wikipedia link", "node": 1}
    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, url_changed=True, effective_progress=True
        )
        agent.page = before_page
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.95,
            target_confidence=0.95,
            joint_confidence=0.95,
            target_margin=0.95,
            latency_ms=100,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is not None
    trig, reason = agent.pending_escalation_reason
    assert trig == "progress_milestone"
    assert "google.com" in reason and "en.wikipedia.org" in reason


def test_same_site_navigation_does_not_trigger_milestone():
    """Navigating within the same domain does not trigger a site change milestone."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    before_page = {
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "fingerprint": "fp_1",
        "actions": [{"id": "1", "kind": "click", "label": "Ada Lovelace", "role": "link", "node": 1}],
    }
    after_page = {
        "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "fingerprint": "fp_2",
        "actions": [],
        "text": "Ada Lovelace article",
        "headings": ["Ada Lovelace"],
    }
    agent.browser.observe.return_value = after_page
    agent.browser.act.return_value = {}

    action = {"id": "1", "kind": "click", "label": "Ada Lovelace", "node": 1}
    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, url_changed=True, effective_progress=True
        )
        agent.page = before_page
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.95,
            target_confidence=0.95,
            joint_confidence=0.95,
            target_margin=0.95,
            latency_ms=100,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is None


def test_empty_or_special_urls_do_not_trigger_site_change():
    """about:blank or empty URLs have empty extracted domains and do not trigger site change."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    before_page = {"url": "about:blank", "fingerprint": "fp_1", "actions": []}
    after_page = {"url": "https://en.wikipedia.org/wiki/Ada_Lovelace", "fingerprint": "fp_2", "actions": []}
    agent.browser.observe.return_value = after_page
    agent.browser.act.return_value = {}

    action = {"id": "1", "kind": "goto", "label": "Go to URL"}
    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(executed_ok=True, page_changed=True, url_changed=True)
        agent.page = before_page
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="GOTO_URL",
            target=None,
            confidence=0.95,
            target_confidence=0.95,
            joint_confidence=0.95,
            target_margin=0.95,
            latency_ms=100,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is None


# ==============================================================================
# 2. 12 Consecutive S1 Steps Milestone Trigger Tests
# ==============================================================================


def test_12_consecutive_s1_steps_triggers_progress_milestone():
    """12 consecutive S1 steps without S2 triggers progress milestone check."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)
    assert agent._s1_steps_since_s2 == 0

    page = {
        "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "fingerprint": "fp_wiki",
        "actions": [{"id": "scroll_down", "kind": "scroll", "label": "Scroll"}],
    }
    agent.page = page
    agent.browser.observe.return_value = page
    agent.browser.act.return_value = {}

    action = {"id": "scroll_down", "kind": "scroll", "label": "Scroll"}

    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, element_state_changed=True, effective_progress=True
        )
        for i in range(1, 12):
            agent._execute_and_verify(
                action=action,
                actor="S1",
                operation="SCROLL_DOWN",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.9,
                latency_ms=50,
                model_name="turnike",
                usage={},
                preset_text=None,
                escalation_note=None,
            )
            assert agent._s1_steps_since_s2 == i
            assert agent.pending_escalation_reason is None

        # 12th consecutive S1 step triggers milestone
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="SCROLL_DOWN",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=50,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )
        assert agent._s1_steps_since_s2 == 12
        assert agent.pending_escalation_reason is not None
        trig, reason = agent.pending_escalation_reason
        assert trig == "progress_milestone"
        assert "12 consecutive S1 steps without S2" in reason


def test_s1_step_counter_resets_on_s2_escalation():
    """_s1_steps_since_s2 resets to 0 whenever S2 runs or resolves an escalation."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)
    agent._s1_steps_since_s2 = 8

    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="S2 checking progress",
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Progress check",
            s1_decision=None,
        )

    assert agent._s1_steps_since_s2 == 0


def test_step_method_triggers_progress_milestone_when_s1_counter_reaches_12():
    """agent.step() immediately escalates with progress_milestone when _s1_steps_since_s2 >= 12."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)
    agent._s1_steps_since_s2 = 12

    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Progress check after 12 steps",
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision) as mock_res:
        snap = agent.step()
        assert mock_res.called
        assert agent._s1_steps_since_s2 == 0
        assert snap["status"] == "ready"


# ==============================================================================
# 3. Subgoal Completion Signal Tests
# ==============================================================================


def test_subgoal_completion_signal_from_s1_done():
    """When System 1 emits DONE on a planned task, it triggers progress_milestone instead of ending session."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.99,
        "target_confidence": 0.99,
        "latency_ms": 40,
        "model": "turnike",
        "usage": {},
    }

    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Verified subgoal 1 completion",
        subgoal_done=True,
        stuck=False,
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision) as mock_sup,
    ):
        snap = agent.step()
        assert mock_sup.called
        _, kwargs = mock_sup.call_args
        assert kwargs.get("trigger_type") == "progress_milestone"
        assert "System 1 signaled subgoal completion" in kwargs.get("escalation_reason", "")
        # Session does not prematurely terminate with done; it advanced active subgoal
        assert snap["status"] == "ready"
        assert agent.ctx.active_subgoal == "Find Charles Babbage lifespans"


def test_subgoal_completion_signal_from_verifier():
    """When verifier results report subgoal_done=True, progress_milestone is scheduled."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    page = {
        "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "fingerprint": "fp_wiki",
        "actions": [{"id": "link_1", "kind": "click", "label": "Heading", "node": 1}],
    }
    agent.page = page
    agent.browser.observe.return_value = page
    agent.browser.act.return_value = {}

    action = {"id": "link_1", "kind": "click", "label": "Heading", "node": 1}

    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True,
            page_changed=True,
            effective_progress=True,
            details={"subgoal_done": True},
        )
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=50,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is not None
    trig, reason = agent.pending_escalation_reason
    assert trig == "progress_milestone"
    assert "Active subgoal criteria appear met" in reason


def test_subgoal_completion_signal_from_page_criteria():
    """When observed page text/headings match active subgoal's done_when text, progress_milestone is scheduled."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    # Active subgoal done_when is: "Ada Lovelace dates recorded"
    page = {
        "url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "fingerprint": "fp_wiki",
        "text": "Here are Ada Lovelace dates recorded in the official archives.",
        "full_text": "Full document with Ada Lovelace dates recorded.",
        "headings": ["Ada Lovelace dates recorded"],
        "actions": [{"id": "link_1", "kind": "click", "label": "Heading", "node": 1}],
    }
    agent.page = page
    agent.browser.observe.return_value = page
    agent.browser.act.return_value = {}

    action = {"id": "link_1", "kind": "click", "label": "Heading", "node": 1}

    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, effective_progress=True
        )
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.9,
            target_confidence=0.9,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=50,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is not None
    trig, reason = agent.pending_escalation_reason
    assert trig == "progress_milestone"
    assert "Active subgoal criteria appear met" in reason


# ==============================================================================
# 4. Subgoal Advancement & Stuck Counter Tests
# ==============================================================================


def test_subgoal_advancement_updates_status_and_active_subgoal():
    """When subgoal_done=True, active subgoal is marked completed, and next subgoal becomes active."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    assert agent.ctx.plan.subgoals[0].status == "active"
    assert agent.ctx.plan.subgoals[1].status == "pending"
    assert agent.ctx.active_subgoal == "Find Ada Lovelace lifespans"

    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Ada Lovelace dates extracted successfully",
        subgoal_done=True,
        stuck=False,
        extracted_results=[
            {"fact": "Ada Lovelace was born in 1815 and died in 1852", "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace"}
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal done check",
            s1_decision=None,
        )

    assert agent.ctx.plan.subgoals[0].status == "completed"
    assert agent.ctx.plan.subgoals[1].status == "active"
    assert agent.ctx.active_subgoal == "Find Charles Babbage lifespans"
    # Result was added to the first subgoal before advancing
    assert len(agent.ctx.plan.subgoals[0].results) == 1
    assert "1815" in agent.ctx.plan.subgoals[0].results[0].fact
    # Working memory was updated with formatted plan
    assert "Find Charles Babbage lifespans" in agent.ctx.working_memory


def test_subgoal_advancement_when_all_subgoals_completed():
    """When advancing past the final subgoal, all subgoals are completed cleanly."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    # Complete first subgoal
    agent.ctx.plan.advance_subgoal()
    assert agent.ctx.plan.subgoals[1].status == "active"

    # Complete second and final subgoal
    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Charles Babbage dates extracted",
        subgoal_done=True,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal 2 done check",
            s1_decision=None,
        )

    assert all(sg.status == "completed" for sg in agent.ctx.plan.subgoals)


def test_stuck_counter_increments_and_resets():
    """_stuck_counter increments on stuck=True, triggers recovery hint at >= 3, and resets on stuck=False."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)
    assert agent._stuck_counter == 0

    stuck_sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="No forward progress",
        subgoal_done=False,
        stuck=True,
    )

    with patch("reflex.agent.resolve_escalation", return_value=stuck_sup):
        # 1st stuck check
        agent._handle_escalation(trigger_type="progress_milestone", reason="Stuck check 1", s1_decision=None)
        assert agent._stuck_counter == 1
        assert agent.ctx.tactical_hint == ""

        # 2nd stuck check
        agent._handle_escalation(trigger_type="progress_milestone", reason="Stuck check 2", s1_decision=None)
        assert agent._stuck_counter == 2
        assert agent.ctx.tactical_hint == ""

        # 3rd stuck check -> sets recovery hint
        agent._handle_escalation(trigger_type="progress_milestone", reason="Stuck check 3", s1_decision=None)
        assert agent._stuck_counter == 3
        assert agent.ctx.tactical_hint != ""
        assert "stuck" in agent.ctx.tactical_hint.lower() or "alternative" in agent.ctx.tactical_hint.lower()

        # Unstuck resolution -> resets counter to 0
        unstuck_sup = SupervisorDecision(
            action_type="UPDATE_SUBGOAL",
            reasoning="Navigated to useful page",
            subgoal_done=False,
            stuck=False,
        )
        with patch("reflex.agent.resolve_escalation", return_value=unstuck_sup):
            agent._handle_escalation(trigger_type="progress_milestone", reason="Unstuck check", s1_decision=None)
            assert agent._stuck_counter == 0


# ==============================================================================
# 5. Simple Task (needs_plan=False) Bypass Tests
# ==============================================================================


def test_simple_task_needs_plan_false_bypasses_all_milestones():
    """Simple tasks with needs_plan=False never trigger progress milestone checks."""
    # Simple task ledger with needs_plan=False
    ledger = TaskLedger(needs_plan=False)
    agent = _create_mock_agent(plan=ledger)
    assert agent.ctx.plan is None or agent.ctx.plan.needs_plan is False

    before_page = {
        "url": "https://www.google.com/search?q=test",
        "fingerprint": "fp_1",
        "actions": [{"id": "1", "kind": "click", "label": "Result", "node": 1}],
    }
    after_page = {
        "url": "https://en.wikipedia.org/wiki/Test",
        "fingerprint": "fp_2",
        "actions": [],
        "text": "Wikipedia page text",
    }
    agent.page = before_page
    agent.browser.observe.return_value = after_page
    agent.browser.act.return_value = {}

    action = {"id": "1", "kind": "click", "label": "Result", "node": 1}

    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, url_changed=True, effective_progress=True
        )
        # Even with site change, needs_plan=False must not schedule milestone
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.95,
            target_confidence=0.95,
            joint_confidence=0.95,
            target_margin=0.95,
            latency_ms=50,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is None

    # Even with 15 S1 steps, needs_plan=False does not schedule milestone
    agent._s1_steps_since_s2 = 15
    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(executed_ok=True, page_changed=True)
        agent._execute_and_verify(
            action=action,
            actor="S1",
            operation="CLICK",
            target=None,
            confidence=0.95,
            target_confidence=0.95,
            joint_confidence=0.95,
            target_margin=0.95,
            latency_ms=50,
            model_name="turnike",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert agent.pending_escalation_reason is None

    # S1 DONE on needs_plan=False marks status="done" directly when verify_done_with_supervisor=False
    agent.config.verify_done_with_supervisor = False
    s1_done = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.99,
        "target_confidence": 0.99,
        "latency_ms": 30,
        "model": "turnike",
        "usage": {},
    }
    with patch("reflex.agent.choose_action", return_value=s1_done):
        snap = agent.step()
        assert snap["status"] == "done"

    # When verify_done_with_supervisor=True, it escalates as verify_done, NOT progress_milestone
    agent.status = "ready"
    agent.config.verify_done_with_supervisor = True
    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="All done")
    with (
        patch("reflex.agent.choose_action", return_value=s1_done),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision) as mock_sup,
    ):
        snap = agent.step()
        assert mock_sup.called
        _, kwargs = mock_sup.call_args
        assert kwargs.get("trigger_type") == "verify_done"
        assert kwargs.get("trigger_type") != "progress_milestone"


# ==============================================================================
# 6. Progress Milestone Additive Extraction & Noop Guard Tests
# ==============================================================================


def test_progress_milestone_extracts_results_and_candidates():
    """Distilled facts and candidate items from progress checks are additively integrated."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Extracted facts and candidate item",
        subgoal_done=False,
        stuck=False,
        extracted_results=[
            {"fact": "Ada Lovelace wrote the first computer algorithm in 1843", "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace", "step": 3}
        ],
        extracted_candidates=[
            {"item": "Ada Lovelace", "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace", "attributes": {"era": "Victorian", "role": "Mathematician"}}
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Progress check",
            s1_decision=None,
        )

    # Facts added to active subgoal
    active_sg = agent.ctx.plan.get_active_subgoal()
    assert len(active_sg.results) == 1
    assert "algorithm in 1843" in active_sg.results[0].fact

    # Candidates table populated
    assert len(agent.ctx.plan.candidates) == 1
    assert agent.ctx.plan.candidates[0].item == "Ada Lovelace"
    assert agent.ctx.plan.candidates[0].attributes.get("role") == "Mathematician"


def test_progress_milestone_without_action_does_not_count_as_noop():
    """Supervisor progress check without override action does not accumulate noop counts."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation=None,
        reasoning="Progress evaluated, continue with current DOM elements",
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Progress check",
            s1_decision=None,
        )

    # Status remains ready, noop count is zero
    assert snap["status"] == "ready"
    assert len(agent._noop_escalation_counts) == 0


def test_supervisor_decision_progress_fields():
    """SupervisorDecision properly holds subgoal_done, next_subgoal, and stuck fields."""
    sd = SupervisorDecision(
        subgoal_done=True,
        next_subgoal="sg_2",
        stuck=False,
    )
    assert sd.subgoal_done is True
    assert sd.next_subgoal == "sg_2"
    assert sd.stuck is False
