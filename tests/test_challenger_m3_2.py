"""Empirical Invariant & Adversarial Challenge Suite for Milestone M3 (challenger_m3_2_run2).

Tests:
1. Invariant 1: _s1_steps_since_s2 must NEVER be negative and must reset to 0 whenever S2 executes or resolves an escalation.
2. Invariant 2: _stuck_counter must NEVER be negative and must reset to 0 whenever stuck=False.
3. Invariant 3: self.ctx.active_subgoal must ALWAYS match the text of self.ctx.plan.get_active_subgoal().
4. Invariant 4: Simple tasks with needs_plan=False must NEVER trigger progress milestones regardless of step count or URL transitions.
5. Invariant 5: Distilled facts and candidate updates must attach to the active subgoal before advancement when subgoal_done=True.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflex.agent import VerificationResult
from reflex.gate import GateEvaluation
from reflex.planner import Subgoal, TaskLedger
from reflex.supervisor import SupervisorDecision
from tests.test_m3_integration import _create_mock_agent, _sample_planned_task

# ==============================================================================
# Invariant 1: _s1_steps_since_s2 tracking & reset
# ==============================================================================


def test_invariant_1_s1_steps_never_negative_and_resets_on_s2_escalation():
    """Invariant 1: _s1_steps_since_s2 must NEVER be negative and must reset to 0 whenever S2 executes or resolves an escalation."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    assert agent._s1_steps_since_s2 == 0

    # 1. Simulate 5 S1 steps
    for i in range(1, 6):
        with patch("reflex.agent.verify_step") as mock_verify:
            mock_verify.return_value = VerificationResult(
                executed_ok=True, page_changed=True, effective_progress=True
            )
            agent._execute_and_verify(
                action={"id": str(i), "kind": "click", "node": i},
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
            assert agent._s1_steps_since_s2 == i
            assert agent._s1_steps_since_s2 >= 0

    # 2. Reset on S2 escalation resolution
    sup_decision = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="S2 checked milestone",
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
    assert agent._s1_steps_since_s2 >= 0

    # 3. Simulate another 3 S1 steps followed by an S2 action execution
    for i in range(1, 4):
        with patch("reflex.agent.verify_step") as mock_verify:
            mock_verify.return_value = VerificationResult(
                executed_ok=True, page_changed=True, effective_progress=True
            )
            agent._execute_and_verify(
                action={"id": f"s1_{i}", "kind": "click", "node": i},
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
            assert agent._s1_steps_since_s2 == i

    # S2 executes direct browser action
    with patch("reflex.agent.verify_step") as mock_verify:
        mock_verify.return_value = VerificationResult(
            executed_ok=True, page_changed=True, effective_progress=True
        )
        agent._execute_and_verify(
            action={"id": "s2_act", "kind": "click", "node": 99},
            actor="S2",
            operation="CLICK",
            target=None,
            confidence=1.0,
            target_confidence=1.0,
            joint_confidence=1.0,
            target_margin=1.0,
            latency_ms=100,
            model_name="muse",
            usage={},
            preset_text=None,
            escalation_note="S2 recovery click",
        )
    assert agent._s1_steps_since_s2 == 0
    assert agent._s1_steps_since_s2 >= 0


@pytest.mark.parametrize("trig", [
    "progress_milestone",
    "verify_done",
    "loop_detected",
    "oscillation",
    "scroll_budget",
    "missing_action",
    "blocked",
])
def test_invariant_1_resets_across_all_escalation_trigger_types(trig: str):
    """Invariant 1: _s1_steps_since_s2 resets to 0 regardless of escalation trigger type."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)
    agent._s1_steps_since_s2 = 7

    sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning=f"Resolving {trig}",
        subgoal_done=False,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=sup):
        agent._handle_escalation(trigger_type=trig, reason=f"Test {trig}", s1_decision=None)

    assert agent._s1_steps_since_s2 == 0


# ==============================================================================
# Invariant 2: _stuck_counter tracking & reset
# ==============================================================================


def test_invariant_2_stuck_counter_never_negative_and_resets_on_stuck_false():
    """Invariant 2: _stuck_counter must NEVER be negative and must reset to 0 whenever stuck=False."""
    plan = _sample_planned_task()
    agent = _create_mock_agent(plan=plan)

    assert agent._stuck_counter == 0

    stuck_sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Stuck on page",
        subgoal_done=False,
        stuck=True,
    )
    unstuck_sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        reasoning="Making progress",
        subgoal_done=False,
        stuck=False,
    )

    with patch("reflex.agent.resolve_escalation", return_value=stuck_sup):
        for expected in range(1, 6):
            agent._handle_escalation(
                trigger_type="progress_milestone", reason=f"Stuck {expected}", s1_decision=None
            )
            assert agent._stuck_counter == expected
            assert agent._stuck_counter >= 0
            if expected >= 3:
                assert agent.ctx.tactical_hint != ""

    # Reset on stuck=False
    with patch("reflex.agent.resolve_escalation", return_value=unstuck_sup):
        agent._handle_escalation(
            trigger_type="progress_milestone", reason="Unstuck check", s1_decision=None
        )
        assert agent._stuck_counter == 0
        assert agent._stuck_counter >= 0

        # Repeated stuck=False remains 0
        agent._handle_escalation(
            trigger_type="progress_milestone", reason="Still unstuck", s1_decision=None
        )
        assert agent._stuck_counter == 0


# ==============================================================================
# Invariant 3: self.ctx.active_subgoal matches plan active subgoal
# ==============================================================================


def test_invariant_3_active_subgoal_matches_plan_during_sequential_advancement():
    """Invariant 3: self.ctx.active_subgoal must match self.ctx.plan.get_active_subgoal().text while subgoals are active."""
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Subgoal 1 text", done_when="done 1", status="active"),
            Subgoal(id="sg2", text="Subgoal 2 text", done_when="done 2", status="pending"),
            Subgoal(id="sg3", text="Subgoal 3 text", done_when="done 3", status="pending"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    # Initial state
    active_sg = agent.ctx.plan.get_active_subgoal()
    assert active_sg is not None
    assert agent.ctx.active_subgoal == active_sg.text == "Subgoal 1 text"

    # Complete SG1 -> advance to SG2
    sup1 = SupervisorDecision(action_type="UPDATE_SUBGOAL", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=sup1):
        agent._handle_escalation(trigger_type="progress_milestone", reason="SG1 done", s1_decision=None)

    active_sg = agent.ctx.plan.get_active_subgoal()
    assert active_sg is not None
    assert agent.ctx.active_subgoal == active_sg.text == "Subgoal 2 text"

    # Complete SG2 -> advance to SG3
    sup2 = SupervisorDecision(action_type="UPDATE_SUBGOAL", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=sup2):
        agent._handle_escalation(trigger_type="progress_milestone", reason="SG2 done", s1_decision=None)

    active_sg = agent.ctx.plan.get_active_subgoal()
    assert active_sg is not None
    assert agent.ctx.active_subgoal == active_sg.text == "Subgoal 3 text"


def test_invariant_3_violation_active_subgoal_stale_when_all_subgoals_completed():
    """Empirical Invariant 3 Failure: When all subgoals are completed, self.ctx.active_subgoal remains stale

    at the completed subgoal's text instead of clearing or matching self.ctx.plan.get_active_subgoal() (which is None).
    """
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Subgoal 1 text", done_when="done 1", status="active"),
            Subgoal(id="sg2", text="Subgoal 2 text", done_when="done 2", status="pending"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    # Complete SG1 -> SG2 active
    sup1 = SupervisorDecision(action_type="UPDATE_SUBGOAL", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=sup1):
        agent._handle_escalation(trigger_type="progress_milestone", reason="SG1 done", s1_decision=None)
    assert agent.ctx.active_subgoal == "Subgoal 2 text"

    # Complete SG2 (the final subgoal)
    sup2 = SupervisorDecision(action_type="UPDATE_SUBGOAL", subgoal_done=True, stuck=False)
    with patch("reflex.agent.resolve_escalation", return_value=sup2):
        agent._handle_escalation(trigger_type="progress_milestone", reason="SG2 done", s1_decision=None)

    # All subgoals are completed
    assert all(sg.status == "completed" for sg in agent.ctx.plan.subgoals)
    active_sg = agent.ctx.plan.get_active_subgoal()
    assert active_sg is None

    # INVARIANT CHECK:
    # If get_active_subgoal() is None, active_subgoal should be empty string "" (no active subgoal).
    # In agent.py lines 781-796, next_sg is None and next_subgoal is None, so self.ctx.active_subgoal is NOT updated!
    # It remains "Subgoal 2 text", creating a state mismatch between DualTierContext and TaskLedger!
    active_sg_expected_text = active_sg.text if active_sg is not None else ""
    assert agent.ctx.active_subgoal == active_sg_expected_text, (
        f"Invariant 3 violated: self.ctx.active_subgoal ({agent.ctx.active_subgoal!r}) "
        f"does not match self.ctx.plan.get_active_subgoal() ({active_sg_expected_text!r})"
    )


def test_invariant_3_violation_updated_subgoal_diverges_from_plan():
    """Empirical Invariant 3 Failure: When S2 returns updated_subgoal, self.ctx.active_subgoal

    is updated but self.ctx.plan.get_active_subgoal().text remains unchanged.
    """
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Original Subgoal 1", done_when="done 1", status="active"),
            Subgoal(id="sg2", text="Subgoal 2", done_when="done 2", status="pending"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        updated_subgoal="Refined Subgoal 1 Text from S2",
        subgoal_done=False,
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=sup):
        agent._handle_escalation(trigger_type="progress_milestone", reason="S2 refine", s1_decision=None)

    active_sg = agent.ctx.plan.get_active_subgoal()
    assert active_sg is not None

    # INVARIANT CHECK:
    # agent.py line 858 sets self.ctx.active_subgoal = sup.updated_subgoal,
    # but does NOT update active_sg.text in self.ctx.plan.
    assert agent.ctx.active_subgoal == active_sg.text, (
        f"Invariant 3 violated: self.ctx.active_subgoal is {agent.ctx.active_subgoal!r} "
        f"but self.ctx.plan.get_active_subgoal().text is {active_sg.text!r}"
    )


def test_invariant_3_violation_next_subgoal_override_ignored():
    """Empirical Invariant 3 Failure: S2 explicitly provides next_subgoal='sg3' with subgoal_done=True,

    but advance_subgoal() unconditionally activates sg2 in linear sequence, ignoring S2's next_subgoal.
    """
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Subgoal 1", done_when="done 1", status="active"),
            Subgoal(id="sg2", text="Subgoal 2", done_when="done 2", status="pending"),
            Subgoal(id="sg3", text="Subgoal 3", done_when="done 3", status="pending"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    # S2 signals SG1 is done, and explicitly instructs agent to skip SG2 and activate SG3
    sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        subgoal_done=True,
        next_subgoal="sg3",
        stuck=False,
    )
    with patch("reflex.agent.resolve_escalation", return_value=sup):
        agent._handle_escalation(trigger_type="progress_milestone", reason="Jump to SG3", s1_decision=None)

    # Because advance_subgoal() is called first, next_sg is sg2 (not None).
    # The elif next_subgoal: block in agent.py line 785 is bypassed entirely.
    assert agent.ctx.active_subgoal == "Subgoal 3", (
        f"next_subgoal override was ignored: active_subgoal is {agent.ctx.active_subgoal!r} instead of 'Subgoal 3'"
    )


# ==============================================================================
# Invariant 4: Simple tasks (needs_plan=False) NEVER trigger progress_milestone
# ==============================================================================


def test_invariant_4_simple_tasks_never_trigger_progress_milestones_plan_none():
    """Invariant 4: When plan is None, 20 consecutive S1 steps and multi-domain transitions NEVER schedule progress_milestone."""
    agent = _create_mock_agent(plan=None)
    assert agent.ctx.plan is None

    domains = ["google.com", "wikipedia.org", "amazon.com", "news.ycombinator.com", "github.com"]
    for i in range(20):
        dom = domains[i % len(domains)]
        agent.page = {
            "url": f"https://{dom}/item_{i}",
            "fingerprint": f"fp_{i}",
            "actions": [],
        }
        agent.browser.observe.return_value = agent.page
        agent.browser.act.return_value = {}

        with patch("reflex.agent.verify_step") as mock_v:
            mock_v.return_value = VerificationResult(
                executed_ok=True, page_changed=True, url_changed=True, effective_progress=True
            )
            agent._execute_and_verify(
                action={"id": f"act_{i}", "kind": "click", "node": i},
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.9,
                latency_ms=20,
                model_name="turnike",
                usage={},
                preset_text=None,
                escalation_note=None,
            )
            assert agent.pending_escalation_reason is None
            assert agent._s1_steps_since_s2 == i + 1

    # When S1 emits DONE on simple task, it completes directly without progress_milestone
    with patch("reflex.agent.choose_action") as mock_choose:
        mock_choose.return_value = {
            "operation": "DONE",
            "confidence": 0.99,
            "target_confidence": 0.99,
            "joint_confidence": 0.99,
            "target_margin": 0.99,
            "choice": None,
            "target": None,
            "latency_ms": 10,
            "model": "turnike",
            "usage": {},
        }
        with patch("reflex.agent.evaluate_decision") as mock_gate:
            mock_gate.return_value = GateEvaluation(
                should_escalate=False,
                trigger_type="none",
                confidence=0.99,
                target_confidence=0.99,
                joint_confidence=0.99,
                target_margin=0.99,
            )
            snap = agent.step()
            assert snap["status"] == "done"
            assert agent.pending_escalation_reason is None


def test_invariant_4_simple_tasks_never_trigger_progress_milestones_needs_plan_false():
    """Invariant 4: When plan has needs_plan=False, 20 consecutive S1 steps and URL transitions NEVER schedule progress_milestone."""
    plan = TaskLedger(
        needs_plan=False,
        subgoals=[Subgoal(id="1", text="Single site goal", done_when="target found")],
    )
    agent = _create_mock_agent(plan=plan)

    domains = ["baykara.com", "yeditepe.edu.tr", "thy.com"]
    for i in range(15):
        dom = domains[i % len(domains)]
        agent.page = {
            "url": f"https://{dom}/step_{i}",
            "fingerprint": f"fp_{i}",
            "actions": [],
        }
        agent.browser.observe.return_value = agent.page
        agent.browser.act.return_value = {}

        with patch("reflex.agent.verify_step") as mock_v:
            mock_v.return_value = VerificationResult(
                executed_ok=True, page_changed=True, url_changed=True, effective_progress=True
            )
            agent._execute_and_verify(
                action={"id": f"act_{i}", "kind": "click", "node": i},
                actor="S1",
                operation="CLICK",
                target=None,
                confidence=0.9,
                target_confidence=0.9,
                joint_confidence=0.9,
                target_margin=0.9,
                latency_ms=20,
                model_name="turnike",
                usage={},
                preset_text=None,
                escalation_note=None,
            )
            assert agent.pending_escalation_reason is None
            assert agent._s1_steps_since_s2 == i + 1

    # Verify step() when _s1_steps_since_s2 >= 12 does NOT trigger progress_milestone
    assert agent._s1_steps_since_s2 >= 12
    with patch("reflex.agent.choose_action") as mock_choose:
        mock_choose.return_value = {
            "operation": "DONE",
            "confidence": 0.99,
            "target_confidence": 0.99,
            "joint_confidence": 0.99,
            "target_margin": 0.99,
            "choice": None,
            "target": None,
            "latency_ms": 10,
            "model": "turnike",
            "usage": {},
        }
        with patch("reflex.agent.evaluate_decision") as mock_gate:
            mock_gate.return_value = GateEvaluation(
                should_escalate=False,
                trigger_type="none",
                confidence=0.99,
                target_confidence=0.99,
                joint_confidence=0.99,
                target_margin=0.99,
            )
            snap = agent.step()
            assert snap["status"] == "done"
            assert agent.pending_escalation_reason is None


# ==============================================================================
# Invariant 5: Distilled facts & candidates attach before advancement
# ==============================================================================


def test_invariant_5_distilled_facts_and_candidates_attach_before_advancement():
    """Invariant 5: Distilled facts and candidate updates must attach to active subgoal before advancement when subgoal_done=True."""
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Find product prices", done_when="price found", status="active"),
            Subgoal(id="sg2", text="Compare warranties", done_when="warranty found", status="pending"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    sup = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        subgoal_done=True,
        stuck=False,
        extracted_results=[
            {"fact": "Product price is $299 at Store A", "source_url": "https://store-a.com/item"}
        ],
        extracted_candidates=[
            {"item": "Product A", "source_url": "https://store-a.com/item", "attributes": {"price": 299}}
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup):
        agent._handle_escalation(
            trigger_type="progress_milestone",
            reason="Subgoal 1 completion check",
            s1_decision=None,
        )

    sg1 = agent.ctx.plan.subgoals[0]
    sg2 = agent.ctx.plan.subgoals[1]

    # SG1 was completed
    assert sg1.status == "completed"
    # Result attached to SG1 (the active subgoal before advance)
    assert len(sg1.results) == 1
    assert sg1.results[0].fact == "Product price is $299 at Store A"
    assert sg1.results[0].source_url == "https://store-a.com/item"

    # SG2 is now active, but its results remain empty
    assert sg2.status == "active"
    assert len(sg2.results) == 0

    # Candidate table attached
    assert len(agent.ctx.plan.candidates) == 1
    assert agent.ctx.plan.candidates[0].item == "Product A"
    assert agent.ctx.plan.candidates[0].attributes == {"price": 299}


def test_invariant_5_candidate_deduplication_during_milestone_escalation():
    """Invariant 5: Multiple milestone escalations adding the same candidate item update attributes without creating duplicate rows."""
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Find flight deals", done_when="flight found", status="active"),
        ],
    )
    agent = _create_mock_agent(plan=plan)

    # First milestone check: adds candidate Flight TK1984 with price
    sup1 = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        subgoal_done=False,
        stuck=False,
        extracted_candidates=[
            {"item": "Flight TK1984", "source_url": "https://airline.com", "attributes": {"price": 350}}
        ],
    )
    with patch("reflex.agent.resolve_escalation", return_value=sup1):
        agent._handle_escalation(trigger_type="progress_milestone", reason="First check", s1_decision=None)

    assert len(agent.ctx.plan.candidates) == 1
    assert agent.ctx.plan.candidates[0].attributes == {"price": 350}

    # Second milestone check: updates same candidate Flight TK1984 with duration
    sup2 = SupervisorDecision(
        action_type="UPDATE_SUBGOAL",
        subgoal_done=False,
        stuck=False,
        extracted_candidates=[
            {"item": "Flight TK1984", "source_url": "https://airline.com", "attributes": {"duration": "3h 15m"}}
        ],
    )
    with patch("reflex.agent.resolve_escalation", return_value=sup2):
        agent._handle_escalation(trigger_type="progress_milestone", reason="Second check", s1_decision=None)

    # Must be deduplicated by item name: count remains 1, attributes merged
    assert len(agent.ctx.plan.candidates) == 1
    assert agent.ctx.plan.candidates[0].attributes == {"price": 350, "duration": "3h 15m"}
