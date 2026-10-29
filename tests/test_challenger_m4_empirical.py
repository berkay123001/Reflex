"""Empirical verification test suite for Milestone M4 DONE Verification & Response Synthesis.

Executed by: challenger_m4_2_run2
Verifies Invariants 1-5 under normal and adversarial stress conditions:
- Invariant 1: If verify_done() returns False, self.status MUST NOT be "done", must be "ready".
- Invariant 2: If verify_done() returns False and missing_sg exists, self.ctx.active_subgoal MUST strictly match missing_sg.text.
- Invariant 3: If verify_done() returns True, self.status MUST be "done", and source-cited final response must be added to chat history citing sources.
- Invariant 4: Simple tasks (needs_plan=False) MUST NEVER be blocked by verify_done().
- Invariant 5: _done_rejected_url tuple (cur_norm, step_count) must be recorded on every rejection.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.planner import (
    Candidate,
    Subgoal,
    SubgoalResult,
    TaskLedger,
    parse_task_ledger,
)
from reflex.supervisor import SupervisorDecision


def _create_mock_agent(
    plan: TaskLedger | None = None,
    initial_url: str = "https://a.com/search?q=test",
    initial_status: str = "ready",
) -> ReflexAgent:
    """Instantiate a ReflexAgent with mocked browser and initial state."""
    with (
        patch("reflex.agent.Browser") as mock_browser_cls,
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_browser = MagicMock()
        mock_browser_cls.return_value = mock_browser
        mock_browser.observe.return_value = {
            "url": initial_url,
            "title": "Search",
            "text": "Page content",
            "full_text": "Full page content",
            "headings": ["Search Heading"],
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
        agent = ReflexAgent(url=initial_url, goal="Test Goal", config=config)
        agent.browser = mock_browser
        agent.status = initial_status
        return agent


# ==============================================================================
# Invariant 1: If verify_done() returns False, self.status MUST NOT be "done", must be "ready"
# ==============================================================================


@pytest.mark.parametrize(
    "subgoals,candidates,budget_sources,initial_url,expected_reason_substr",
    [
        # Subgoal incomplete (pending)
        (
            [
                {"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []},
            ],
            [],
            1,
            "https://a.com/1",
            "is not completed",
        ),
        # Subgoal marked completed but empty results
        (
            [
                {"id": "sg_1", "text": "Subgoal 1", "status": "completed", "results": []},
            ],
            [],
            1,
            "https://a.com/1",
            "has no verified results",
        ),
        # Sources below budget (only 1 domain: a.com vs budget 2)
        (
            [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1",
                    "status": "completed",
                    "results": [{"fact": "F1", "source_url": "https://a.com/1", "step": 1}],
                },
                {
                    "id": "sg_2",
                    "text": "Subgoal 2",
                    "status": "completed",
                    "results": [{"fact": "F2", "source_url": "https://a.com/2", "step": 2}],
                },
            ],
            [],
            2,
            "https://a.com/1",
            "budget requires 2",
        ),
        # Candidates count below budget in comparison task
        (
            [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1",
                    "status": "completed",
                    "results": [{"fact": "F1", "source_url": "https://a.com/1", "step": 1}],
                },
            ],
            [
                {"item": "Item 1", "source_url": "https://a.com/1", "attributes": {"p": 10}},
            ],
            3,  # Budget requires 3, only 1 candidate
            "https://a.com/1",
            "expected at least 3",
        ),
    ],
)
def test_invariant_1_status_is_never_done_when_rejected(
    subgoals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    budget_sources: int,
    initial_url: str,
    expected_reason_substr: str,
):
    """Invariant 1: Direct verification call on unsatisfied plan must set status to 'ready' and never 'done'."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect all",
        "budget": {"sources": budget_sources, "max_steps": 25},
        "subgoals": subgoals,
        "candidates": candidates,
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url=initial_url, initial_status="running")

    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] != "done", f"Status must NOT be 'done', got {snap['status']}"
    assert snap["status"] == "ready", f"Status must be 'ready', got {snap['status']}"
    assert agent.status != "done"
    assert agent.status == "ready"

    # Confirm reason appears in escalation note
    notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
    assert any(expected_reason_substr in m["message"] for m in notes)


def test_invariant_1_via_s1_step_execution():
    """Invariant 1: When S1 emits DONE but plan verification fails, step() must return status 'ready'."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [{"fact": "F1", "source_url": "https://a.com/1", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "completed",
                "results": [],  # Empty results!
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_status="running")
    agent.ctx.active_subgoal = ""

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.99,
        "target_confidence": 0.99,
        "latency_ms": 15,
        "model": "turnike",
        "usage": {},
    }

    sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Premature complete", summary_for_user="Done.")
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup),
    ):
        snap = agent.step()
        assert snap["status"] != "done"
        assert snap["status"] == "ready"
        assert agent.status == "ready"


def test_invariant_1_via_s2_supervisor_complete_task():
    """Invariant 1: When S2 supervisor attempts COMPLETE_TASK but plan criteria fail, status must be 'ready'."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []},
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_status="running")

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Premature complete")
    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
        assert snap["status"] != "done"
        assert snap["status"] == "ready"
        assert agent.status == "ready"


# ==============================================================================
# Invariant 2: If verify_done() returns False and missing_sg exists,
# self.ctx.active_subgoal MUST strictly match missing_sg.text
# ==============================================================================


def test_invariant_2_missing_subgoal_text_strict_match():
    """Invariant 2: When verify_done() identifies a missing subgoal, active_subgoal must strictly equal its text."""
    complex_text = "Find 'Special' & Unique #123 (Sub-goal / Task) — äöü 🚀"
    raw = {
        "needs_plan": True,
        "budget": {"sources": 1, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_special",
                "text": complex_text,
                "status": "pending",
                "results": [],
            }
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = "Some other active subgoal"

    agent._verify_and_complete_task_from_plan()

    assert agent.ctx.active_subgoal == complex_text
    assert agent.ctx.active_subgoal == plan.subgoals[0].text
    assert plan.subgoals[0].status == "active"


def test_invariant_2_reactivation_clears_other_actives():
    """Invariant 2: Re-activating a missing subgoal must revert any previously active subgoals to 'pending'."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "active",  # Currently active
                "results": [{"fact": "Fact 1", "source_url": "https://a.com", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2 Missing",
                "status": "completed",
                "results": [],  # Marked completed but defective (no results)
            },
            {
                "id": "sg_3",
                "text": "Subgoal 3",
                "status": "pending",
                "results": [],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = "Subgoal 1"

    agent._verify_and_complete_task_from_plan()

    # sg_2 was defective and must be reactivated
    assert agent.ctx.active_subgoal == "Subgoal 2 Missing"
    assert plan.subgoals[1].status == "active"
    # sg_1 must be demoted from active to pending
    assert plan.subgoals[0].status == "pending"
    # sg_3 remains pending
    assert plan.subgoals[2].status == "pending"
    # Exactly one subgoal is active
    assert sum(1 for s in plan.subgoals if s.status == "active") == 1


def test_invariant_2_budget_failure_reactivates_last_subgoal():
    """Invariant 2: When sources budget fails, the last subgoal is reactivated."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 3, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "First Subgoal",
                "status": "completed",
                "results": [{"fact": "F1", "source_url": "https://a.com/1", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Second Subgoal",
                "status": "completed",
                "results": [{"fact": "F2", "source_url": "https://a.com/2", "step": 2}],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    # Both subgoals and initial_url are on a.com -> only 1 distinct domain vs budget 3
    agent = _create_mock_agent(plan=plan, initial_url="https://a.com/home")
    agent.ctx.active_subgoal = ""

    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] == "ready"
    assert agent.status == "ready"
    # Last subgoal is sg_2
    assert agent.ctx.active_subgoal == "Second Subgoal"
    assert plan.subgoals[1].status == "active"
    assert plan.subgoals[0].status == "completed"


def test_invariant_2_empty_subgoals_plan_does_not_crash():
    """Invariant 2 edge case: If a plan somehow has no subgoals and fails budget, missing_sg is None."""
    plan = TaskLedger(needs_plan=True, subgoals=[], budget={"sources": 2})
    agent = _create_mock_agent(plan=plan, initial_url="https://a.com")
    agent.ctx.active_subgoal = "Original Subgoal"

    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] == "ready"
    assert agent.status == "ready"
    # When missing_sg is None, active_subgoal is untouched
    assert agent.ctx.active_subgoal == "Original Subgoal"


# ==============================================================================
# Invariant 3: If verify_done() returns True, self.status MUST be "done",
# and source-cited final response must be added to chat history citing sources.
# ==============================================================================


def test_invariant_3_done_approved_synthesizes_all_citations():
    """Invariant 3: Successful DONE verification sets status to 'done' and emits source-cited response."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Alan Turing details",
                "status": "completed",
                "results": [
                    {
                        "fact": "Alan Turing born 23 June 1912 in London",
                        "source_url": "https://en.wikipedia.org/wiki/Alan_Turing",
                        "step": 2,
                    }
                ],
            },
            {
                "id": "sg_2",
                "text": "Find Alonzo Church details",
                "status": "completed",
                "results": [
                    {
                        "fact": "Alonzo Church introduced lambda calculus in 1936",
                        "source_url": "https://plato.stanford.edu/entries/church/",
                        "step": 4,
                    }
                ],
            },
        ],
        "candidates": [
            {
                "item": "Turing Machine",
                "source_url": "https://en.wikipedia.org/wiki/Turing_machine",
                "attributes": {"year": 1936, "type": "Abstract Model"},
            },
            {
                "item": "Lambda Calculus",
                "source_url": "https://plato.stanford.edu/entries/lambda-calculus/",
                "attributes": {"year": 1936, "type": "Formal System"},
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.primary_goal = "Investigate Turing and Church foundations of computing"

    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] == "done"
    assert agent.status == "done"

    # Verify chat history
    chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
    assert len(chat_msgs) > 0, "A chat message must be appended to chat history"
    final_text = chat_msgs[-1]["message"]

    # Header with primary goal
    assert "Investigate Turing and Church foundations of computing" in final_text

    # Subgoal findings and facts
    assert "Alan Turing born 23 June 1912" in final_text
    assert "Alonzo Church introduced lambda calculus" in final_text

    # Source citations
    assert "Kaynak: https://en.wikipedia.org/wiki/Alan_Turing" in final_text
    assert "Kaynak: https://plato.stanford.edu/entries/church/" in final_text

    # Candidates comparison table
    assert "Karşılaştırma ve Aday Tablosu:" in final_text
    assert "Turing Machine" in final_text
    assert "Lambda Calculus" in final_text
    assert "Kaynak: https://en.wikipedia.org/wiki/Turing_machine" in final_text
    assert "Kaynak: https://plato.stanford.edu/entries/lambda-calculus/" in final_text


def test_invariant_3_via_s1_step_execution():
    """Invariant 3: S1 DONE -> System 2 COMPLETE_TASK without a summary falls back to the cited ledger synthesis."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 1, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [{"fact": "Verified Fact", "source_url": "https://example.com/source", "step": 1}],
            }
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = ""

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.98,
        "target_confidence": 0.98,
        "latency_ms": 10,
        "model": "turnike",
        "usage": {},
    }

    sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Verified", summary_for_user=None)
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup),
    ):
        snap = agent.step()
        assert snap["status"] == "done"
        assert agent.status == "done"
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert len(chat_msgs) > 0
        assert "Kaynak: https://example.com/source" in chat_msgs[-1]["message"]


# ==============================================================================
# Invariant 4: Simple tasks (needs_plan=False) MUST NEVER be blocked by verify_done()
# ==============================================================================


def test_invariant_4_simple_task_direct_verify_and_complete():
    """Invariant 4: _verify_and_complete_task_from_plan() immediately returns 'done' for simple tasks."""
    # When plan is None
    agent = _create_mock_agent(plan=None)
    snap = agent._verify_and_complete_task_from_plan()
    assert snap["status"] == "done"
    assert agent.status == "done"

    # When plan has needs_plan=False
    plan = TaskLedger(needs_plan=False)
    agent2 = _create_mock_agent(plan=plan)
    snap2 = agent2._verify_and_complete_task_from_plan()
    assert snap2["status"] == "done"
    assert agent2.status == "done"


def test_invariant_4_simple_task_planner_verify_done_never_blocks():
    """Invariant 4: TaskLedger.verify_done() returns True for simple tasks."""
    plan = TaskLedger(needs_plan=False)
    is_done, reason, missing_sg = plan.verify_done()
    assert is_done is True
    assert missing_sg is None
    assert "No plan required" in reason


def test_invariant_4_simple_task_s1_done_step_execution():
    """Invariant 4: System 1 DONE on simple task is never blocked by plan verification."""
    raw = {"needs_plan": False}
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.config.verify_done_with_supervisor = False
    agent.history.append({"step": 1, "url": "https://simple.com", "operation": "GOTO_URL"})

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 20,
        "model": "turnike",
        "usage": {},
    }

    with patch("reflex.agent.choose_action", return_value=s1_decision):
        snap = agent.step()
        assert snap["status"] == "done"
        assert agent.status == "done"


def test_invariant_4_simple_task_s2_complete_task_execution():
    """Invariant 4: Supervisor COMPLETE_TASK on simple task is never blocked by plan verification."""
    raw = {"needs_plan": False}
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)

    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Simple task complete",
        summary_for_user="Done simple task.",
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
        assert snap["status"] == "done"
        assert agent.status == "done"


# ==============================================================================
# Invariant 5: _done_rejected_url tuple (cur_norm, step_count) must be recorded on every rejection
# ==============================================================================


@pytest.mark.parametrize(
    "url_input,expected_norm_url",
    [
        ("https://example.com/checkout", "https://example.com/checkout"),
        ("https://example.com/checkout/", "https://example.com/checkout"),
        ("https://example.com/checkout#payment-section", "https://example.com/checkout"),
        ("https://example.com/checkout/#order_summary", "https://example.com/checkout"),
        ("https://example.com/search?q=query#results", "https://example.com/search?q=query"),
        ("https://example.com/", "https://example.com"),
    ],
)
def test_invariant_5_url_normalization_on_rejection(url_input: str, expected_norm_url: str):
    """Invariant 5: _done_rejected_url correctly normalizes URL (strips fragment and trailing slash)."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [{"id": "sg_1", "text": "Incomplete", "status": "pending", "results": []}],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url=url_input)
    agent.page["url"] = url_input
    agent.history = [{"step": 1}, {"step": 2}, {"step": 3}]

    agent._verify_and_complete_task_from_plan()

    assert agent._done_rejected_url == (expected_norm_url, 3)


def test_invariant_5_suppression_window_and_expiration():
    """Invariant 5: Anti-loop suppression is active for <= 8 steps and expires at step 9."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [{"id": "sg_1", "text": "Incomplete", "status": "pending", "results": []}],
    }
    target_url = "https://example.com/page"

    # Agent 1: within window (delta = 13 - 5 = 8 <= 8)
    plan1 = parse_task_ledger(raw)
    agent1 = _create_mock_agent(plan=plan1, initial_url=target_url)
    agent1.page["url"] = target_url
    agent1.page["actions"] = [{"id": "b1", "kind": "click", "label": "Btn", "node": 1}]
    agent1.history = [{"step": i} for i in range(5)]
    agent1._verify_and_complete_task_from_plan()
    assert agent1._done_rejected_url == ("https://example.com/page", 5)

    agent1.history = [{"step": i} for i in range(13)]  # delta = 8
    s1_click = {
        "operation": "CLICK",
        "target": "b1",
        "choice": "b1",
        "confidence": 0.8,
        "target_confidence": 0.8,
        "latency_ms": 10,
        "model": "turnike",
        "usage": {},
    }
    with patch("reflex.agent.choose_action", return_value=s1_click) as mock_choose1:
        agent1.step()
        assert mock_choose1.called
        _, kwargs1 = mock_choose1.call_args
        assert kwargs1.get("suppress_done") is True

    # Agent 2: outside window (delta = 14 - 5 = 9 > 8)
    plan2 = parse_task_ledger(raw)
    agent2 = _create_mock_agent(plan=plan2, initial_url=target_url)
    agent2.page["url"] = target_url
    agent2.page["actions"] = [{"id": "b1", "kind": "click", "label": "Btn", "node": 1}]
    agent2.history = [{"step": i} for i in range(5)]
    agent2._verify_and_complete_task_from_plan()
    assert agent2._done_rejected_url == ("https://example.com/page", 5)

    agent2.history = [{"step": i} for i in range(14)]  # delta = 9
    with patch("reflex.agent.choose_action", return_value=s1_click) as mock_choose2:
        agent2.step()
        assert mock_choose2.called
        _, kwargs2 = mock_choose2.call_args
        assert kwargs2.get("suppress_done") is False


def test_invariant_5_recorded_on_supervisor_rejection():
    """Invariant 5: When supervisor escalates with trigger_type='verify_done', _done_rejected_url is recorded."""
    raw = {"needs_plan": False}
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url="https://example.com/store#cart")
    agent.page["url"] = "https://example.com/store#cart"
    agent.history = [{"step": 1}]

    # Supervisor returns UPDATE_SUBGOAL (rejecting direct DONE)
    sup_decision = SupervisorDecision(action_type="UPDATE_SUBGOAL", reasoning="Not done yet")
    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
        assert agent._done_rejected_url == ("https://example.com/store", 1)


# ==============================================================================
# Adversarial Stress Harness
# ==============================================================================


def test_stress_fuzzed_subgoals_and_candidates():
    """Stress test: Random permutations of subgoals and candidates maintain all invariants."""
    import random

    rng = random.Random(42)

    for trial in range(50):
        num_sgs = rng.randint(1, 5)
        budget_sources = rng.randint(1, 4)
        subgoals: list[Subgoal] = []
        all_completed = True
        has_empty_completed = False

        for i in range(num_sgs):
            status = rng.choice(["pending", "active", "completed", "failed"])
            if status != "completed":
                all_completed = False
            has_results = rng.choice([True, False])
            results: list[SubgoalResult] = []
            if has_results:
                domain = rng.choice(["alpha.com", "beta.org", "gamma.net"])
                results.append(SubgoalResult(fact=f"Fact {i}", source_url=f"https://{domain}/page{i}", step=i))
            elif status == "completed":
                has_empty_completed = True

            subgoals.append(
                Subgoal(
                    id=f"sg_{i}",
                    text=f"Subgoal text {i}",
                    status=status,
                    results=results,
                )
            )

        num_cands = rng.randint(0, 3)
        candidates: list[Candidate] = []
        for j in range(num_cands):
            c_domain = rng.choice(["alpha.com", "delta.io"])
            candidates.append(
                Candidate(
                    item=f"Item {j}",
                    source_url=f"https://{c_domain}/item{j}",
                    attributes={"score": j * 10},
                )
            )

        ledger = TaskLedger(
            needs_plan=True,
            budget={"sources": budget_sources, "max_steps": 30},
            subgoals=subgoals,
            candidates=candidates,
        )

        agent = _create_mock_agent(plan=ledger, initial_url="https://alpha.com/run")
        snap = agent._verify_and_complete_task_from_plan()

        distinct_domains = {
            r.source_url.split("//")[1].split("/")[0].removeprefix("www.")
            for sg in subgoals
            for r in sg.results
            if r.source_url and "//" in r.source_url
        } | {
            c.source_url.split("//")[1].split("/")[0].removeprefix("www.")
            for c in candidates
            if c.source_url and "//" in c.source_url
        }  # the visited start page (alpha.com) is not a source unless a result/candidate came from it

        should_pass = (
            all_completed
            and (not has_empty_completed or len(candidates) > 0)  # a filled candidate table is evidence
            and len(distinct_domains) >= budget_sources
            and (len(candidates) == 0 or len(candidates) >= budget_sources)
        )

        if should_pass:
            assert snap["status"] == "done", f"Trial {trial}: expected done, got {snap['status']}"
            assert agent.status == "done"
            chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
            assert len(chat_msgs) > 0
            assert "Kaynak:" in chat_msgs[-1]["message"]
        else:
            assert snap["status"] == "ready", f"Trial {trial}: expected ready, got {snap['status']}"
            assert agent.status == "ready"
            assert agent._done_rejected_url is not None
            assert agent._done_rejected_url[0] == "https://alpha.com/run"
            active_sgs = [s for s in ledger.subgoals if s.status == "active"]
            assert len(active_sgs) <= 1
            if active_sgs:
                assert agent.ctx.active_subgoal == active_sgs[0].text


def test_adversarial_malformed_urls_and_schemes():
    """Adversarial test: Non-standard URL schemes (data:, javascript:, about:, file:) do not corrupt verify_done."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [
                    {"fact": "F1", "source_url": "javascript:void(0)", "step": 1},
                    {"fact": "F2", "source_url": "about:blank", "step": 2},
                    {"fact": "F3", "source_url": "data:text/html,<b>hi</b>", "step": 3},
                    {"fact": "F4", "source_url": "https://legit-site.com/doc", "step": 4},
                ],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "completed",
                "results": [
                    {"fact": "F5", "source_url": "file:///etc/hosts", "step": 5},
                ],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url="https://legit-site.com/doc")

    # Only 1 legitimate web domain (legit-site.com) vs budget 2
    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] == "ready"
    assert agent.status == "ready"
    # sg_2 reactivated
    assert agent.ctx.active_subgoal == "Subgoal 2"


def test_adversarial_unicode_and_special_characters_synthesis():
    """Adversarial test: Full Turkish unicode, emojis, and multiline text format cleanly in synthesis."""
    turkish_goal = "İstanbul Boğazı & Çanakkale Köprüleri Karşılaştırması 🌉 🇹🇷"
    raw = {
        "needs_plan": True,
        "budget": {"sources": 1, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "15 Temmuz Şehitler Köprüsü bilgileri (Boğaziçi)",
                "status": "completed",
                "results": [
                    {
                        "fact": "Açılış: 29 Ekim 1973\nUzunluk: 1.560 m — Ana açıklık: 1.074 m.",
                        "source_url": "https://kgm.gov.tr/Sayfalar/KGM/SiteTr/Projeler/Bogazici.aspx",
                        "step": 2,
                    }
                ],
            }
        ],
        "candidates": [
            {
                "item": "1915 Çanakkale Köprüsü",
                "source_url": "https://1915canakkale.com",
                "attributes": {
                    "orta_açıklık": "2023 m",
                    "toplam_uzunluk": "4.608 m",
                    "tarih": "18 Mart 2022",
                },
            }
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.primary_goal = turkish_goal

    snap = agent._verify_and_complete_task_from_plan()

    assert snap["status"] == "done"
    chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
    assert len(chat_msgs) > 0
    msg = chat_msgs[-1]["message"]

    assert turkish_goal in msg
    assert "15 Temmuz Şehitler Köprüsü" in msg
    assert "1915 Çanakkale Köprüsü" in msg
    assert "orta_açıklık: 2023 m" in msg
    assert "Kaynak: https://kgm.gov.tr/Sayfalar/KGM/SiteTr/Projeler/Bogazici.aspx" in msg
    assert "Kaynak: https://1915canakkale.com" in msg


def test_adversarial_progressive_recovery_flow():
    """Adversarial test: Agent rejects DONE -> reactivates sg -> sg completes -> DONE succeeds."""
    raw = {
        "needs_plan": True,
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [{"fact": "Fact 1", "source_url": "https://site-a.com/page", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "pending",
                "results": [],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url="https://site-a.com/page")

    # Step 1: Premature DONE attempt -> REJECTED
    snap1 = agent._verify_and_complete_task_from_plan()
    assert snap1["status"] == "ready"
    assert agent.status == "ready"
    assert agent.ctx.active_subgoal == "Subgoal 2"
    assert plan.subgoals[1].status == "active"

    # Step 2: Subgoal 2 is fulfilled by adding a result from a second source
    plan.add_result(
        fact="Fact 2 from Site B",
        source_url="https://site-b.org/info",
        step=2,
        subgoal_id="sg_2",
    )
    plan.subgoals[1].status = "completed"
    agent.ctx.active_subgoal = ""

    # Step 3: Second DONE attempt -> APPROVED
    snap2 = agent._verify_and_complete_task_from_plan()
    assert snap2["status"] == "done"
    assert agent.status == "done"
    chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
    assert len(chat_msgs) > 0
    final_text = chat_msgs[-1]["message"]
    assert "Fact 1" in final_text
    assert "Fact 2 from Site B" in final_text
    assert "Kaynak: https://site-a.com/page" in final_text
    assert "Kaynak: https://site-b.org/info" in final_text
