"""Comprehensive Unit and Integration Test Suite for Milestone M4 (DONE Verification & Response Synthesis).

Covers:
1. Multi-criteria plan-based DONE verification in reflex/agent.py (R4)
2. DONE approved when all criteria met (subgoals completed with results, candidates >= budget, sources >= budget)
3. DONE rejected when subgoal incomplete -> missing subgoal reactivated as active
4. DONE rejected when subgoal has no verified results -> subgoal reactivated
5. DONE rejected when distinct sources < budget.sources -> subgoal reactivated
6. DONE rejected when candidates table < budget.sources in comparison tasks -> subgoal reactivated
7. Anti-loop guard _done_rejected_url set on rejection and respected on subsequent steps (suppress_done)
8. Source citations (Kaynak: / Source:) and candidate comparison table in synthesized final response (R4, R5)
9. Simple task pass-through: needs_plan=False completes directly without plan verification (R1, R4)
10. Task completion attempted via S2 COMPLETE_TASK and via System 1 DONE when all subgoals completed
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.planner import TaskLedger, parse_task_ledger
from reflex.supervisor import SupervisorDecision


def _create_mock_agent(plan: TaskLedger | None = None, initial_url: str = "https://www.google.com/search?q=test") -> ReflexAgent:
    """Helper to instantiate ReflexAgent with browser and network mocked out."""
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
        agent = ReflexAgent(url=initial_url, goal="Test Primary Goal", config=config)
        agent.browser = mock_browser
        return agent


def _sample_planned_task_completed() -> TaskLedger:
    """A planned task where all criteria are satisfied (2 completed subgoals with results from 2 sources)."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect dates from Wikipedia and Britannica",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace lifespans",
                "done_when": "Ada Lovelace dates recorded",
                "status": "completed",
                "results": [
                    {
                        "fact": "Ada Lovelace born 1815, died 1852",
                        "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                        "step": 2,
                    }
                ],
            },
            {
                "id": "sg_2",
                "text": "Find Charles Babbage lifespans",
                "done_when": "Charles Babbage dates recorded",
                "status": "completed",
                "results": [
                    {
                        "fact": "Charles Babbage born 1791, died 1871",
                        "source_url": "https://www.britannica.com/biography/Charles-Babbage",
                        "step": 4,
                    }
                ],
            },
        ],
        "candidates": [],
    }
    return parse_task_ledger(raw)


# ==============================================================================
# 1. Approval Tests
# ==============================================================================


def test_m4_done_approved_via_s2_complete_task():
    """When all criteria are met and Supervisor returns COMPLETE_TASK, its own summary is the final answer."""
    plan = _sample_planned_task_completed()
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = ""

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 30,
        "model": "turnike",
        "usage": {},
    }
    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="All subgoals and sources satisfied",
        summary_for_user="Task completed successfully.",
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()
        assert snap["status"] == "done"
        assert agent.status == "done"

        # System 2's summary is the answer; the ledger template is not used
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert chat_msgs[-1]["message"] == "Task completed successfully."


def test_m4_done_approved_via_s1_done_when_all_subgoals_completed():
    """System 1 DONE with all subgoals completed goes to System 2 (verify_done) with the ledger check as evidence."""
    plan = _sample_planned_task_completed()
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = ""

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 25,
        "model": "turnike",
        "usage": {},
    }
    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Verified on page",
        summary_for_user="Ada Lovelace 1815, Charles Babbage 1791.",
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision) as s2,
    ):
        snap = agent.step()
        assert snap["status"] == "done"
        assert agent.status == "done"

        assert s2.call_count == 1
        assert s2.call_args.kwargs["trigger_type"] == "verify_done"
        assert "Ledger check: All completion criteria satisfied" in s2.call_args.kwargs["escalation_reason"]
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert chat_msgs[-1]["message"] == "Ada Lovelace 1815, Charles Babbage 1791."


# ==============================================================================
# 2. Rejection & Reactivation Tests
# ==============================================================================


def test_m4_done_rejected_when_subgoal_incomplete():
    """When a subgoal is not completed, DONE is rejected and the incomplete subgoal is reactivated."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect both figures",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace lifespans",
                "done_when": "Ada Lovelace dates recorded",
                "status": "completed",
                "results": [
                    {
                        "fact": "Ada Lovelace 1815-1852",
                        "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                        "step": 2,
                    }
                ],
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
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = ""

    # Attempt DONE via S2 COMPLETE_TASK
    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Attempting premature completion",
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(
            trigger_type="verify_done",
            reason="Verifying completion",
            s1_decision=None,
        )

        # Must reject DONE
        assert snap["status"] == "ready"
        assert agent.status == "ready"

        # Subgoal 2 must be reactivated
        sg2 = next(s for s in agent.ctx.plan.subgoals if s.id == "sg_2")
        assert sg2.status == "active"
        assert agent.ctx.active_subgoal == "Find Charles Babbage lifespans"

        # Anti-loop guard must be set
        cur_norm = str(agent.page.get("url", "")).split("#")[0].rstrip("/")
        assert agent._done_rejected_url == (cur_norm, len(agent.history))

        # Escalation note must be logged
        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("DONE rejected by plan verification" in m["message"] and "Find Charles Babbage lifespans" in m["message"] for m in notes)


def test_m4_done_rejected_when_subgoal_has_no_results():
    """When a subgoal is marked completed but has no results, DONE is rejected and the empty subgoal is reactivated."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect both figures",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace lifespans",
                "done_when": "Ada Lovelace dates recorded",
                "status": "completed",
                "results": [
                    {
                        "fact": "Ada Lovelace 1815-1852",
                        "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
                        "step": 2,
                    }
                ],
            },
            {
                "id": "sg_2",
                "text": "Find Charles Babbage lifespans",
                "done_when": "Charles Babbage dates recorded",
                "status": "completed",
                "results": [],  # Empty results!
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.active_subgoal = ""

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

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Looks complete", summary_for_user="Done.")
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()

        # Rejected!
        assert snap["status"] == "ready"
        assert agent.status == "ready"

        # Subgoal 2 reactivated from completed to active
        sg2 = next(s for s in agent.ctx.plan.subgoals if s.id == "sg_2")
        assert sg2.status == "active"
        assert agent.ctx.active_subgoal == "Find Charles Babbage lifespans"

        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("has no verified results" in m["message"] for m in notes)


def test_m4_done_rejected_when_sources_below_budget():
    """When distinct sources count is below budget.sources, DONE is rejected and the last subgoal is reactivated."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Collect from 3 sources",
        "budget": {"sources": 3, "max_steps": 25},  # Requires 3 distinct sources
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [
                    {"fact": "Fact 1", "source_url": "https://site-a.com/page1", "step": 1}
                ],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "completed",
                "results": [
                    {"fact": "Fact 2", "source_url": "https://site-b.com/page2", "step": 2}
                ],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    # Only 2 distinct domains (site-a.com, site-b.com) vs budget 3
    agent = _create_mock_agent(plan=plan, initial_url="https://site-a.com/page1")
    agent.ctx.active_subgoal = ""

    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Attempting completion with 2 of 3 sources",
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(
            trigger_type="verify_done",
            reason="Check sources",
            s1_decision=None,
        )

        assert snap["status"] == "ready"
        assert agent.status == "ready"

        # Last subgoal reactivated
        assert agent.ctx.active_subgoal == "Subgoal 2"
        sg2 = next(s for s in agent.ctx.plan.subgoals if s.id == "sg_2")
        assert sg2.status == "active"

        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("budget requires 3" in m["message"] for m in notes)


def test_m4_done_rejected_when_candidates_below_budget():
    """In comparison tasks, if candidate count is below budget.sources, DONE is rejected."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Compare headphone prices across 3 stores",
        "budget": {"sources": 3, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Search Store A",
                "status": "completed",
                "results": [{"fact": "Found Store A", "source_url": "https://store-a.com/item", "step": 1}],
            }
        ],
        "candidates": [
            {
                "item": "Sony WH-1000XM5",
                "source_url": "https://store-a.com/item",
                "attributes": {"price": "350 USD"},
            }
        ],
    }
    plan = parse_task_ledger(raw)
    # Only 1 candidate vs budget 3
    agent = _create_mock_agent(plan=plan, initial_url="https://store-a.com/item")
    agent.ctx.active_subgoal = ""

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Attempt complete")
    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
        assert snap["status"] == "ready"
        assert agent.status == "ready"
        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("Candidate table has 1 items, expected at least 3" in m["message"] for m in notes)


# ==============================================================================
# 3. Anti-Loop Guard Tests
# ==============================================================================


def test_m4_anti_loop_guard_sets_and_suppresses_done():
    """Anti-loop guard _done_rejected_url is set upon rejection and passes suppress_done=True on same URL."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Finish criteria",
        "budget": {"sources": 2, "max_steps": 20},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Incomplete subgoal",
                "status": "pending",
                "results": [],
            }
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url="https://example.com/checkout")
    agent.ctx.active_subgoal = ""

    # 1. Trigger rejection via S2 COMPLETE_TASK
    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Early done")
    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
        assert snap["status"] == "ready"

    # Anti-loop guard recorded
    assert agent._done_rejected_url == ("https://example.com/checkout", 0)

    # 2. Next step on same URL: suppress_done must be True
    s1_click = {
        "operation": "CLICK",
        "target": "link_1",
        "choice": "link_1",
        "confidence": 0.85,
        "target_confidence": 0.85,
        "latency_ms": 20,
        "model": "turnike",
        "usage": {},
    }
    agent.page["actions"] = [{"id": "link_1", "kind": "click", "label": "Proceed", "node": 1}]

    with patch("reflex.agent.choose_action", return_value=s1_click) as mock_choose:
        agent.step()
        assert mock_choose.called
        _, kwargs = mock_choose.call_args
        assert kwargs.get("suppress_done") is True

    # 3. Next step on different URL: suppress_done must be False
    agent.page["url"] = "https://other.com/page"
    with patch("reflex.agent.choose_action", return_value=s1_click) as mock_choose:
        agent.step()
        assert mock_choose.called
        _, kwargs = mock_choose.call_args
        assert kwargs.get("suppress_done") is False


# ==============================================================================
# 4. Final Answer Synthesis Tests
# ==============================================================================


def test_m4_final_answer_synthesis_citations_and_candidate_table():
    """Final answer synthesis includes primary goal, facts with source citations, and candidate comparison table."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Compare headphones",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Review Sony WH-1000XM5",
                "status": "completed",
                "results": [
                    {
                        "fact": "Sony WH-1000XM5 has 30h battery life and industry-leading ANC",
                        "source_url": "https://techradar.com/reviews/sony-wh-1000xm5",
                        "step": 2,
                    }
                ],
            },
            {
                "id": "sg_2",
                "text": "Review Bose QC Ultra",
                "status": "completed",
                "results": [
                    {
                        "fact": "Bose QC Ultra has spatial audio and 24h battery life",
                        "source_url": "https://rtings.com/headphones/reviews/bose/quietcomfort-ultra",
                        "step": 5,
                    }
                ],
            },
        ],
        "candidates": [
            {
                "item": "Sony WH-1000XM5",
                "source_url": "https://techradar.com/reviews/sony-wh-1000xm5",
                "attributes": {"fiyat": "399 USD", "puan": "4.8/5"},
            },
            {
                "item": "Bose QC Ultra",
                "source_url": "https://rtings.com/headphones/reviews/bose/quietcomfort-ultra",
                "attributes": {"fiyat": "429 USD", "puan": "4.7/5"},
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.ctx.primary_goal = "Compare Sony WH-1000XM5 vs Bose QC Ultra"
    agent.ctx.active_subgoal = ""

    s1_decision = {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.99,
        "target_confidence": 0.99,
        "latency_ms": 20,
        "model": "turnike",
        "usage": {},
    }

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Looks complete", summary_for_user=None)
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()
        assert snap["status"] == "done"

        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert len(chat_msgs) > 0
        final_text = chat_msgs[-1]["message"]

        # Check primary goal header
        assert "Compare Sony WH-1000XM5 vs Bose QC Ultra" in final_text
        # Check subgoals findings
        assert "Sony WH-1000XM5 has 30h battery life" in final_text
        assert "Bose QC Ultra has spatial audio" in final_text
        # Check source citations
        assert "https://techradar.com/reviews/sony-wh-1000xm5" in final_text
        assert "https://rtings.com/headphones/reviews/bose/quietcomfort-ultra" in final_text
        assert "Kaynak:" in final_text
        # Check candidate table
        assert "Karşılaştırma ve Aday Tablosu:" in final_text
        assert "fiyat: 399 USD" in final_text
        assert "fiyat: 429 USD" in final_text


# ==============================================================================
# 5. Simple Task Pass-Through Tests
# ==============================================================================


def test_m4_simple_task_bypass_s2_complete_task():
    """Simple tasks (needs_plan=False) complete directly when Supervisor returns COMPLETE_TASK."""
    raw = {"needs_plan": False}
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)

    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Simple task single-site completion",
        summary_for_user="Simple task finished.",
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        snap = agent._handle_escalation(
            trigger_type="verify_done",
            reason="Simple task verify",
            s1_decision=None,
        )

        assert snap["status"] == "done"
        assert agent.status == "done"
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert any("Simple task finished." in m["message"] for m in chat_msgs)


def test_m4_simple_task_bypass_s1_done_direct():
    """Simple tasks (needs_plan=False) complete directly via System 1 DONE without plan verification."""
    raw = {"needs_plan": False}
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)
    agent.config.verify_done_with_supervisor = False
    agent.history.append({"step": 1, "url": "https://www.google.com/search?q=test", "operation": "GOTO_URL"})

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
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert any("high confidence verification passed" in m["message"] for m in chat_msgs)


# ==============================================================================
# 6. Reactivation Invariant Synchronization Tests
# ==============================================================================


def test_m4_subgoal_reactivation_clears_other_active_subgoals():
    """When a missing subgoal is reactivated, any other active subgoal is reverted to pending."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Two subgoals",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [{"fact": "Fact 1", "source_url": "https://site-a.com/1", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "completed",
                "results": [],  # Empty results
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan)

    # Artificially set sg_1 as active to test single-active invariant
    plan.subgoals[0].status = "active"
    agent.ctx.active_subgoal = "Subgoal 1"

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Premature completion")
    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)

        # sg_2 was missing (no results); it must be active, and sg_1 must be pending (not active)
        assert plan.subgoals[1].status == "active"
        assert plan.subgoals[0].status == "pending"
        assert agent.ctx.active_subgoal == "Subgoal 2"
        assert agent.ctx.plan.get_active_subgoal() == plan.subgoals[1]


def test_m4_visited_pages_without_results_do_not_count_as_sources():
    """A page visited without any recorded result (e.g. a search page) is not a source: the first COMPLETE_TASK
    is sent back with the budget reason; a repeated COMPLETE_TASK is accepted."""
    raw = {
        "needs_plan": True,
        "finish_criteria": "Research topic across 2 sites",
        "budget": {"sources": 2, "max_steps": 25},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Subgoal 1",
                "status": "completed",
                "results": [{"fact": "Fact from site A", "source_url": "https://site-a.com/article", "step": 1}],
            },
            {
                "id": "sg_2",
                "text": "Subgoal 2",
                "status": "completed",
                "results": [{"fact": "Fact also from site A", "source_url": "https://site-a.com/summary", "step": 2}],
            },
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(plan=plan, initial_url="https://site-a.com/article")
    agent.ctx.active_subgoal = ""

    # Add a history record showing site-b.com was also visited
    agent.history.append({
        "step": 1,
        "url": "https://site-b.com/data",
        "before_url": "https://site-a.com/article",
    })

    # site-b.com was only visited: results come from site-a.com alone, so 1 source < budget 2
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

    sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Looks complete", summary_for_user="Site A facts.")
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()
        assert snap["status"] == "ready"
        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("Found 1 sources, budget requires 2" in m["message"] for m in notes)
        assert "Found 1 sources" in agent.ctx.tactical_hint

        snap = agent.step()
        assert snap["status"] == "done"
        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert chat_msgs[-1]["message"] == "Site A facts."


def test_m4_reporting_subgoal_closed_by_s2_summary():
    """A trailing 'compare and report' subgoal cannot be satisfied by browsing: System 2's COMPLETE_TASK closes it
    with its summary as the result, so the task finishes on the first COMPLETE_TASK."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "budget": {"sources": 1, "max_steps": 10},
        "subgoals": [
            {"id": "sg_1", "text": "Find both birth dates", "status": "completed",
             "results": [{"fact": "Babbage 1791, Lovelace 1815", "source_url": "https://en.wikipedia.org/wiki/Ada", "step": 2}]},
            {"id": "sg_2", "text": "Compare the dates and report which came first", "status": "active", "results": []},
        ],
    })
    agent = _create_mock_agent(plan=plan, initial_url="https://en.wikipedia.org/wiki/Charles_Babbage")
    summary = "Babbage (1791) Lovelace'tan (1815) önce doğdu."
    sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Both dates known", summary_for_user=summary)
    with patch("reflex.agent.resolve_escalation", return_value=sup):
        snap = agent._handle_escalation(trigger_type="verify_done", reason="S1 DONE", s1_decision=None)
    assert snap["status"] == "done"
    assert plan.subgoals[1].status == "completed"
    assert plan.subgoals[1].results[0].fact == summary
    chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
    assert chat_msgs[-1]["message"] == summary


def test_m4_unattended_run_tells_only_s2_not_to_ask_preferences():
    """interactive=False: System 2's context says the user is unavailable; System 1's goal text is unchanged."""
    unattended = _create_mock_agent()
    assert unattended.ctx.user_present is False
    s2_block = unattended.ctx.build_supervisor_context_block([])
    assert "the user is NOT available" in s2_block
    assert "NOT available" not in unattended.ctx.compile_system1_goal()

    unattended.ctx.user_present = True
    assert "NOT available" not in unattended.ctx.build_supervisor_context_block([])
