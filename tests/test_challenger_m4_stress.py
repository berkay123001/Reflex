"""Adversarial Empirical Stress Test Suite for Milestone M4 (DONE Verification & Response Synthesis).

Challenger: challenger_m4_1_run2
Scope:
1. DONE rejection under defective combinations:
   - 0 subgoals complete (all pending, or subgoals list empty)
   - 1/3 complete (subgoals 2 and 3 pending)
   - 3/3 complete but 1 (or all) have empty results
   - Candidate table empty or deficit under budget.sources
   - Candidate table with duplicate domains / single domain
   - Distinct sources < budget.sources (domain extraction boundary cases)
2. Subgoal reactivation state machine & invariants:
   - Missing subgoal set to active, other active subgoals reverted to pending
   - active_subgoal string matches reactivated subgoal text
   - agent status remains "ready", never "done"
3. Anti-loop guard (_done_rejected_url & suppress_done):
   - Sliding 8-step window across repeated steps on same URL
   - Window expiration at step 9
   - Navigation to other URL (suppress_done=False) and return (suppress_done=True)
   - URL normalization across fragments (#anchor) and trailing slashes
   - Rejection counter reset on subsequent rejection
   - Clearing guard on user reply
4. synthesize_final_answer stress:
   - Unicode, Turkish characters, emojis, CJK, RTL text
   - Extreme string lengths (20,000+ chars, 50 subgoals, 50 candidates)
   - Query parameters, fragments, ports, parentheses in URLs
   - Markdown injection, HTML tags, special characters
   - Degenerate/empty fields (results=[], attributes={}, empty URLs)
5. Agent execution flows:
   - S1 DONE interception when subgoals incomplete vs complete
   - S2 COMPLETE_TASK interception across all defective states
   - Simple task (needs_plan=False) pass-through
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.planner import (
    Budget,
    Candidate,
    Subgoal,
    SubgoalResult,
    TaskLedger,
    parse_task_ledger,
)
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation

# ==============================================================================
# Helper Factories
# ==============================================================================


def _build_test_agent(
    plan: TaskLedger | None = None,
    initial_url: str = "https://www.example.com/start",
    goal: str = "Empirical Challenger Stress Goal",
) -> ReflexAgent:
    """Instantiate a ReflexAgent with mocked browser, network, and message interpreter."""
    with (
        patch("reflex.agent.Browser") as mock_browser_cls,
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_browser = MagicMock()
        mock_browser_cls.return_value = mock_browser
        mock_browser.observe.return_value = {
            "url": initial_url,
            "title": "Challenger Test Page",
            "text": "Challenger page text",
            "full_text": "Challenger full text content",
            "headings": ["Test Header"],
            "actions": [
                {"id": "click_btn", "kind": "click", "label": "Search Store", "node": 10},
                {"id": "scroll_act", "kind": "scroll", "label": "Scroll", "node": 11},
            ],
            "fingerprint": "fp_challenger",
        }
        mock_browser.fresh.return_value = True

        interp_res = MagicMock(spec=UserMessageInterpretation)
        interp_res.model_dump.return_value = {}
        interp_res.allow_arbitrary_choices = False
        interp_res.constraints = []
        interp_res.plan = plan
        mock_interp.return_value = interp_res

        config = ReflexConfig(max_steps=40, screenshots=False)
        agent = ReflexAgent(url=initial_url, goal=goal, config=config)
        agent.browser = mock_browser
        return agent


def _build_s1_decision(op: str = "DONE", confidence: float = 0.95) -> dict[str, Any]:
    """Helper to create a standard System 1 decision dict."""
    return {
        "operation": op,
        "target": None if op == "DONE" else "click_btn",
        "choice": None if op == "DONE" else "click_btn",
        "confidence": confidence,
        "target_confidence": confidence,
        "latency_ms": 25,
        "model": "turnike",
        "usage": {},
    }


# ==============================================================================
# 1. Defective Combinations in DONE Verification (verify_done)
# ==============================================================================


class TestDoneRejectionDefectiveCombinations:
    """Empirical verification of DONE rejection across all defective combinations."""

    def test_verify_done_zero_subgoals_completed_all_pending(self):
        """0/3 subgoals complete: verify_done must reject and return first pending subgoal."""
        subgoals = [
            Subgoal(id="sg_1", text="Search Store 1", status="pending", results=[]),
            Subgoal(id="sg_2", text="Search Store 2", status="pending", results=[]),
            Subgoal(id="sg_3", text="Search Store 3", status="pending", results=[]),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "is not completed" in reason
        assert missing_sg is not None
        assert missing_sg.id == "sg_1"

    def test_verify_done_one_of_three_completed(self):
        """1/3 subgoals complete: verify_done must reject and return second (first incomplete) subgoal."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store 1",
                status="completed",
                results=[SubgoalResult(fact="Found price $10", source_url="https://store1.com/item", step=1)],
            ),
            Subgoal(id="sg_2", text="Search Store 2", status="pending", results=[]),
            Subgoal(id="sg_3", text="Search Store 3", status="pending", results=[]),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Subgoal 'Search Store 2' is not completed" in reason
        assert missing_sg is not None
        assert missing_sg.id == "sg_2"

    def test_verify_done_two_of_three_completed_middle_pending(self):
        """Subgoals 1 and 3 complete, but subgoal 2 is pending: rejects and flags subgoal 2."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store 1",
                status="completed",
                results=[SubgoalResult(fact="Found price $10", source_url="https://store1.com/item", step=1)],
            ),
            Subgoal(id="sg_2", text="Search Store 2", status="pending", results=[]),
            Subgoal(
                id="sg_3",
                text="Search Store 3",
                status="completed",
                results=[SubgoalResult(fact="Found price $30", source_url="https://store3.com/item", step=3)],
            ),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Subgoal 'Search Store 2' is not completed" in reason
        assert missing_sg is not None
        assert missing_sg.id == "sg_2"

    def test_verify_done_three_of_three_completed_but_one_has_empty_results(self):
        """3/3 marked completed, but subgoal 2 has empty results: must reject and flag subgoal 2."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store 1",
                status="completed",
                results=[SubgoalResult(fact="Store 1 price $10", source_url="https://store1.com/item", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Search Store 2",
                status="completed",
                results=[],  # Defective: marked completed without any verified results!
            ),
            Subgoal(
                id="sg_3",
                text="Search Store 3",
                status="completed",
                results=[SubgoalResult(fact="Store 3 price $30", source_url="https://store3.com/item", step=3)],
            ),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Subgoal 'Search Store 2' has no verified results" in reason
        assert missing_sg is not None
        assert missing_sg.id == "sg_2"

    def test_verify_done_all_completed_but_all_have_empty_results(self):
        """All subgoals completed but none have results: must flag the very first defective subgoal."""
        subgoals = [
            Subgoal(id="sg_1", text="Step 1", status="completed", results=[]),
            Subgoal(id="sg_2", text="Step 2", status="completed", results=[]),
            Subgoal(id="sg_3", text="Step 3", status="completed", results=[]),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Subgoal 'Step 1' has no verified results" in reason
        assert missing_sg is not None
        assert missing_sg.id == "sg_1"

    def test_verify_done_empty_subgoals_list_with_needs_plan(self):
        """Degenerate edge case: needs_plan=True but subgoals list is empty."""
        ledger = TaskLedger(needs_plan=True, subgoals=[], budget=Budget(sources=1))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "budget requires 1" in reason
        assert missing_sg is None

    def test_verify_done_candidate_table_one_candidate_budget_three(self):
        """Candidate table has 1 candidate, but budget requires 3: must reject."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store 1",
                status="completed",
                results=[SubgoalResult(fact="Found item", source_url="https://store1.com/item", step=1)],
            )
        ]
        candidates = [Candidate(item="Sony XM5", source_url="https://store1.com/item", attributes={"price": "$350"})]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, candidates=candidates, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Candidate table has 1 items, expected at least 3" in reason
        assert missing_sg is subgoals[0]

    def test_verify_done_candidate_table_two_candidates_budget_three(self):
        """Candidate table has 2 candidates, but budget requires 3: must reject."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store 1",
                status="completed",
                results=[SubgoalResult(fact="Found item 1", source_url="https://store1.com/item", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Search Store 2",
                status="completed",
                results=[SubgoalResult(fact="Found item 2", source_url="https://store2.com/item", step=2)],
            ),
        ]
        candidates = [
            Candidate(item="Sony XM5 Store 1", source_url="https://store1.com/item", attributes={"price": "$350"}),
            Candidate(item="Sony XM5 Store 2", source_url="https://store2.com/item", attributes={"price": "$360"}),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, candidates=candidates, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Candidate table has 2 items, expected at least 3" in reason
        assert missing_sg is subgoals[1]

    def test_verify_done_three_candidates_from_same_domain(self):
        """Candidate table has 3 items, but all come from the same domain (amazon.com) when budget requires 3 distinct sources."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Search Store",
                status="completed",
                results=[
                    SubgoalResult(fact="Amazon seller 1", source_url="https://www.amazon.com/dp/1", step=1),
                    SubgoalResult(fact="Amazon seller 2", source_url="https://www.amazon.com/dp/2", step=2),
                    SubgoalResult(fact="Amazon seller 3", source_url="https://www.amazon.com/dp/3", step=3),
                ],
            )
        ]
        candidates = [
            Candidate(item="Headphone A", source_url="https://www.amazon.com/dp/1", attributes={"price": "$100"}),
            Candidate(item="Headphone B", source_url="https://www.amazon.com/dp/2", attributes={"price": "$110"}),
            Candidate(item="Headphone C", source_url="https://www.amazon.com/dp/3", attributes={"price": "$120"}),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, candidates=candidates, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        # Even though len(candidates) == 3, distinct domains is only 1 (amazon.com) < budget 3
        assert is_done is False
        assert "Found 1 sources, budget requires 3" in reason
        assert missing_sg is subgoals[0]

    def test_verify_done_sources_below_budget_domain_normalization(self):
        """All subgoals complete with results, but distinct sources < budget.sources under domain normalization."""
        # URLs with port, www, trailing slash, queries all belong to same domain
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Step 1",
                status="completed",
                results=[SubgoalResult(fact="Fact 1", source_url="https://www.techsite.com/article1", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Step 2",
                status="completed",
                results=[SubgoalResult(fact="Fact 2", source_url="http://techsite.com:8080/article2?ref=rss", step=2)],
            ),
            Subgoal(
                id="sg_3",
                text="Step 3",
                status="completed",
                results=[SubgoalResult(fact="Fact 3", source_url="https://techsite.com/article3/#section", step=3)],
            ),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, budget=Budget(sources=2))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is False
        assert "Found 1 sources, budget requires 2" in reason
        assert missing_sg is subgoals[-1]

    def test_verify_done_all_criteria_satisfied_passes(self):
        """All criteria satisfied: 3 subgoals completed, each with results, from 3 distinct sources."""
        subgoals = [
            Subgoal(
                id="sg_1",
                text="Check Store 1",
                status="completed",
                results=[SubgoalResult(fact="Store 1 price", source_url="https://store1.com/p", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Check Store 2",
                status="completed",
                results=[SubgoalResult(fact="Store 2 price", source_url="https://store2.com/p", step=2)],
            ),
            Subgoal(
                id="sg_3",
                text="Check Store 3",
                status="completed",
                results=[SubgoalResult(fact="Store 3 price", source_url="https://store3.com/p", step=3)],
            ),
        ]
        candidates = [
            Candidate(item="Item 1", source_url="https://store1.com/p", attributes={"price": 10}),
            Candidate(item="Item 2", source_url="https://store2.com/p", attributes={"price": 20}),
            Candidate(item="Item 3", source_url="https://store3.com/p", attributes={"price": 30}),
        ]
        ledger = TaskLedger(needs_plan=True, subgoals=subgoals, candidates=candidates, budget=Budget(sources=3))

        is_done, reason, missing_sg = ledger.verify_done()
        assert is_done is True
        assert "All completion criteria satisfied" in reason
        assert missing_sg is None


# ==============================================================================
# 2. Agent Subgoal Reactivation State Machine & Invariants
# ==============================================================================


class TestAgentSubgoalReactivationInvariants:
    """Stress tests verifying agent state invariants upon DONE rejection."""

    def test_reactivation_clears_other_active_and_sets_active_subgoal(self):
        """When DONE is rejected, exactly the missing subgoal is set to active and others reverted to pending."""
        raw = {
            "needs_plan": True,
            "finish_criteria": "Research 2 sources",
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1 Text",
                    "status": "completed",
                    "results": [{"fact": "Fact 1", "source_url": "https://source1.org/1", "step": 1}],
                },
                {
                    "id": "sg_2",
                    "text": "Subgoal 2 Text",
                    "status": "pending",
                    "results": [],
                },
                {
                    "id": "sg_3",
                    "text": "Subgoal 3 Text",
                    "status": "pending",
                    "results": [],
                },
            ],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan)

        # Artificially set sg_3 as active to test single-active invariant enforcement
        plan.subgoals[2].status = "active"
        agent.ctx.active_subgoal = "Subgoal 3 Text"

        sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Premature supervisor finish")
        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify completion", s1_decision=None)

            # 1. Agent status remains ready, never done
            assert snap["status"] == "ready"
            assert agent.status == "ready"

            # 2. sg_2 was the first uncompleted subgoal: must be active
            assert plan.subgoals[1].status == "active"
            assert plan.subgoals[0].status == "completed"
            assert plan.subgoals[2].status == "pending"  # sg_3 demoted to pending!

            # 3. Context active_subgoal is synchronized
            assert agent.ctx.active_subgoal == "Subgoal 2 Text"
            assert agent.ctx.plan.get_active_subgoal() == plan.subgoals[1]

            # 4. Escalation note recorded
            notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
            assert any("Subgoal 2 Text" in m["message"] for m in notes)

    def test_reactivation_of_completed_subgoal_lacking_results(self):
        """When a completed subgoal lacks results, it is reactivated from completed to active."""
        raw = {
            "needs_plan": True,
            "finish_criteria": "Both figures",
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Ada Lovelace",
                    "status": "completed",
                    "results": [],  # Defective!
                },
                {
                    "id": "sg_2",
                    "text": "Charles Babbage",
                    "status": "completed",
                    "results": [{"fact": "Charles Babbage 1791", "source_url": "https://source2.org", "step": 2}],
                },
            ],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan)

        s1_done = _build_s1_decision("DONE")
        sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Looks complete", summary_for_user="Done.")
        with (
            patch("reflex.agent.choose_action", return_value=s1_done),
            patch("reflex.agent.resolve_escalation", return_value=sup),
        ):
            snap = agent.step()
            assert snap["status"] == "ready"
            assert agent.status == "ready"

            sg1 = plan.subgoals[0]
            assert sg1.status == "active"
            assert agent.ctx.active_subgoal == "Ada Lovelace"

            notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
            assert any("has no verified results" in m["message"] for m in notes)


# ==============================================================================
# 3. Anti-Loop Guard Stress & URL Normalization
# ==============================================================================


class TestAntiLoopGuardStress:
    """Stress tests for anti-loop guard across repeated steps, URL variations, and reset lifecycle."""

    def test_anti_loop_guard_eight_step_sliding_window(self):
        """Guard suppresses DONE for exactly 8 steps on same URL, and expires at step 9."""
        raw = {
            "needs_plan": True,
            "finish_criteria": "Research criteria",
            "budget": {"sources": 2, "max_steps": 30},
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Pending Subgoal",
                    "status": "pending",
                    "results": [],
                }
            ],
        }
        plan = parse_task_ledger(raw)
        test_url = "https://example.com/checkout/step1"
        agent = _build_test_agent(plan=plan, initial_url=test_url)

        # Trigger DONE rejection via S2 COMPLETE_TASK at initial state (len(history) == 0)
        sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Early done")
        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)

        assert agent._done_rejected_url == (test_url, 0)

        # Verify suppress_done calculation directly and when step() executes
        for h_len in range(9):
            agent.history = [{"step": i + 1, "operation": "CLICK"} for i in range(h_len)]
            cur_norm_url = test_url.split("#")[0].rstrip("/")
            suppress_done = (
                agent._done_rejected_url is not None
                and agent._done_rejected_url[0] == cur_norm_url
                and (len(agent.history) - agent._done_rejected_url[1]) <= 8
            )
            assert suppress_done is True, f"Failed at history length {h_len}: expected suppress_done=True"

        # At history length 9: (9 - 0) == 9 > 8 -> must expire to False!
        agent.history = [{"step": i + 1, "operation": "CLICK"} for i in range(9)]
        cur_norm_url = test_url.split("#")[0].rstrip("/")
        suppress_done = (
            agent._done_rejected_url is not None
            and agent._done_rejected_url[0] == cur_norm_url
            and (len(agent.history) - agent._done_rejected_url[1]) <= 8
        )
        assert suppress_done is False, "Failed at history length 9: expected suppress_done=False"

    def test_anti_loop_guard_url_normalization_with_fragments_and_slashes(self):
        """URL fragments (#anchor) and trailing slashes are normalized so suppress_done applies seamlessly."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []}],
        }
        plan = parse_task_ledger(raw)
        base_url = "https://example.com/products/item"
        agent = _build_test_agent(plan=plan, initial_url=base_url + "#section-reviews")

        # Rejection recorded with fragment stripped
        sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Early done")
        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)

        assert agent._done_rejected_url == (base_url, 0)

        # 1. Page changes fragment to #specs -> must still suppress DONE
        agent.page["url"] = base_url + "#section-specs"
        cur_norm = str(agent.page.get("url", "")).split("#")[0].rstrip("/")
        suppress = (
            agent._done_rejected_url is not None
            and agent._done_rejected_url[0] == cur_norm
            and (len(agent.history) - agent._done_rejected_url[1]) <= 8
        )
        assert suppress is True

        # 2. Page has trailing slash -> must still suppress DONE
        agent.page["url"] = base_url + "/"
        cur_norm = str(agent.page.get("url", "")).split("#")[0].rstrip("/")
        suppress = (
            agent._done_rejected_url is not None
            and agent._done_rejected_url[0] == cur_norm
            and (len(agent.history) - agent._done_rejected_url[1]) <= 8
        )
        assert suppress is True

        # 3. Page changes to a different path -> must NOT suppress DONE
        agent.page["url"] = "https://example.com/other-page"
        cur_norm = str(agent.page.get("url", "")).split("#")[0].rstrip("/")
        suppress = (
            agent._done_rejected_url is not None
            and agent._done_rejected_url[0] == cur_norm
            and (len(agent.history) - agent._done_rejected_url[1]) <= 8
        )
        assert suppress is False

        # 4. Page returns to original base URL -> must resume suppress DONE (within 8 steps)
        agent.page["url"] = base_url
        cur_norm = str(agent.page.get("url", "")).split("#")[0].rstrip("/")
        suppress = (
            agent._done_rejected_url is not None
            and agent._done_rejected_url[0] == cur_norm
            and (len(agent.history) - agent._done_rejected_url[1]) <= 8
        )
        assert suppress is True

    def test_repeated_complete_task_is_accepted_after_one_rejection(self):
        """The ledger sends a premature COMPLETE_TASK back once (setting the anti-loop guard); System 2 insisting
        again is final, so an unmet budget cannot lock the task."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 25},
            "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []}],
        }
        plan = parse_task_ledger(raw)
        test_url = "https://example.com/checkout"
        agent = _build_test_agent(plan=plan, initial_url=test_url)

        # First rejection at history=0
        sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Early done")
        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            agent._handle_escalation(trigger_type="verify_done", reason="Verify 1", s1_decision=None)
        assert agent._done_rejected_url == (test_url, 0)

        # Advance 4 steps
        for _ in range(4):
            agent.history.append({"step": len(agent.history) + 1, "operation": "CLICK"})

        assert len(agent.history) == 4

        # Second rejection at history=4
        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            agent._handle_escalation(trigger_type="verify_done", reason="Verify 2", s1_decision=None)

        assert agent.status == "done"
        assert agent._done_rejected_url == (test_url, 0)
        notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
        assert any("accepting" in m["message"] for m in notes)

    def test_anti_loop_guard_cleared_on_new_goal(self):
        """When user provides a new goal via reply_user (intent=NEW_GOAL), _done_rejected_url is cleanly reset to None."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []}],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan, initial_url="https://example.com/page")

        # Manually set guard
        agent._done_rejected_url = ("https://example.com/page", 2)

        # Simulate user replying with a NEW_GOAL
        mock_interp = UserMessageInterpretation(
            intent="NEW_GOAL",
            new_goal="Search completely new topic",
            constraints=[],
            needs_plan=True,
            plan=plan,
        )
        with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
            agent.reply_user("Let's look at another topic entirely")
            assert agent._done_rejected_url is None

    def test_anti_loop_guard_retained_on_provide_fact_same_page(self):
        """When user provides a fact on the same page, anti-loop suppression persists to prevent immediate premature DONE."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "pending", "results": []}],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan, initial_url="https://example.com/page")

        # Manually set guard
        agent._done_rejected_url = ("https://example.com/page", 2)

        # User provides fact
        mock_interp = UserMessageInterpretation(
            intent="PROVIDE_FACT",
            fact_value="Store XYZ",
            constraints=[],
            needs_plan=True,
            plan=plan,
        )
        with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
            agent.reply_user("Store XYZ")
            # Guard remains active on this page
            assert agent._done_rejected_url == ("https://example.com/page", 2)


# ==============================================================================
# 4. synthesize_final_answer Stress, Unicode, Long Strings, Query Params
# ==============================================================================


class TestSynthesizeFinalAnswerStress:
    """Stress tests for synthesize_final_answer handling of extreme inputs."""

    def test_synthesize_unicode_emojis_and_turkish(self):
        """Turkish characters, symbols, emojis, and non-ASCII text format correctly without corruption."""
        sg = Subgoal(id="sg_1", text="Üniversite Araştırması", status="completed")
        ledger = TaskLedger(
            needs_plan=True,
            subgoals=[sg],
            finish_criteria="Türkçe ve emoji karşılaştırma testi 🇹🇷",
        )
        ledger.add_result(
            "İstanbul Teknik Üniversitesi (İTÜ) ve Boğaziçi Üniversitesi verileri karşılaştırıldı. Şık ve hızlı! 🚀 🔥",
            "https://itu.edu.tr/araştırma?bölüm=bilgisayar&yıl=2026",
            step=2,
        )
        ledger.add_candidate(
            "MacBook Pro M3 Max (Uzay Grisi) 💻",
            "https://apple.com/tr/shop/macbook-pro?model=16inç",
            {"fiyat": "124.999 ₺", "puan": "4.9/5 ★", "özellikler": "64GB RAM, 2TB SSD, Türkçe Q klavye"},
        )

        res = ledger.synthesize_final_answer("Dizüstü Bilgisayar Karşılaştırması 💻")
        assert "Dizüstü Bilgisayar Karşılaştırması 💻" in res
        assert "İstanbul Teknik Üniversitesi (İTÜ)" in res
        assert "🚀 🔥" in res
        assert "https://itu.edu.tr/araştırma?bölüm=bilgisayar&yıl=2026" in res
        assert "MacBook Pro M3 Max (Uzay Grisi) 💻" in res
        assert "124.999 ₺" in res
        assert "Türkçe Q klavye" in res

    def test_synthesize_international_cjk_arabic_cyrillic(self):
        """CJK, Arabic RTL, Cyrillic, and Greek alphabets format cleanly without crashing."""
        sg = Subgoal(id="sg_1", text="Global Research Subgoal", status="completed")
        ledger = TaskLedger(needs_plan=True, subgoals=[sg])
        ledger.add_result(
            "CJK: 東京大学の研究成果 (2026). Arabic: جامعة القاهرة \u200f(2026). Cyrillic: МГУ им. Ломоносова.",
            "https://global-research.org/study?lang=zh-CN&ref=αβγ",
            step=3,
        )
        ledger.add_candidate(
            "ソニー ワイヤレスヘッドホン",
            "https://sony.jp/item/1000xm5",
            {"価格": "45,000 円", "評価": "4.8"},
        )

        res = ledger.synthesize_final_answer("Global Study / 全球研究 / بحث عالمي")
        assert "東京大学" in res
        assert "جامعة القاهرة" in res
        assert "МГУ им. Ломоносова" in res
        assert "ソニー ワイヤレスヘッドホン" in res
        assert "45,000 円" in res

    def test_synthesize_extreme_string_lengths(self):
        """Extremely long facts (25,000+ chars) and primary goals format without OOM or truncation errors."""
        long_fact = "Extremely long fact content: " + ("abcdefghijklmnopqrstuvwxyz0123456789 " * 700)
        long_goal = "Massive Goal: " + ("GOAL " * 500)

        sg = Subgoal(id="sg_1", text="Massive Subgoal", status="completed")
        ledger = TaskLedger(needs_plan=True, subgoals=[sg])
        ledger.add_result(long_fact, "https://long-data.org/dump", step=10)
        ledger.add_candidate(
            "Massive Item",
            "https://long-data.org/dump",
            {"description": "attr " * 500},
        )

        res = ledger.synthesize_final_answer(long_goal)
        assert len(res) > 30000
        assert long_fact in res
        assert "https://long-data.org/dump" in res

    def test_synthesize_complex_urls_query_params_and_parentheses(self):
        """URLs containing query params, port numbers, IPv6 brackets, and Wikipedia parentheses format properly."""
        wiki_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        query_url = "https://store.com:8443/search?query=laptop&filter=price<1000&sort=desc#spec-table"

        sg = Subgoal(id="sg_1", text="Wikipedia Search", status="completed")
        ledger = TaskLedger(needs_plan=True, subgoals=[sg])
        ledger.add_result("Wikipedia definition of Python", wiki_url, step=1)
        ledger.add_candidate(
            "ThinkPad T14",
            query_url,
            {"price": "$899", "url": query_url},
        )

        res = ledger.synthesize_final_answer("Python and ThinkPad")
        assert wiki_url in res
        assert query_url in res
        assert "*(Kaynak: https://en.wikipedia.org/wiki/Python_(programming_language))*" in res

    def test_synthesize_markdown_and_html_special_characters(self):
        """Facts containing markdown formatting, HTML tags, and code blocks do not break formatting."""
        sg = Subgoal(id="sg_1", text="Security Subgoal", status="completed")
        ledger = TaskLedger(needs_plan=True, subgoals=[sg])
        ledger.add_result(
            "Found code snippet: `console.log('<script>alert(\"XSS\")</script>');` and **bold text** with *italics*.",
            "https://security.org/vulnerabilities",
            step=1,
        )
        ledger.add_candidate(
            "<CustomWidget id='0' />",
            "https://security.org/widget",
            {"tag": "<div>safe</div>", "formula": "a < b and c > d"},
        )

        res = ledger.synthesize_final_answer("Security Review")
        assert "<script>alert(\"XSS\")</script>" in res
        assert "<CustomWidget id='0' />" in res
        assert "<div>safe</div>" in res

    def test_synthesize_degenerate_empty_cases(self):
        """Empty subgoals, empty results, empty candidate attributes produce expected fallback strings."""
        # 1. No subgoals and no candidates
        ledger_empty = TaskLedger(needs_plan=True)
        res_empty = ledger_empty.synthesize_final_answer("Empty Ledger Goal")
        assert res_empty == "### Görev Sonucu: Empty Ledger Goal\n"

        # 2. Subgoal with empty results list -> outputs '(Bulgu kaydedilmedi)'
        ledger_no_results = TaskLedger(
            needs_plan=True,
            subgoals=[Subgoal(id="sg_1", text="Unrecorded Subgoal", results=[])],
        )
        res_no_results = ledger_no_results.synthesize_final_answer("Goal")
        assert "• (Bulgu kaydedilmedi)" in res_no_results

        # 3. Result with empty source_url -> outputs fact without *(Kaynak: )*
        sg = Subgoal(id="sg_1", text="Subgoal 1", status="completed")
        ledger_no_src = TaskLedger(needs_plan=True, subgoals=[sg])
        ledger_no_src.add_result("Fact with no URL", "", step=1)
        res_no_src = ledger_no_src.synthesize_final_answer("Goal")
        assert "• Fact with no URL" in res_no_src
        assert "*(Kaynak:" not in res_no_src

        # 4. Candidate with empty attributes -> outputs 'Detay yok'
        ledger_no_attrs = TaskLedger(
            needs_plan=True,
            candidates=[Candidate(item="Item With No Attrs", source_url="", attributes={})],
        )
        res_no_attrs = ledger_no_attrs.synthesize_final_answer("Goal")
        assert "- **Item With No Attrs**: Detay yok" in res_no_attrs


# ==============================================================================
# 5. Agent-Level Multi-Tier Interception Stress
# ==============================================================================


class TestAgentLevelInterceptionStress:
    """Empirical verification that premature completion attempts from S1 or S2 are reliably intercepted."""

    def test_s1_done_intercepted_when_subgoals_incomplete_escalates(self):
        """When S1 emits DONE but subgoals are incomplete, agent intercepts and escalates to S2 supervisor."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1 Incomplete",
                    "status": "active",
                    "results": [],
                }
            ],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan)

        s1_done = _build_s1_decision("DONE", confidence=0.99)
        sup_update = SupervisorDecision(
            action_type="UPDATE_SUBGOAL",
            updated_subgoal="Keep searching Store 1",
            reasoning="Subgoal not finished",
        )

        with (
            patch("reflex.agent.choose_action", return_value=s1_done),
            patch("reflex.agent.resolve_escalation", return_value=sup_update) as mock_sup,
        ):
            agent.step()
            # Must NOT set status to done!
            assert agent.status == "ready"
            # S2 supervisor was consulted because task had an active incomplete subgoal
            assert mock_sup.called
            assert mock_sup.call_args[1]["trigger_type"] == "progress_milestone"

    def test_s1_done_intercepted_and_rejected_when_all_subgoals_completed_but_sources_deficit(self):
        """S1 DONE with all subgoals completed goes to System 2 with the ledger deficit as evidence; System 2's first
        COMPLETE_TASK is sent back with the budget reason."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 3, "max_steps": 20},  # Requires 3 sources
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1",
                    "status": "completed",
                    "results": [{"fact": "Fact 1", "source_url": "https://single-domain.com/1", "step": 1}],
                },
                {
                    "id": "sg_2",
                    "text": "Subgoal 2",
                    "status": "completed",
                    "results": [{"fact": "Fact 2", "source_url": "https://single-domain.com/2", "step": 2}],
                },
            ],
        }
        plan = parse_task_ledger(raw)
        agent = _build_test_agent(plan=plan, initial_url="https://single-domain.com/1")
        agent.ctx.active_subgoal = ""

        s1_done = _build_s1_decision("DONE", confidence=0.95)

        sup = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Looks complete", summary_for_user="Done.")
        with (
            patch("reflex.agent.choose_action", return_value=s1_done),
            patch("reflex.agent.resolve_escalation", return_value=sup) as s2,
        ):
            snap = agent.step()
            assert snap["status"] == "ready"
            assert agent.status == "ready"

            # Anti-loop guard set
            assert agent._done_rejected_url is not None

            # Subgoal 2 reactivated
            assert plan.subgoals[1].status == "active"
            assert agent.ctx.active_subgoal == "Subgoal 2"

            notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
            assert any("budget requires 3" in m["message"] for m in notes)
            assert "Ledger check: Found 1 sources, budget requires 3" in s2.call_args.kwargs["escalation_reason"]

    def test_s2_complete_task_intercepted_and_rejected_when_sources_deficit(self):
        """When S2 returns COMPLETE_TASK prematurely, agent intercepts via _verify_and_complete_task_from_plan and rejects."""
        raw = {
            "needs_plan": True,
            "budget": {"sources": 2, "max_steps": 20},
            "subgoals": [
                {
                    "id": "sg_1",
                    "text": "Subgoal 1",
                    "status": "completed",
                    "results": [{"fact": "Fact", "source_url": "https://only-one.com/article", "step": 1}],
                }
            ],
        }
        plan = parse_task_ledger(raw)
        # initial_url has same domain only-one.com so distinct sources is 1 < 2
        agent = _build_test_agent(plan=plan, initial_url="https://only-one.com/article")

        sup_decision = SupervisorDecision(action_type="COMPLETE_TASK", reasoning="Premature supervisor finish")

        with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
            snap = agent._handle_escalation(trigger_type="verify_done", reason="Verify", s1_decision=None)
            assert snap["status"] == "ready"
            assert agent.status == "ready"

            notes = [m for m in agent.ctx.messages if m.get("kind") == "escalation_note"]
            assert any("Found 1 sources, budget requires 2" in m["message"] for m in notes)

    def test_simple_task_bypass_completes_without_plan_verification(self):
        """needs_plan=False simple task is not hindered by plan verification and completes directly."""
        plan = parse_task_ledger({"needs_plan": False})
        agent = _build_test_agent(plan=plan)
        agent.config.verify_done_with_supervisor = False
        agent.history.append({"step": 1, "operation": "GOTO_URL", "url": "https://simple.com"})

        s1_done = _build_s1_decision("DONE", confidence=0.98)
        with patch("reflex.agent.choose_action", return_value=s1_done):
            snap = agent.step()
            assert snap["status"] == "done"
            assert agent.status == "done"
            assert any("high confidence verification passed" in m["message"] for m in agent.ctx.messages)
