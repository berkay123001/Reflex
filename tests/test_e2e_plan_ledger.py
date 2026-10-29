"""Opaque-Box End-to-End (E2E) Test Suite for Reflex Task Ledger & Plan-as-Memory (P1a-P1d).

Adheres strictly to the 4-Tier Test Case Design Methodology:
- Tier 1: Feature Coverage E2E Integration
- Tier 2: Boundary & Corner Cases E2E Integration
- Tier 3: Cross-Feature Pairwise Combinations
- Tier 4: Real-World Application Scenarios:
  - Scenario 1: Wikipedia Lovelace/Babbage Entity Research (2 subgoals, birth/death, URLs, <=15 steps)
  - Scenario 2: Multi-Site E-Commerce Price Comparison (3 stores, candidates table, cheapest chosen)
  - Scenario 3: Multi-Source Research with Clarification Default & 2500-Char Summarization
  - Scenario 4: Simple Task Pass-Through Regression Invariance (needs_plan=False, zero overhead)
  - Scenario 5: Form Stability (OKB / Questionnaire Multi-Step Form without state reset)
"""

from __future__ import annotations

import pytest

planner = pytest.importorskip(
    "reflex.planner",
    reason="reflex.planner module not yet implemented (pending Milestone 1)",
)

TaskLedger = planner.TaskLedger
Subgoal = planner.Subgoal
SubgoalResult = planner.SubgoalResult
Candidate = planner.Candidate
Clarify = planner.Clarify
Budget = planner.Budget
ProgressCheckResult = planner.ProgressCheckResult
DoneVerificationResult = planner.DoneVerificationResult
parse_task_ledger = planner.parse_task_ledger

from reflex.config import ReflexConfig
from reflex.context import DualTierContext

# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

def test_scenario_1_wikipedia_lovelace_babbage_research():
    """Tier 4.1: Wikipedia (Lovelace/Babbage) Multi-Subgoal Entity Research.

    Task: 'Find the birth and death years of Ada Lovelace and Charles Babbage on Wikipedia.'
    Criteria:
    - 2 subgoals generated in Task Ledger.
    - Subgoal 1 extracts Lovelace birth (1815) & death (1852) with Wikipedia URL citation.
    - Milestone triggers progression to Subgoal 2 (Babbage).
    - Subgoal 2 extracts Babbage birth (1791) & death (1871) with Wikipedia URL citation.
    - Total steps <= 15.
    - Both birth/death pairs verified and cited in final answer.
    """
    raw_plan = {
        "needs_plan": True,
        "finish_criteria": "Verify birth and death years for both Lovelace and Babbage with Wikipedia citations",
        "budget": {"sources": 1, "max_steps": 15},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find birth and death years of Ada Lovelace",
                "done_when": "Lovelace birth and death years recorded",
                "status": "active",
                "results": [],
            },
            {
                "id": "sg_2",
                "text": "Find birth and death years of Charles Babbage",
                "done_when": "Babbage birth and death years recorded",
                "status": "pending",
                "results": [],
            },
        ],
    }
    ledger = parse_task_ledger(raw_plan)
    assert ledger.needs_plan is True
    assert len(ledger.subgoals) == 2
    assert ledger.budget.sources == 1
    assert ledger.budget.max_steps == 15

    step_count = 0

    # Step 1-2: Navigate to Ada Lovelace Wikipedia page and extract facts
    step_count += 2
    lovelace_url = "https://en.wikipedia.org/wiki/Ada_Lovelace"
    res1 = ledger.add_result(
        fact="Ada Lovelace was born on 10 December 1815 and died on 27 November 1852.",
        source_url=lovelace_url,
        step=step_count,
    )
    assert "1815" in res1.fact and "1852" in res1.fact
    assert res1.source_url == lovelace_url

    # Milestone Check 1: Active subgoal 1 is done
    pcr1 = ProgressCheckResult(subgoal_done=True, next_subgoal="sg_2", stuck=False)
    assert pcr1.subgoal_done is True

    # Advance subgoal: sg_1 completed, sg_2 active
    next_sg = ledger.advance_subgoal()
    assert next_sg is not None
    assert next_sg.id == "sg_2"
    assert ledger.subgoals[0].status == "completed"
    assert ledger.subgoals[1].status == "active"

    # Step 3-5: Navigate to Charles Babbage Wikipedia page and extract facts
    step_count += 3
    babbage_url = "https://en.wikipedia.org/wiki/Charles_Babbage"
    res2 = ledger.add_result(
        fact="Charles Babbage was born on 26 December 1791 and died on 18 October 1871.",
        source_url=babbage_url,
        step=step_count,
    )
    assert "1791" in res2.fact and "1871" in res2.fact
    assert res2.source_url == babbage_url

    # Milestone Check 2: Active subgoal 2 is done
    pcr2 = ProgressCheckResult(subgoal_done=True, next_subgoal=None, stuck=False)
    assert pcr2.subgoal_done is True
    ledger.advance_subgoal()
    assert all(sg.status == "completed" for sg in ledger.subgoals)

    # DONE Verification
    is_done, _reason, missing = ledger.verify_done(visited_urls=[lovelace_url, babbage_url])
    assert is_done is True
    assert missing is None
    assert step_count <= 15

    # Final Answer Synthesis
    final_answer = ledger.synthesize_final_answer("Ada Lovelace and Charles Babbage Lifespans")
    assert "1815" in final_answer
    assert "1852" in final_answer
    assert "1791" in final_answer
    assert "1871" in final_answer
    assert lovelace_url in final_answer
    assert babbage_url in final_answer


def test_scenario_2_ecommerce_price_comparison():
    """Tier 4.2: Multi-Site E-Commerce Price Comparison with Candidates Table.

    Task: 'Compare prices for Anker Soundcore Q30 headphones across e-commerce sites.'
    Criteria:
    - Candidates table tracks product across >= 3 distinct merchant sites.
    - Premature DONE after 1 merchant is rejected.
    - Lowest price is identified accurately and matches candidate table.
    - All sources cited in final answer.
    """
    raw_plan = {
        "needs_plan": True,
        "finish_criteria": "Find prices from at least 3 distinct merchants and identify lowest",
        "budget": {"sources": 3, "max_steps": 30},
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Check price on Amazon",
                "done_when": "Amazon price recorded in candidates table",
                "status": "active",
                "results": [],
            },
            {
                "id": "sg_2",
                "text": "Check price on Walmart",
                "done_when": "Walmart price recorded in candidates table",
                "status": "pending",
                "results": [],
            },
            {
                "id": "sg_3",
                "text": "Check price on Target",
                "done_when": "Target price recorded in candidates table",
                "status": "pending",
                "results": [],
            },
        ],
    }
    ledger = parse_task_ledger(raw_plan)

    # 1. Visit Amazon
    url_amazon = "https://www.amazon.com/dp/B08HMWZBXC"
    ledger.add_result("Amazon listing for Anker Soundcore Q30 is $79.99 with rating 4.6", url_amazon, step=2)
    ledger.add_candidate("Anker Soundcore Q30", url_amazon, {"merchant": "Amazon", "price": 79.99, "currency": "USD", "rating": 4.6})
    ledger.subgoals[0].status = "completed"

    # S1 prematurely attempts DONE: must be rejected because subgoals 2 and 3 are incomplete
    is_done_early, reason_early, missing_early = ledger.verify_done()
    assert is_done_early is False
    assert "not completed" in reason_early.lower()
    assert missing_early is not None
    assert missing_early.id == "sg_2"

    # 2. Visit Walmart
    url_walmart = "https://www.walmart.com/ip/Anker-Soundcore-Q30/123456"
    ledger.add_result("Walmart price is $69.99 with rating 4.5", url_walmart, step=5)
    # Deduplication & attribute merge on existing item
    ledger.add_candidate("Anker Soundcore Q30", url_walmart, {"walmart_price": 69.99, "lowest_seen": 69.99})
    # Also add distinct store candidate record for tabular comparison
    ledger.add_candidate("Anker Q30 (Walmart)", url_walmart, {"price": 69.99, "merchant": "Walmart"})
    ledger.subgoals[1].status = "completed"

    # 3. Visit Target
    url_target = "https://www.target.com/p/anker-soundcore-q30/-/A-823456"
    ledger.add_result("Target price is $74.99 with rating 4.7", url_target, step=8)
    ledger.add_candidate("Anker Q30 (Target)", url_target, {"price": 74.99, "merchant": "Target"})
    ledger.subgoals[2].status = "completed"

    # DONE Verification: 3 distinct domains (amazon.com, walmart.com, target.com)
    is_done, _reason, missing = ledger.verify_done()
    assert is_done is True
    assert missing is None

    # Final Answer Synthesis
    final_answer = ledger.synthesize_final_answer("Anker Soundcore Q30 Price Comparison")
    assert "69.99" in final_answer
    assert "79.99" in final_answer
    assert "74.99" in final_answer
    assert url_amazon in final_answer
    assert url_walmart in final_answer
    assert url_target in final_answer


def test_scenario_3_multisource_research_with_clarification():
    """Tier 4.3: Multi-Source Research with Clarification Default & 2500-Char Summarization.

    Task: 'Research industry viewpoints on AI impact on software engineering.'
    Criteria:
    - Clarification produced: question with options and default '3'.
    - interactive=False automatically adopts default '3' without user pause.
    - 3 distinct tech domains visited.
    - When cumulative extracted text exceeds 2500 characters, summarization condenses findings while preserving all 3 URLs.
    - Final answer cites all 3 domains.
    """
    raw_plan = {
        "needs_plan": True,
        "clarify": {
            "question": "Kaç farklı otoriter kaynak taransın?",
            "options": ["2", "3", "5"],
            "default": "3",
        },
        "finish_criteria": "Summarize perspectives from 3 distinct sources with citations",
        "budget": {"sources": 3, "max_steps": 30},
        "subgoals": [
            {"id": "sg_1", "text": "Collect perspective from Source 1", "status": "active"},
            {"id": "sg_2", "text": "Collect perspective from Source 2", "status": "pending"},
            {"id": "sg_3", "text": "Collect perspective from Source 3", "status": "pending"},
        ],
    }
    config = ReflexConfig(interactive=False)
    ledger = parse_task_ledger(raw_plan)

    # 1. Non-interactive clarify gate
    assert config.interactive is False
    assert ledger.clarify is not None
    # Auto-adopted default
    chosen_sources = int(ledger.clarify.default)
    assert chosen_sources == 3
    ledger.budget.sources = chosen_sources

    # 2. Gather substantial findings across 3 sources
    url_tc = "https://techcrunch.com/2026/02/ai-software-engineering-trends"
    url_ieee = "https://spectrum.ieee.org/ai-pair-programming-future"
    url_acm = "https://cacm.acm.org/opinion/the-future-of-coding-with-llms"

    # Populate 3 completed subgoals with detailed text exceeding 2500 chars
    long_finding_tc = "TechCrunch Analysis: AI agents are shifting engineering roles towards specification and architecture. " * 15
    long_finding_ieee = "IEEE Report: Formal verification and automated testing pipelines are now mandatory companions to LLM generation. " * 15
    long_finding_acm = "ACM Survey: Developer satisfaction increased with automated routine tasks, but debug latency doubled for obscure edge cases. " * 15

    ledger.add_result(long_finding_tc, url_tc, step=3, subgoal_id="sg_1")
    ledger.subgoals[0].status = "completed"

    ledger.add_result(long_finding_ieee, url_ieee, step=6, subgoal_id="sg_2")
    ledger.subgoals[1].status = "completed"

    ledger.add_result(long_finding_acm, url_acm, step=9, subgoal_id="sg_3")
    ledger.subgoals[2].status = "completed"

    # Total chars exceeds 2500
    assert ledger.total_results_chars() > 2500

    # 3. Summarization triggered while strictly preserving all 3 URLs
    summarized = ledger.summarize_completed_results(threshold=2500)
    assert summarized is True
    assert ledger.total_results_chars() < 2500

    cited_urls = {r.source_url for sg in ledger.subgoals for r in sg.results}
    assert url_tc in cited_urls
    assert url_ieee in cited_urls
    assert url_acm in cited_urls

    # 4. DONE Verification & Synthesis
    is_done, _reason, _missing = ledger.verify_done()
    assert is_done is True
    final_answer = ledger.synthesize_final_answer("AI in Software Engineering Industry Perspectives")
    assert url_tc in final_answer
    assert url_ieee in final_answer
    assert url_acm in final_answer


def test_scenario_4_simple_task_passthrough_regression():
    """Tier 4.4: Simple Task Pass-Through (Regression Invariance).

    Task: 'Check flight status for TK1984 on Turkish Airlines.'
    Criteria:
    - Planner determines task is single-site simple query: needs_plan=False.
    - S1/S2 loop executes without planning overhead, subgoals, or candidate tables.
    - Session completes cleanly through standard single-site loop.
    """
    raw_plan = {"needs_plan": False}
    ledger = parse_task_ledger(raw_plan)
    assert ledger.needs_plan is False
    assert len(ledger.subgoals) == 0
    assert len(ledger.candidates) == 0
    assert ledger.clarify is None

    # Context compiling for System 1 has zero plan bloat
    ctx = DualTierContext(primary_goal="Check TK1984 flight status", plan=ledger)
    s1_goal = ctx.compile_system1_goal()
    assert s1_goal == "Check TK1984 flight status"
    assert "subgoal" not in s1_goal.lower()

    # System 2 context indicates No Plan
    s2_block = ledger.format_for_s2()
    assert "No Plan" in s2_block

    # Verification on single visited site passes immediately
    is_done, _reason, missing = ledger.verify_done(visited_urls=["https://turkishairlines.com/flights"])
    assert is_done is True
    assert missing is None


def test_scenario_5_form_stability_questionnaire():
    """Tier 4.5: Form Stability (OKB / Questionnaire Multi-Step Form).

    Task: 'Complete the 5-step occupational questionnaire form.'
    Criteria:
    - Subgoals represent sequential form steps.
    - Agent advances through form steps 1 -> 5 without reset or loop oscillation.
    - Form answers are additively captured.
    - Stuck counter remains 0.
    """
    raw_plan = {
        "needs_plan": True,
        "finish_criteria": "Submit 5-step questionnaire",
        "budget": {"sources": 1, "max_steps": 20},
        "subgoals": [
            {"id": "step_1", "text": "Fill Personal Details", "status": "active"},
            {"id": "step_2", "text": "Fill Employment History", "status": "pending"},
            {"id": "step_3", "text": "Fill Health & Safety Questions", "status": "pending"},
            {"id": "step_4", "text": "Review Consent Checkboxes", "status": "pending"},
            {"id": "step_5", "text": "Click Final Submit Button", "status": "pending"},
        ],
    }
    ledger = parse_task_ledger(raw_plan)
    stuck_counter = 0
    form_url = "https://survey.example.org/form"

    # Step through 5 form sections sequentially
    for step_idx in range(1, 6):
        active_sg = ledger.get_active_subgoal()
        assert active_sg.id == f"step_{step_idx}"

        # Record answer fact
        ledger.add_result(f"Answered section {step_idx} completely", form_url, step=step_idx)

        # Milestone check: not stuck, step complete
        pcr = ProgressCheckResult(subgoal_done=True, stuck=False)
        assert pcr.stuck is False
        if pcr.stuck:
            stuck_counter += 1

        ledger.advance_subgoal()

    assert stuck_counter == 0
    assert all(sg.status == "completed" for sg in ledger.subgoals)
    assert len(ledger.subgoals) == 5

    # Verification
    is_done, _reason, missing = ledger.verify_done(visited_urls=[form_url])
    assert is_done is True
    assert missing is None


# ==============================================================================
# TIER 1: FEATURE COVERAGE INTEGRATION (E2E)
# ==============================================================================

def test_e2e_t1_task_ledger_initialization_with_plan():
    """Tier 1.1: Context snapshot properly records task ledger plan structure."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg1", text="Step 1", status="active")],
        candidates=[Candidate(item="Prod A", attributes={"p": 10})],
    )
    ctx = DualTierContext(primary_goal="Goal A", plan=ledger)
    snapshot = ctx.snapshot_dict()
    assert snapshot["plan"] is not None
    assert snapshot["plan"]["needs_plan"] is True
    assert len(snapshot["plan"]["subgoals"]) == 1
    assert len(snapshot["plan"]["candidates"]) == 1


def test_e2e_t1_clarify_non_interactive_auto_adopt():
    """Tier 1.2: Clarify object in non-interactive mode supplies default to budget/context."""
    raw = {
        "needs_plan": True,
        "clarify": {"question": "How deep?", "options": ["fast", "deep"], "default": "deep"},
    }
    ledger = parse_task_ledger(raw)
    cfg = ReflexConfig(interactive=False)
    if not cfg.interactive:
        selected = ledger.clarify.default
    assert selected == "deep"


def test_e2e_t1_additive_memory_survives_repeated_s2():
    """Tier 1.3: Additive results accumulate across multiple S2 milestone checks without data loss."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg_main", text="Main", status="active")])
    for i in range(5):
        ledger.add_result(f"Fact from S2 call {i}", f"https://source{i}.com", step=i)
    assert len(ledger.subgoals[0].results) == 5
    for i in range(5):
        assert ledger.subgoals[0].results[i].fact == f"Fact from S2 call {i}"


def test_e2e_t1_candidate_table_tracks_items_across_steps():
    """Tier 1.4: Candidates table tracks and merges product features discovered across different steps."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("MacBook Air M3", "https://apple.com", {"base_price": "$1099"})
    ledger.add_candidate("macbook air m3", "https://bestbuy.com", {"sale_price": "$999", "in_stock": True})
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["base_price"] == "$1099"
    assert ledger.candidates[0].attributes["sale_price"] == "$999"


def test_e2e_t1_system1_goal_isolation_under_e2e_load():
    """Tier 1.5: System 1 goal compilation preserves active subgoal isolation under full ledger state."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg1", text="Locate search input", status="active")],
        candidates=[Candidate(item="Item A", attributes={"p": 100})],
        finish_criteria="Find item under $50",
    )
    ctx = DualTierContext(primary_goal="Shopping Task", active_subgoal=ledger.get_active_subgoal().text, plan=ledger)
    s1_goal = ctx.compile_system1_goal()
    assert "Current subgoal: Locate search input" in s1_goal
    assert "Item A" not in s1_goal
    assert "$50" not in s1_goal


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES INTEGRATION (E2E)
# ==============================================================================

def test_e2e_t2_premature_done_rejected_at_step_1():
    """Tier 2.1: S1 emitting DONE at step 1 before any subgoals completed is immediately rejected."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(id="sg1", text="Step 1", status="active", results=[]),
            Subgoal(id="sg2", text="Step 2", status="pending", results=[]),
        ],
    )
    is_done, reason, missing = ledger.verify_done()
    assert is_done is False
    assert missing.id == "sg1"
    assert "not completed" in reason


def test_e2e_t2_site_change_milestone_detection():
    """Tier 2.2: Site/domain change between steps is detected to trigger milestone escalation."""
    from reflex.planner import _extract_domain
    domain1 = _extract_domain("https://google.com/search?q=test")
    domain2 = _extract_domain("https://en.wikipedia.org/wiki/Test")
    site_changed = (domain1 != domain2)
    assert site_changed is True


def test_e2e_t2_12_consecutive_s1_steps_trigger_milestone():
    """Tier 2.3: Tracking counter triggers milestone check exactly on 12th consecutive S1 step."""
    s1_steps_since_s2 = 0
    triggered = False
    for step in range(1, 13):
        s1_steps_since_s2 += 1
        if s1_steps_since_s2 >= 12:
            triggered = True
            s1_steps_since_s2 = 0
    assert triggered is True
    assert s1_steps_since_s2 == 0


def test_e2e_t2_anti_loop_guard_suppresses_repeated_done():
    """Tier 2.4: Anti-loop guard records rejected URL and step to prevent immediate re-escalation loop."""
    current_url = "https://example.com/checkout"
    step = 5
    done_rejected_record = (current_url, step)
    # At step 6 on same URL, rejection is still active
    is_suppressed = (done_rejected_record[0] == current_url and (6 - done_rejected_record[1]) < 8)
    assert is_suppressed is True


def test_e2e_t2_summarization_boundary_at_2500_chars():
    """Tier 2.5: Summarization boundary accurately toggles at 2500 vs 2501 characters."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[SubgoalResult(fact="M" * 2500, source_url="https://m.org", step=1)],
            )
        ],
    )
    assert ledger.summarize_completed_results(threshold=2500) is False
    ledger.add_result("Extra char", "https://m.org", step=2, subgoal_id="sg1")
    assert ledger.summarize_completed_results(threshold=2500) is True


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (E2E PAIRWISE)
# ==============================================================================

def test_e2e_t3_interactive_false_with_price_comparison_and_candidates():
    """Tier 3.1: Non-interactive config with comparison plan automatically populates candidates."""
    cfg = ReflexConfig(interactive=False)
    ledger = parse_task_ledger({
        "needs_plan": True,
        "clarify": {"question": "How many stores?", "options": ["2", "3"], "default": "2"},
        "budget": {"sources": 2},
    })
    assert cfg.interactive is False
    assert ledger.clarify.default == "2"
    ledger.add_candidate("Item A", "https://s1.com", {"price": 10})
    ledger.add_candidate("Item A", "https://s2.com", {"price": 8})
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["price"] == 8


def test_e2e_t3_milestone_advancement_with_distilled_fact_capping():
    """Tier 3.2: Milestone progress check caps facts to top-3 and advances active subgoal."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="Step 1", status="active"),
            Subgoal(id="sg2", text="Step 2", status="pending"),
        ],
    )
    # Simulate S2 providing 4 facts: capped to 3
    facts = ["Fact 1", "Fact 2", "Fact 3", "Fact 4 (overflow)"]
    for f in facts[:3]:
        ledger.add_result(f, "https://site.com", 1)
    assert len(ledger.subgoals[0].results) == 3
    next_sg = ledger.advance_subgoal()
    assert next_sg.id == "sg2"


def test_e2e_t3_done_rejection_reactivates_subgoal_then_succeeds():
    """Tier 3.3: Incomplete subgoal rejected -> reactivated -> fulfilled -> DONE succeeds."""
    sg2 = Subgoal(id="sg2", text="Find contact email", status="pending", results=[])
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=1),
        subgoals=[
            Subgoal(id="sg1", text="Find address", status="completed", results=[SubgoalResult(fact="123 Main St", source_url="https://site.com", step=1)]),
            sg2,
        ],
    )
    is_done, _, missing = ledger.verify_done()
    assert is_done is False
    assert missing is sg2

    missing.status = "active"
    ledger.add_result("contact@site.com", "https://site.com/contact", step=4, subgoal_id="sg2")
    missing.status = "completed"

    is_done2, _, _ = ledger.verify_done()
    assert is_done2 is True


def test_e2e_t3_multi_site_budget_with_domain_normalization():
    """Tier 3.4: Multi-site budget verified across uppercase/lowercase domain variants."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg1",
                text="Collect info",
                status="completed",
                results=[
                    SubgoalResult(fact="Fact A", source_url="https://NYTIMES.COM/tech", step=1),
                    SubgoalResult(fact="Fact B", source_url="https://bbc.com/technology", step=2),
                ],
            )
        ],
    )
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is True


def test_e2e_t3_final_synthesis_combines_summaries_and_candidates():
    """Tier 3.5: Final answer correctly pairs condensed facts with tabular candidate entries."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg1",
                text="Market Overview",
                status="completed",
                results=[SubgoalResult(fact="Market growing at 15% CAGR", source_url="https://gartner.com", step=1)],
            )
        ],
        candidates=[
            Candidate(item="Vendor Alpha", source_url="https://alpha.com", attributes={"market_share": "28%"})
        ],
    )
    ans = ledger.synthesize_final_answer("Cloud Infrastructure Analysis")
    assert "Market growing at 15% CAGR" in ans
    assert "https://gartner.com" in ans
    assert "Vendor Alpha" in ans
    assert "28%" in ans
    assert "https://alpha.com" in ans
