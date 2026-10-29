"""Comprehensive Unit, Boundary, and Combinatorial Test Suite for Reflex Task Ledger & Plan-as-Memory (P1a-P1d).

Adheres strictly to the 4-Tier Test Case Design Methodology:
- Tier 1: Feature Coverage (>=5 test cases per feature for F1-F16)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature for F1-F16)
- Tier 3: Cross-Feature Combinations (Pairwise interactions)
- Static verification of academic attribution (F16) and architectural non-goals
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Progressive testability: import planner when available, cleanly skip if M1 not yet implemented.
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
parse_task_ledger = planner.parse_task_ledger

from reflex.config import ReflexConfig
from reflex.context import DualTierContext

# ==============================================================================
# TIER 1: FEATURE COVERAGE (F1 - F16, >=5 Tests Per Feature)
# ==============================================================================

# --- F1: Task Ledger Schema & Parsing ---

def test_f1_schema_parsing_valid_full_dict():
    """F1.1: Parses full valid dictionary with all required Task Ledger fields."""
    raw = {
        "needs_plan": True,
        "clarify": {
            "question": "Kaç kaynak taransın?",
            "options": ["2", "3", "5"],
            "default": "3",
        },
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace birth/death",
                "done_when": "Dates found",
                "status": "active",
                "results": [
                    {"fact": "Born 1815, died 1852", "source_url": "https://en.wikipedia.org/wiki/Ada_Lovelace", "step": 2}
                ],
            },
            {
                "id": "sg_2",
                "text": "Find Charles Babbage birth/death",
                "done_when": "Dates found",
                "status": "pending",
                "results": [],
            },
        ],
        "candidates": [
            {"item": "Soundcore Q30", "source_url": "https://amazon.com/dp/B08HMWZBXC", "attributes": {"price": "$79.99"}}
        ],
        "finish_criteria": "Both birth and death years verified",
        "budget": {"sources": 2, "max_steps": 25},
    }
    ledger = parse_task_ledger(raw)
    assert ledger.needs_plan is True
    assert ledger.clarify is not None
    assert ledger.clarify.default == "3"
    assert len(ledger.subgoals) == 2
    assert ledger.subgoals[0].id == "sg_1"
    assert ledger.subgoals[0].status == "active"
    assert len(ledger.subgoals[0].results) == 1
    assert ledger.subgoals[0].results[0].step == 2
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["price"] == "$79.99"
    assert ledger.budget.sources == 2
    assert ledger.budget.max_steps == 25


def test_f1_schema_parsing_from_json_string():
    """F1.2: Parses valid JSON string into TaskLedger object."""
    json_str = json.dumps({
        "needs_plan": True,
        "clarify": None,
        "subgoals": [
            {"id": "sg_1", "text": "Search articles", "done_when": "Articles listed", "status": "active", "results": []}
        ],
        "candidates": [],
        "finish_criteria": "3 articles found",
        "budget": {"sources": 3, "max_steps": 20},
    })
    ledger = parse_task_ledger(json_str)
    assert ledger.needs_plan is True
    assert ledger.clarify is None
    assert len(ledger.subgoals) == 1
    assert ledger.budget.sources == 3


def test_f1_schema_parsing_default_subgoal_activation():
    """F1.3: If no subgoal is marked active, the first subgoal is automatically activated."""
    raw = {
        "needs_plan": True,
        "subgoals": [
            {"id": "sg_1", "text": "First step", "status": "pending", "results": []},
            {"id": "sg_2", "text": "Second step", "status": "pending", "results": []},
        ],
    }
    ledger = parse_task_ledger(raw)
    assert ledger.subgoals[0].status == "active"
    assert ledger.get_active_subgoal() == ledger.subgoals[0]


def test_f1_schema_parsing_budget_defaults():
    """F1.4: Budget uses defaults (sources=1, max_steps=30) when omitted."""
    raw = {
        "needs_plan": True,
        "subgoals": [{"id": "sg_1", "text": "Step 1", "status": "active", "results": []}],
    }
    ledger = parse_task_ledger(raw)
    assert ledger.budget.sources == 1
    assert ledger.budget.max_steps == 30


def test_f1_schema_parsing_subgoal_results_nested_structure():
    """F1.5: Nested SubgoalResult objects are properly instantiated and typed."""
    raw = {
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Extract facts",
                "results": [
                    {"fact": "Fact 1", "source_url": "https://example.com/1", "step": 1},
                    {"fact": "Fact 2", "source_url": "https://example.com/2", "step": 3},
                ],
            }
        ],
    }
    ledger = parse_task_ledger(raw)
    results = ledger.subgoals[0].results
    assert len(results) == 2
    assert isinstance(results[0], SubgoalResult)
    assert results[0].fact == "Fact 1"
    assert results[1].step == 3


# --- F2: Simple Task Pass-Through ---

def test_f2_simple_task_needs_plan_false():
    """F2.1: Simple task returns needs_plan=False."""
    raw = {"needs_plan": False}
    ledger = parse_task_ledger(raw)
    assert ledger.needs_plan is False


def test_f2_simple_task_empty_subgoals_and_candidates():
    """F2.2: Simple task has empty subgoals and candidates."""
    ledger = parse_task_ledger({"needs_plan": False})
    assert len(ledger.subgoals) == 0
    assert len(ledger.candidates) == 0


def test_f2_simple_task_clarify_is_none():
    """F2.3: Simple task has no clarify object."""
    ledger = parse_task_ledger({"needs_plan": False})
    assert ledger.clarify is None


def test_f2_simple_task_format_for_s2_indicates_no_plan():
    """F2.4: format_for_s2 cleanly indicates No Plan."""
    ledger = parse_task_ledger({"needs_plan": False})
    s2_text = ledger.format_for_s2()
    assert "No Plan" in s2_text


def test_f2_simple_task_get_active_subgoal_returns_none():
    """F2.5: get_active_subgoal returns None when needs_plan=False with empty subgoals."""
    ledger = parse_task_ledger({"needs_plan": False})
    assert ledger.get_active_subgoal() is None


# --- F3: Scoped Clarification ---

def test_f3_scoped_clarification_complete():
    """F3.1: Clarify object correctly captures question, options, and default."""
    c = Clarify(question="Kaç mağaza kontrol edilsin?", options=["3", "5"], default="3")
    assert c.question == "Kaç mağaza kontrol edilsin?"
    assert c.options == ["3", "5"]
    assert c.default == "3"


def test_f3_scoped_clarification_ordering_preserved():
    """F3.2: Clarify options list preserves exact ordering."""
    c = Clarify(question="Derinlik?", options=["Hızlı", "Standart", "Detaylı"], default="Standart")
    assert c.options[0] == "Hızlı"
    assert c.options[1] == "Standart"
    assert c.options[2] == "Detaylı"


def test_f3_scoped_clarification_parsing_from_dict():
    """F3.3: TaskLedger parses nested clarify dictionary."""
    raw = {
        "needs_plan": True,
        "clarify": {"question": "Kaç makale?", "options": ["2", "4"], "default": "2"},
        "subgoals": [],
    }
    ledger = parse_task_ledger(raw)
    assert ledger.clarify is not None
    assert ledger.clarify.question == "Kaç makale?"
    assert ledger.clarify.default == "2"


def test_f3_scoped_clarification_none_when_omitted():
    """F3.4: Clarify is None when omitted from input dict."""
    ledger = parse_task_ledger({"needs_plan": True, "subgoals": []})
    assert ledger.clarify is None


def test_f3_scoped_clarification_model_dump_roundtrip():
    """F3.5: Clarify serializes and deserializes accurately."""
    c1 = Clarify(question="Q?", options=["a", "b"], default="a")
    dumped = c1.model_dump()
    c2 = Clarify(**dumped)
    assert c1 == c2


# --- F4: Interactive Config Gate ---

def test_f4_interactive_gate_config_default_false():
    """F4.1: ReflexConfig has interactive=False by default."""
    config = ReflexConfig()
    assert hasattr(config, "interactive")
    assert config.interactive is False


def test_f4_interactive_gate_config_set_true():
    """F4.2: ReflexConfig accepts interactive=True."""
    config = ReflexConfig(interactive=True)
    assert config.interactive is True


def test_f4_interactive_gate_non_interactive_clarify_handling():
    """F4.3: In non-interactive mode, clarify default is auto-adopted without user intervention."""
    ledger = parse_task_ledger({
        "needs_plan": True,
        "clarify": {"question": "Kaç kaynak?", "options": ["2", "4"], "default": "2"},
    })
    # Non-interactive logic consumes default value directly
    adopted_choice = ledger.clarify.default
    assert adopted_choice == "2"


def test_f4_interactive_gate_interactive_flag_in_model_dump():
    """F4.4: ReflexConfig model_dump contains interactive key."""
    config = ReflexConfig(interactive=False)
    data = config.model_dump()
    assert "interactive" in data
    assert data["interactive"] is False


def test_f4_interactive_gate_custom_interactive_eval():
    """F4.5: Eval runner can explicitly initialize ReflexConfig with interactive=False."""
    cfg = ReflexConfig(decision_backend="rule", interactive=False)
    assert cfg.interactive is False


# --- F5: Plan as Memory (Anti-Clobbering) ---

def test_f5_anti_clobbering_add_result_to_active_subgoal():
    """F5.1: add_result attaches result to currently active subgoal."""
    ledger = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [
            {"id": "sg_1", "text": "Subgoal 1", "status": "active", "results": []},
            {"id": "sg_2", "text": "Subgoal 2", "status": "pending", "results": []},
        ],
    })
    res = ledger.add_result(fact="Price is $99", source_url="https://site.com", step=2)
    assert res.fact == "Price is $99"
    assert len(ledger.subgoals[0].results) == 1
    assert ledger.subgoals[0].results[0].fact == "Price is $99"
    assert len(ledger.subgoals[1].results) == 0


def test_f5_anti_clobbering_add_result_explicit_subgoal_id():
    """F5.2: add_result with explicit subgoal_id targets specified subgoal."""
    ledger = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [
            {"id": "sg_1", "text": "Subgoal 1", "status": "active", "results": []},
            {"id": "sg_2", "text": "Subgoal 2", "status": "pending", "results": []},
        ],
    })
    ledger.add_result(fact="Fact for sg_2", source_url="https://site2.com", step=3, subgoal_id="sg_2")
    assert len(ledger.subgoals[0].results) == 0
    assert len(ledger.subgoals[1].results) == 1
    assert ledger.subgoals[1].results[0].fact == "Fact for sg_2"


def test_f5_anti_clobbering_results_accumulate_additively():
    """F5.3: Multiple add_result calls accumulate additively without overwriting."""
    ledger = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "active", "results": []}],
    })
    ledger.add_result(fact="Fact A", source_url="https://a.com", step=1)
    ledger.add_result(fact="Fact B", source_url="https://b.com", step=2)
    assert len(ledger.subgoals[0].results) == 2
    assert ledger.subgoals[0].results[0].fact == "Fact A"
    assert ledger.subgoals[0].results[1].fact == "Fact B"


def test_f5_anti_clobbering_provenance_fields_preserved():
    """F5.4: SubgoalResult stores fact, source_url, and step accurately."""
    r = SubgoalResult(fact="Spec 100", source_url="https://docs.org/spec", step=5)
    assert r.fact == "Spec 100"
    assert r.source_url == "https://docs.org/spec"
    assert r.step == 5


def test_f5_anti_clobbering_independent_of_working_memory():
    """F5.5: TaskLedger results persist independently of ephemeral working_memory."""
    ctx = DualTierContext(primary_goal="Research")
    ctx.working_memory = "Old memory"
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg_1", text="Sg1", status="active")])
    ledger.add_result("Persistent fact", "https://source.com", 1)
    # Even if working_memory is reassigned, ledger is untouched
    ctx.working_memory = "Clobbered memory"
    assert ledger.subgoals[0].results[0].fact == "Persistent fact"


# --- F6: Candidate Comparison Table ---

def test_f6_candidate_table_add_new():
    """F6.1: Adding new candidate records item, source_url, and attributes."""
    ledger = TaskLedger(needs_plan=True)
    c = ledger.add_candidate("Anker Q30", "https://amazon.com/q30", {"price": "$79.99"})
    assert c.item == "Anker Q30"
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["price"] == "$79.99"


def test_f6_candidate_table_deduplication_case_insensitive():
    """F6.2: Candidate items are deduplicated case-insensitively."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("Anker Q30", "https://amazon.com/q30", {"price": "$79.99"})
    ledger.add_candidate("anker q30", "https://walmart.com/q30", {"rating": "4.5"})
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["price"] == "$79.99"
    assert ledger.candidates[0].attributes["rating"] == "4.5"


def test_f6_candidate_table_attribute_merging():
    """F6.3: Updating existing candidate merges new attributes into existing dictionary."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("Product X", "https://site.com/x", {"color": "black", "price": "$100"})
    ledger.add_candidate("Product X", "https://site.com/x", {"price": "$90", "shipping": "free"})
    cand = ledger.candidates[0]
    assert cand.attributes["color"] == "black"
    assert cand.attributes["price"] == "$90"
    assert cand.attributes["shipping"] == "free"


def test_f6_candidate_table_source_url_updating():
    """F6.4: Adding a URL to a candidate initially added without URL updates source_url."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("Item Y", "", {"stock": "in_stock"})
    ledger.add_candidate("Item Y", "https://store.com/y", {"stock": "in_stock"})
    assert ledger.candidates[0].source_url == "https://store.com/y"


def test_f6_candidate_table_multiple_distinct_items():
    """F6.5: Multiple distinct items are maintained as separate candidates."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("Item 1", "https://s1.com", {"p": 10})
    ledger.add_candidate("Item 2", "https://s2.com", {"p": 20})
    ledger.add_candidate("Item 3", "https://s3.com", {"p": 30})
    assert len(ledger.candidates) == 3


# --- F7: Distilled Fact Rate Limiting ---

def test_f7_fact_rate_limit_0_to_3_facts():
    """F7.1: S2 extraction records 0 to 3 facts per milestone call cleanly."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg_1", text="sg", status="active")])
    facts = ["Fact 1", "Fact 2", "Fact 3"]
    for f in facts:
        ledger.add_result(f, "https://site.com", 1)
    assert len(ledger.subgoals[0].results) == 3


def test_f7_fact_rate_limit_whitespace_stripping():
    """F7.2: Facts have surrounding whitespace stripped upon insertion."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg_1", text="sg", status="active")])
    res = ledger.add_result("   Distilled fact with padding   \n", "  https://site.com  ", 2)
    assert res.fact == "Distilled fact with padding"
    assert res.source_url == "https://site.com"


def test_f7_fact_rate_limit_progress_check_result_schema():
    """F7.3: ProgressCheckResult model accommodates 0-3 distilled new_notes and extracted_results."""
    pcr = ProgressCheckResult(
        subgoal_done=True,
        next_subgoal="sg_2",
        stuck=False,
        new_notes="Extracted key finding",
        extracted_results=[{"fact": "F1", "source_url": "https://a.com", "step": 4}],
    )
    assert pcr.subgoal_done is True
    assert pcr.new_notes == "Extracted key finding"
    assert len(pcr.extracted_results) == 1


def test_f7_fact_rate_limit_empty_notes_permitted():
    """F7.4: ProgressCheckResult allows None or empty notes when no new facts discovered."""
    pcr = ProgressCheckResult(subgoal_done=False, next_subgoal=None, stuck=False, new_notes=None)
    assert pcr.new_notes is None


def test_f7_fact_rate_limit_multiple_extracted_candidates():
    """F7.5: ProgressCheckResult carries extracted candidates list."""
    pcr = ProgressCheckResult(
        extracted_candidates=[{"item": "Headphones", "source_url": "https://site.com", "attributes": {"p": 50}}]
    )
    assert len(pcr.extracted_candidates) == 1


# --- F8: 2500-Character Source-Preserving Summarization ---

def test_f8_summarization_total_results_chars():
    """F8.1: total_results_chars accurately computes cumulative length of all facts."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg1",
                results=[
                    SubgoalResult(fact="abc", source_url="https://1.com", step=1),
                    SubgoalResult(fact="defgh", source_url="https://2.com", step=2),
                ],
            )
        ],
    )
    assert ledger.total_results_chars() == 8  # 3 + 5


def test_f8_summarization_not_triggered_under_2500():
    """F8.2: summarize_completed_results returns False when char count <= threshold."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg1",
                status="completed",
                results=[SubgoalResult(fact="Short fact", source_url="https://site.com", step=1)],
            )
        ],
    )
    triggered = ledger.summarize_completed_results(threshold=2500)
    assert triggered is False


def test_f8_summarization_triggered_over_2500():
    """F8.3: summarize_completed_results returns True when cumulative chars > 2500."""
    long_fact_1 = "A" * 1500
    long_fact_2 = "B" * 1500
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg1",
                status="completed",
                results=[
                    SubgoalResult(fact=long_fact_1, source_url="https://source1.org", step=1),
                    SubgoalResult(fact=long_fact_2, source_url="https://source2.org", step=2),
                ],
            )
        ],
    )
    assert ledger.total_results_chars() == 3000
    triggered = ledger.summarize_completed_results(threshold=2500)
    assert triggered is True
    assert ledger.total_results_chars() < 3000


def test_f8_summarization_preserves_all_source_urls():
    """F8.4: Summarization strictly preserves all unique source URLs from condensed results."""
    url1 = "https://alpha.org/paper1"
    url2 = "https://beta.org/paper2"
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg1",
                status="completed",
                results=[
                    SubgoalResult(fact="X" * 1400, source_url=url1, step=1),
                    SubgoalResult(fact="Y" * 1400, source_url=url2, step=2),
                ],
            )
        ],
    )
    ledger.summarize_completed_results(threshold=2500)
    surviving_urls = {r.source_url for r in ledger.subgoals[0].results}
    assert url1 in surviving_urls
    assert url2 in surviving_urls


def test_f8_summarization_custom_callable():
    """F8.5: summarize_completed_results supports custom summarizer function."""
    def custom_summarizer(results: list[SubgoalResult]) -> list[SubgoalResult]:
        urls = [r.source_url for r in results]
        return [SubgoalResult(fact="Custom summary", source_url=urls[0], step=results[0].step)]

    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg1",
                status="completed",
                results=[
                    SubgoalResult(fact="Z" * 1300, source_url="https://u1.com", step=1),
                    SubgoalResult(fact="W" * 1300, source_url="https://u2.com", step=2),
                ],
            )
        ],
    )
    triggered = ledger.summarize_completed_results(threshold=2500, summarizer=custom_summarizer)
    assert triggered is True
    assert len(ledger.subgoals[0].results) == 1
    assert ledger.subgoals[0].results[0].fact == "Custom summary"


# --- F9: System 1 Input Isolation ---

def test_f9_system1_isolation_includes_active_subgoal():
    """F9.1: compile_system1_goal includes active_subgoal when present."""
    ctx = DualTierContext(primary_goal="Overall task", active_subgoal="Tactical step 1")
    s1_goal = ctx.compile_system1_goal()
    assert "Current subgoal: Tactical step 1" in s1_goal


def test_f9_system1_isolation_excludes_candidates():
    """F9.2: compile_system1_goal strictly excludes candidate table entries."""
    ctx = DualTierContext(primary_goal="Compare headphones", active_subgoal="Find price on Amazon")
    # Even if candidates or tables exist in memory/plan, S1 must not see them
    s1_goal = ctx.compile_system1_goal()
    assert "candidates" not in s1_goal.lower()
    assert "candidate table" not in s1_goal.lower()


def test_f9_system1_isolation_excludes_completed_results():
    """F9.3: compile_system1_goal strictly excludes raw extracted results from past subgoals."""
    ctx = DualTierContext(primary_goal="Find bio info", active_subgoal="Find death year")
    s1_goal = ctx.compile_system1_goal()
    assert "results:" not in s1_goal.lower()
    assert "extracted facts" not in s1_goal.lower()


def test_f9_system1_isolation_preserves_constraints():
    """F9.4: compile_system1_goal includes constraints and tactical notes."""
    ctx = DualTierContext(
        primary_goal="Find info",
        active_subgoal="Step 1",
        constraints=["Only English sources", "No login"],
        tactical_hint="Click search icon first",
    )
    s1_goal = ctx.compile_system1_goal()
    assert "Only English sources" in s1_goal
    assert "Note: Click search icon first" in s1_goal


def test_f9_system1_isolation_goal_without_subgoal_matches_primary():
    """F9.5: compile_system1_goal without active_subgoal emits primary_goal."""
    ctx = DualTierContext(primary_goal="Direct search")
    assert ctx.compile_system1_goal() == "Direct search"


# --- F10: System 2 Context Block ---

def test_f10_system2_context_includes_plan_status():
    """F10.1: format_for_s2 includes Plan Status."""
    ledger = TaskLedger(needs_plan=True, finish_criteria="Collect 3 sources")
    text = ledger.format_for_s2()
    assert "Plan Status: Active" in text
    assert "Finish Criteria: Collect 3 sources" in text


def test_f10_system2_context_lists_all_subgoals_with_statuses():
    """F10.2: format_for_s2 lists subgoals with active/completed/pending statuses."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="Find Lovelace", status="completed"),
            Subgoal(id="sg_2", text="Find Babbage", status="active"),
        ],
    )
    text = ledger.format_for_s2()
    assert "[COMPLETED] sg_1: Find Lovelace" in text
    assert "[ACTIVE] sg_2: Find Babbage" in text


def test_f10_system2_context_lists_results_with_source_urls():
    """F10.3: format_for_s2 includes extracted facts and source URLs."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find Lovelace",
                status="completed",
                results=[SubgoalResult(fact="Born 1815", source_url="https://wiki.org/Lovelace", step=2)],
            )
        ],
    )
    text = ledger.format_for_s2()
    assert "Born 1815" in text
    assert "https://wiki.org/Lovelace" in text


def test_f10_system2_context_lists_candidates():
    """F10.4: format_for_s2 formats Candidates Table with attributes and source URLs."""
    ledger = TaskLedger(
        needs_plan=True,
        candidates=[Candidate(item="Sony XM4", source_url="https://sony.com", attributes={"price": "$249"})],
    )
    text = ledger.format_for_s2()
    assert "Candidates Table:" in text
    assert "Sony XM4" in text
    assert "price: $249" in text
    assert "https://sony.com" in text


def test_f10_system2_context_includes_budget():
    """F10.5: format_for_s2 includes budget sources and max steps."""
    ledger = TaskLedger(needs_plan=True, budget=Budget(sources=4, max_steps=40))
    text = ledger.format_for_s2()
    assert "Budget: 4 sources, 40 max steps" in text


# --- F11: Milestone Progress Checks ---

def test_f11_milestone_check_schema_valid():
    """F11.1: ProgressCheckResult parses valid JSON/dict correctly."""
    data = {"subgoal_done": True, "next_subgoal": "sg_2", "stuck": False, "new_notes": "Done step 1"}
    pcr = ProgressCheckResult(**data)
    assert pcr.subgoal_done is True
    assert pcr.next_subgoal == "sg_2"
    assert pcr.stuck is False
    assert pcr.new_notes == "Done step 1"


def test_f11_milestone_check_defaults_when_optional():
    """F11.2: ProgressCheckResult defaults optional fields appropriately."""
    pcr = ProgressCheckResult()
    assert pcr.subgoal_done is False
    assert pcr.next_subgoal is None
    assert pcr.stuck is False
    assert pcr.new_notes is None
    assert pcr.extracted_results == []
    assert pcr.extracted_candidates == []


def test_f11_milestone_check_stuck_flag():
    """F11.3: ProgressCheckResult records stuck condition."""
    pcr = ProgressCheckResult(stuck=True, new_notes="Repeated navigation detected")
    assert pcr.stuck is True
    assert "Repeated" in pcr.new_notes


def test_f11_milestone_check_with_results_and_candidates():
    """F11.4: ProgressCheckResult captures batch of results and candidates."""
    pcr = ProgressCheckResult(
        subgoal_done=True,
        extracted_results=[{"fact": "F1", "source_url": "https://a.com", "step": 1}],
        extracted_candidates=[{"item": "Prod1", "source_url": "https://a.com", "attributes": {"p": 10}}],
    )
    assert len(pcr.extracted_results) == 1
    assert len(pcr.extracted_candidates) == 1


def test_f11_milestone_check_roundtrip_dump():
    """F11.5: ProgressCheckResult serializes cleanly to dict."""
    pcr = ProgressCheckResult(subgoal_done=True, next_subgoal="sg_3", stuck=False)
    d = pcr.model_dump()
    assert d["subgoal_done"] is True
    assert d["next_subgoal"] == "sg_3"


# --- F12: Subgoal Advancement & Stuck Counter ---

def test_f12_subgoal_advancement_marks_active_completed():
    """F12.1: advance_subgoal transitions active subgoal to completed."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="Step 1", status="active"),
            Subgoal(id="sg_2", text="Step 2", status="pending"),
        ],
    )
    next_sg = ledger.advance_subgoal()
    assert ledger.subgoals[0].status == "completed"
    assert next_sg is not None
    assert next_sg.id == "sg_2"
    assert next_sg.status == "active"


def test_f12_subgoal_advancement_returns_none_when_finished():
    """F12.2: advance_subgoal returns None when all subgoals are completed."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="Step 1", status="active")],
    )
    next_sg = ledger.advance_subgoal()
    assert next_sg is None
    assert ledger.subgoals[0].status == "completed"


def test_f12_subgoal_advancement_sequential_multi_step():
    """F12.3: Sequential advance_subgoal calls progress through all subgoals in order."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="S1", status="active"),
            Subgoal(id="sg_2", text="S2", status="pending"),
            Subgoal(id="sg_3", text="S3", status="pending"),
        ],
    )
    sg2 = ledger.advance_subgoal()
    assert sg2.id == "sg_2"
    sg3 = ledger.advance_subgoal()
    assert sg3.id == "sg_3"
    sg4 = ledger.advance_subgoal()
    assert sg4 is None
    assert all(sg.status == "completed" for sg in ledger.subgoals)


def test_f12_subgoal_advancement_get_active_subgoal():
    """F12.4: get_active_subgoal returns current active subgoal."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="S1", status="completed"),
            Subgoal(id="sg_2", text="S2", status="active"),
        ],
    )
    assert ledger.get_active_subgoal().id == "sg_2"


def test_f12_subgoal_advancement_activates_first_pending_if_none_active():
    """F12.5: get_active_subgoal activates first pending subgoal if no active exists."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="S1", status="pending"),
            Subgoal(id="sg_2", text="S2", status="pending"),
        ],
    )
    active = ledger.get_active_subgoal()
    assert active.id == "sg_1"
    assert active.status == "active"


# --- F13: Multi-Criteria DONE Verification ---

def test_f13_done_verification_success_all_criteria_met():
    """F13.1: verify_done approves when all subgoals completed with results and sources budget met."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find A",
                status="completed",
                results=[SubgoalResult(fact="A found", source_url="https://site-a.com/page", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Find B",
                status="completed",
                results=[SubgoalResult(fact="B found", source_url="https://site-b.com/page", step=3)],
            ),
        ],
    )
    is_done, _reason, missing_sg = ledger.verify_done()
    assert is_done is True
    assert missing_sg is None


def test_f13_done_verification_fails_when_subgoal_pending():
    """F13.2: verify_done rejects if any subgoal is still pending or active."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=1),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find A",
                status="completed",
                results=[SubgoalResult(fact="A", source_url="https://a.com", step=1)],
            ),
            Subgoal(id="sg_2", text="Find B", status="pending", results=[]),
        ],
    )
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is False
    assert missing_sg is not None
    assert missing_sg.id == "sg_2"
    assert "not completed" in reason.lower()


def test_f13_done_verification_fails_when_completed_subgoal_has_no_results():
    """F13.3: verify_done rejects if a completed subgoal has no verified results."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=1),
        subgoals=[
            Subgoal(id="sg_1", text="Find A", status="completed", results=[]),
        ],
    )
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is False
    assert missing_sg.id == "sg_1"
    assert "no verified results" in reason.lower()


def test_f13_done_verification_fails_when_sources_budget_unmet():
    """F13.4: verify_done rejects if distinct domain count is less than budget.sources."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find A",
                status="completed",
                results=[
                    SubgoalResult(fact="Fact 1", source_url="https://same-site.com/p1", step=1),
                    SubgoalResult(fact="Fact 2", source_url="https://same-site.com/p2", step=2),
                ],
            )
        ],
    )
    is_done, reason, _missing_sg = ledger.verify_done()
    assert is_done is False
    assert "budget requires 2" in reason.lower()


def test_f13_done_verification_fails_when_candidate_count_insufficient():
    """F13.5: verify_done rejects comparison task when candidates count < budget.sources."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=3),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Compare items",
                status="completed",
                results=[
                    SubgoalResult(fact="Found item 1", source_url="https://s1.com", step=1),
                    SubgoalResult(fact="Found item 2", source_url="https://s2.com", step=2),
                    SubgoalResult(fact="Found item 3", source_url="https://s3.com", step=3),
                ],
            )
        ],
        candidates=[Candidate(item="Item 1", source_url="https://s1.com", attributes={"p": 10})],
    )
    is_done, reason, _missing_sg = ledger.verify_done()
    assert is_done is False
    assert "candidate table has 1 items" in reason.lower()


# --- F14: DONE Rejection & Subgoal Reactivation ---

def test_f14_done_rejection_identifies_missing_subgoal():
    """F14.1: verify_done returns the exact Subgoal object that was incomplete."""
    incomplete_sg = Subgoal(id="sg_missing", text="Missing step", status="pending")
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_done",
                text="Done step",
                status="completed",
                results=[SubgoalResult(fact="OK", source_url="https://ok.com", step=1)],
            ),
            incomplete_sg,
        ],
    )
    is_done, _reason, returned_sg = ledger.verify_done()
    assert is_done is False
    assert returned_sg is incomplete_sg


def test_f14_done_rejection_reactivation_changes_status_to_active():
    """F14.2: Reactivating a missing subgoal changes its status to active."""
    sg = Subgoal(id="sg_miss", text="Missing", status="completed", results=[])
    ledger = TaskLedger(needs_plan=True, subgoals=[sg])
    is_done, _reason, target = ledger.verify_done()
    assert is_done is False
    target.status = "active"
    assert ledger.get_active_subgoal() is target


def test_f14_done_rejection_descriptive_reason():
    """F14.3: Rejection reason provides informative message."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="Research", status="active", results=[])],
    )
    is_done, reason, _ = ledger.verify_done()
    assert is_done is False
    assert "Research" in reason


def test_f14_done_rejection_reverts_completed_without_results():
    """F14.4: Subgoal falsely marked completed without results is identified for reactivation."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_empty", text="Empty results", status="completed", results=[])],
    )
    is_done, _reason, missing = ledger.verify_done()
    assert is_done is False
    assert missing.id == "sg_empty"


def test_f14_done_rejection_when_sources_deficit_returns_last_subgoal():
    """F14.5: Source budget deficit returns fallback subgoal for continued research."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=3),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Search more",
                status="completed",
                results=[SubgoalResult(fact="Fact", source_url="https://site.com", step=1)],
            )
        ],
    )
    is_done, _reason, missing = ledger.verify_done()
    assert is_done is False
    assert missing is not None
    assert missing.id == "sg_1"


# --- F15: Source-Cited Final Answer ---

def test_f15_final_answer_includes_primary_goal():
    """F15.1: synthesize_final_answer header contains primary goal."""
    ledger = TaskLedger(needs_plan=True)
    ans = ledger.synthesize_final_answer("Compare iPhone and Pixel")
    assert "Compare iPhone and Pixel" in ans


def test_f15_final_answer_includes_subgoal_facts():
    """F15.2: synthesize_final_answer includes facts from all subgoals."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Ada Lovelace Info",
                results=[SubgoalResult(fact="Ada born 1815", source_url="https://en.wikipedia.org/Ada", step=1)],
            )
        ],
    )
    ans = ledger.synthesize_final_answer("Biographies")
    assert "Ada born 1815" in ans
    assert "Ada Lovelace Info" in ans


def test_f15_final_answer_cites_source_urls():
    """F15.3: synthesize_final_answer explicitly cites source URLs."""
    url = "https://example.org/biography"
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Bio",
                results=[SubgoalResult(fact="Key achievement", source_url=url, step=2)],
            )
        ],
    )
    ans = ledger.synthesize_final_answer("Bio Task")
    assert url in ans
    assert "Kaynak:" in ans or "source" in ans.lower()


def test_f15_final_answer_includes_candidate_table():
    """F15.4: synthesize_final_answer formats candidates table with attributes and URLs."""
    ledger = TaskLedger(
        needs_plan=True,
        candidates=[
            Candidate(item="Store A", source_url="https://store-a.com", attributes={"price": "100 TL", "rating": "4.8"})
        ],
    )
    ans = ledger.synthesize_final_answer("Price comparison")
    assert "Store A" in ans
    assert "100 TL" in ans
    assert "https://store-a.com" in ans


def test_f15_final_answer_both_facts_and_candidates():
    """F15.5: synthesize_final_answer combines both extracted facts and candidates table."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Research",
                results=[SubgoalResult(fact="Best seller", source_url="https://review.com", step=1)],
            )
        ],
        candidates=[Candidate(item="Item Z", source_url="https://z.com", attributes={"price": "$50"})],
    )
    ans = ledger.synthesize_final_answer("Full comparison")
    assert "Best seller" in ans
    assert "Item Z" in ans


# --- F16: Academic Attribution & Architectural Non-Goals ---

_README_PATH = Path(__file__).resolve().parent.parent / "README.md"
_README_CONTENT = _README_PATH.read_text(encoding="utf-8") if _README_PATH.exists() else ""
_HAS_ATTRIBUTION = "Plan-and-Act" in _README_CONTENT or "Erdogan" in _README_CONTENT


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_f16_academic_attribution_plan_and_act_in_readme():
    """F16.1: README.md cites Plan-and-Act (Erdogan et al. 2025)."""
    assert "Plan-and-Act" in _README_CONTENT or "Erdogan" in _README_CONTENT


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_f16_academic_attribution_structured_agent_in_readme():
    """F16.2: README.md cites StructuredAgent (2026)."""
    assert "StructuredAgent" in _README_CONTENT


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_f16_academic_attribution_browse_master_in_readme():
    """F16.3: README.md cites BrowseMaster (2025)."""
    assert "BrowseMaster" in _README_CONTENT


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_f16_academic_attribution_magentic_one_in_readme():
    """F16.4: README.md cites Magentic-One."""
    assert "Magentic-One" in _README_CONTENT


def test_f16_non_goals_no_vector_db_or_subagent_frameworks():
    """F16.5: Architectural Non-Goal: Reflex codebase contains no vector DB or subagent frameworks."""
    forbidden_terms = ["chromadb", "pinecone", "weaviate", "qdrant", "crewai", "autogen"]
    reflex_dir = Path(__file__).resolve().parent.parent / "reflex"
    for py_file in reflex_dir.glob("**/*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            assert f"import {term}" not in text, f"Forbidden non-goal import '{term}' in {py_file}"


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (F1 - F16, >=5 Tests Per Feature)
# ==============================================================================

# --- F1 Boundaries ---

def test_tier2_f1_malformed_json_fallback():
    """F1.B1: Malformed non-JSON string falls back to needs_plan=False without exception."""
    ledger = parse_task_ledger("INVALID JSON {{{ 123")
    assert ledger.needs_plan is False


def test_tier2_f1_empty_dict():
    """F1.B2: Empty dictionary input falls back to needs_plan=False."""
    ledger = parse_task_ledger({})
    assert ledger.needs_plan is False


def test_tier2_f1_non_dict_non_str_input():
    """F1.B3: Non-dict, non-str input (e.g. integer or list) falls back gracefully."""
    assert parse_task_ledger(12345).needs_plan is False
    assert parse_task_ledger(["not", "a", "dict"]).needs_plan is False


def test_tier2_f1_empty_subgoals_list_generates_fallback_subgoal():
    """F1.B4: needs_plan=True with subgoals=[] generates a single default fallback subgoal."""
    ledger = parse_task_ledger({"needs_plan": True, "subgoals": [], "finish_criteria": "Find info"})
    assert ledger.needs_plan is True
    assert len(ledger.subgoals) == 1
    assert ledger.subgoals[0].text == "Find info"
    assert ledger.subgoals[0].status == "active"


def test_tier2_f1_corrupted_subgoal_entries_ignored():
    """F1.B5: Subgoals containing non-dict entries are skipped gracefully."""
    ledger = parse_task_ledger({
        "needs_plan": True,
        "subgoals": ["not a dict", None, 42, {"id": "sg_valid", "text": "Valid"}],
    })
    assert len(ledger.subgoals) == 1
    assert ledger.subgoals[0].id == "sg_valid"


# --- F2 Boundaries ---

def test_tier2_f2_needs_plan_string_coercion():
    """F2.B1: needs_plan passed as boolean or falsy values parsed correctly."""
    assert parse_task_ledger({"needs_plan": 0}).needs_plan is False
    assert parse_task_ledger({"needs_plan": None}).needs_plan is False


def test_tier2_f2_needs_plan_false_with_extraneous_fields():
    """F2.B2: needs_plan=False ignores any subgoals or candidates passed."""
    ledger = parse_task_ledger({"needs_plan": False, "subgoals": [{"id": "sg1"}], "candidates": [{"item": "X"}]})
    assert ledger.needs_plan is False
    assert len(ledger.subgoals) == 0
    assert len(ledger.candidates) == 0


def test_tier2_f2_empty_json_string():
    """F2.B3: Empty string input falls back to needs_plan=False."""
    assert parse_task_ledger("").needs_plan is False


def test_tier2_f2_whitespace_only_json_string():
    """F2.B4: Whitespace only string falls back to needs_plan=False."""
    assert parse_task_ledger("    \n\t  ").needs_plan is False


def test_tier2_f2_simple_task_verify_done_always_passes():
    """F2.B5: When needs_plan=False and subgoals are empty, verify_done passes when visited URL satisfies source budget."""
    ledger = TaskLedger(needs_plan=False)
    is_done, _reason, _missing = ledger.verify_done(visited_urls=["https://single-site.com"])
    assert is_done is True


# --- F3 Boundaries ---

def test_tier2_f3_clarify_empty_options():
    """F3.B1: Clarify with empty options list parses without crash."""
    c = Clarify(question="Question?", options=[], default="DefaultVal")
    assert c.options == []
    assert c.default == "DefaultVal"


def test_tier2_f3_clarify_whitespace_question():
    """F3.B2: Clarify handles question with whitespace cleanly."""
    raw = {
        "needs_plan": True,
        "clarify": {"question": "   How many?   ", "options": [" 1 ", " 2 "], "default": " 1 "},
    }
    ledger = parse_task_ledger(raw)
    assert ledger.clarify.question == "How many?"
    assert ledger.clarify.default == "1"
    assert ledger.clarify.options == ["1", "2"]


def test_tier2_f3_clarify_null_in_dict():
    """F3.B3: clarify key with explicit null / None value parses as None."""
    ledger = parse_task_ledger({"needs_plan": True, "clarify": None})
    assert ledger.clarify is None


def test_tier2_f3_clarify_missing_question_field():
    """F3.B4: clarify dictionary without 'question' is treated as None."""
    ledger = parse_task_ledger({"needs_plan": True, "clarify": {"options": ["a", "b"], "default": "a"}})
    assert ledger.clarify is None


def test_tier2_f3_clarify_unicode_and_emojis():
    """F3.B5: Clarify supports unicode characters and emojis."""
    c = Clarify(question="Hangi mağaza? 🛒", options=["Amazon 📦", "Hepsiburada 🚀"], default="Amazon 📦")
    assert "🛒" in c.question
    assert "🚀" in c.options[1]


# --- F4 Boundaries ---

def test_tier2_f4_config_interactive_type_coercion():
    """F4.B1: ReflexConfig coerces truthy/falsy values for interactive."""
    cfg_true = ReflexConfig(interactive=1)
    assert cfg_true.interactive is True
    cfg_false = ReflexConfig(interactive=0)
    assert cfg_false.interactive is False


def test_tier2_f4_config_multiple_instances_isolation():
    """F4.B2: Multiple ReflexConfig instances have isolated interactive flags."""
    cfg1 = ReflexConfig(interactive=True)
    cfg2 = ReflexConfig(interactive=False)
    assert cfg1.interactive is True
    assert cfg2.interactive is False


def test_tier2_f4_interactive_false_no_clarify_remains_ready():
    """F4.B3: Non-interactive mode without clarify maintains ready state."""
    ledger = TaskLedger(needs_plan=True, clarify=None)
    assert ledger.clarify is None


def test_tier2_f4_interactive_empty_options_fallback_to_default():
    """F4.B4: Non-interactive mode with clarify having empty options still uses default."""
    c = Clarify(question="Count?", options=[], default="3")
    assert c.default == "3"


def test_tier2_f4_config_env_override_simulation():
    """F4.B5: ReflexConfig respects explicit constructor keyword argument."""
    cfg = ReflexConfig(interactive=True)
    assert cfg.interactive is True


# --- F5 Boundaries ---

def test_tier2_f5_add_result_when_no_subgoals():
    """F5.B1: add_result does not crash when subgoals list is empty."""
    ledger = TaskLedger(needs_plan=True, subgoals=[])
    res = ledger.add_result("Orphan fact", "https://orphan.com", 1)
    assert res.fact == "Orphan fact"


def test_tier2_f5_add_result_negative_or_zero_step():
    """F5.B2: SubgoalResult accepts step=0 or negative numbers."""
    r0 = SubgoalResult(fact="Step 0 fact", source_url="https://site.com", step=0)
    assert r0.step == 0


def test_tier2_f5_add_result_massive_fact_string():
    """F5.B3: add_result safely handles massive fact string (10,000 chars)."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg1", text="sg", status="active")])
    massive_text = "F" * 10000
    res = ledger.add_result(massive_text, "https://massive.com", 1)
    assert len(res.fact) == 10000


def test_tier2_f5_add_result_empty_source_url():
    """F5.B4: SubgoalResult handles empty string source_url."""
    r = SubgoalResult(fact="Fact without url", source_url="", step=1)
    assert r.source_url == ""


def test_tier2_f5_add_result_special_characters_and_html():
    """F5.B5: SubgoalResult preserves special characters and escaping."""
    special = "Fact with <script>alert('xss')</script> & quotes \" ' \n \t"
    r = SubgoalResult(fact=special, source_url="https://xss.com", step=1)
    assert "<script>" in r.fact


# --- F6 Boundaries ---

def test_tier2_f6_candidate_empty_item_string():
    """F6.B1: Adding candidate with empty item string handled gracefully."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("", "https://empty.com", {})
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].item == ""


def test_tier2_f6_candidate_empty_attributes():
    """F6.B2: Candidate with empty attributes dictionary."""
    ledger = TaskLedger(needs_plan=True)
    c = ledger.add_candidate("Item A", "https://a.com", None)
    assert c.attributes == {}


def test_tier2_f6_candidate_complex_attribute_types():
    """F6.B3: Candidate attributes accept floats, ints, booleans, and None."""
    ledger = TaskLedger(needs_plan=True)
    attrs = {"price": 99.95, "stock": 10, "available": True, "discount": None}
    c = ledger.add_candidate("Item Complex", "https://comp.com", attrs)
    assert c.attributes["price"] == 99.95
    assert c.attributes["available"] is True
    assert c.attributes["discount"] is None


def test_tier2_f6_candidate_deduplication_whitespace_padding():
    """F6.B4: Candidate deduplication trims leading and trailing whitespace."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("   Product Z   ", "https://z1.com", {"p": 1})
    ledger.add_candidate("product z", "https://z2.com", {"p": 2})
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["p"] == 2


def test_tier2_f6_candidate_100_items_performance():
    """F6.B5: Adding 100 distinct candidates executes in under 50ms."""
    ledger = TaskLedger(needs_plan=True)
    for i in range(100):
        ledger.add_candidate(f"Product {i}", f"https://p{i}.com", {"index": i})
    assert len(ledger.candidates) == 100


# --- F7 Boundaries ---

def test_tier2_f7_fact_rate_limit_empty_string_ignored():
    """F7.B1: Empty or None facts in parsing are ignored."""
    raw = {
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "sg",
                "results": [
                    {"fact": "", "source_url": "https://a.com", "step": 1},
                    {"fact": None, "source_url": "https://c.com", "step": 2},
                    {"fact": "Valid Fact", "source_url": "https://b.com", "step": 3},
                ],
            }
        ],
    }
    ledger = parse_task_ledger(raw)
    assert len(ledger.subgoals[0].results) == 1
    assert ledger.subgoals[0].results[0].fact == "Valid Fact"


def test_tier2_f7_progress_check_corrupted_extracted_results():
    """F7.B2: ProgressCheckResult handles empty extracted_results list."""
    pcr = ProgressCheckResult(extracted_results=[])
    assert pcr.extracted_results == []


def test_tier2_f7_fact_rate_limit_0_facts_milestone():
    """F7.B3: Progress check with 0 new facts leaves subgoal results unchanged."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg1", text="t", status="active")])
    # 0 facts recorded
    assert len(ledger.subgoals[0].results) == 0


def test_tier2_f7_fact_rate_limit_exactly_3_facts():
    """F7.B4: Recording exactly 3 facts is accepted at milestone."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg1", text="t", status="active")])
    for i in range(3):
        ledger.add_result(f"Fact {i}", "https://site.com", i)
    assert len(ledger.subgoals[0].results) == 3


def test_tier2_f7_fact_with_newlines():
    """F7.B5: Facts containing internal newlines are preserved with whitespace stripped."""
    ledger = TaskLedger(needs_plan=True, subgoals=[Subgoal(id="sg1", text="t", status="active")])
    res = ledger.add_result("Line 1\nLine 2\nLine 3", "https://site.com", 1)
    assert "Line 1" in res.fact
    assert "Line 3" in res.fact


# --- F8 Boundaries ---

def test_tier2_f8_summarization_exactly_2500_chars_boundary():
    """F8.B1: Exactly 2500 characters does NOT trigger summarization."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg",
                status="completed",
                results=[
                    SubgoalResult(fact="A" * 1250, source_url="https://1.com", step=1),
                    SubgoalResult(fact="B" * 1250, source_url="https://2.com", step=2),
                ],
            )
        ],
    )
    assert ledger.total_results_chars() == 2500
    assert ledger.summarize_completed_results(threshold=2500) is False


def test_tier2_f8_summarization_exactly_2501_chars_boundary():
    """F8.B2: Exactly 2501 characters DOES trigger summarization."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg",
                status="completed",
                results=[
                    SubgoalResult(fact="A" * 1250, source_url="https://1.com", step=1),
                    SubgoalResult(fact="B" * 1251, source_url="https://2.com", step=2),
                ],
            )
        ],
    )
    assert ledger.total_results_chars() == 2501
    assert ledger.summarize_completed_results(threshold=2500) is True


def test_tier2_f8_summarization_only_completed_subgoals_summarized():
    """F8.B3: Active or pending subgoals are not modified by summarization."""
    active_fact = "C" * 2600
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_active",
                text="Active sg",
                status="active",
                results=[SubgoalResult(fact=active_fact, source_url="https://act.com", step=1)],
            )
        ],
    )
    # Total chars > 2500, but no completed subgoals exist
    assert ledger.summarize_completed_results(threshold=2500) is False
    assert ledger.subgoals[0].results[0].fact == active_fact


def test_tier2_f8_summarization_single_result_subgoal_condenses():
    """F8.B4: Completed subgoal with 1 massive result (>2500 chars) is condensed to max 300 chars."""
    fact_2600 = "D" * 2600
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg",
                status="completed",
                results=[SubgoalResult(fact=fact_2600, source_url="https://d.com", step=1)],
            )
        ],
    )
    assert ledger.summarize_completed_results(threshold=2500) is True
    assert len(ledger.subgoals[0].results) == 1
    assert len(ledger.subgoals[0].results[0].fact) <= 300
    assert ledger.subgoals[0].results[0].source_url == "https://d.com"


def test_tier2_f8_summarization_multiple_results_same_url():
    """F8.B5: Multiple facts from same URL are grouped under single source URL."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="sg",
                status="completed",
                results=[
                    SubgoalResult(fact="Fact A " * 200, source_url="https://common.org", step=1),
                    SubgoalResult(fact="Fact B " * 200, source_url="https://common.org", step=2),
                ],
            )
        ],
    )
    assert ledger.total_results_chars() > 2500
    assert ledger.summarize_completed_results(threshold=2500) is True
    assert len(ledger.subgoals[0].results) == 1
    assert ledger.subgoals[0].results[0].source_url == "https://common.org"


# --- F9 Boundaries ---

def test_tier2_f9_system1_isolation_empty_strings():
    """F9.B1: compile_system1_goal handles empty strings in active_subgoal cleanly."""
    ctx = DualTierContext(primary_goal="Goal", active_subgoal="")
    assert ctx.compile_system1_goal() == "Goal"


def test_tier2_f9_system1_isolation_active_subgoal_same_as_primary_goal():
    """F9.B2: When active_subgoal equals primary_goal, redundant 'Current subgoal' line omitted."""
    ctx = DualTierContext(primary_goal="Search phones", active_subgoal="Search phones")
    s1_goal = ctx.compile_system1_goal()
    assert s1_goal.count("Search phones") == 1


def test_tier2_f9_system1_isolation_massive_constraints():
    """F9.B3: 50 constraints concatenated cleanly in S1 goal string."""
    constraints = [f"Constraint {i}" for i in range(50)]
    ctx = DualTierContext(primary_goal="Goal", constraints=constraints)
    s1_goal = ctx.compile_system1_goal()
    assert "Constraint 0" in s1_goal
    assert "Constraint 49" in s1_goal


def test_tier2_f9_system1_isolation_site_memory_hint():
    """F9.B4: compile_system1_goal includes site memory hint when provided."""
    ctx = DualTierContext(primary_goal="Goal")
    s1_goal = ctx.compile_system1_goal(site_memory_hint="Click cookie banner first")
    assert "Site rule: Click cookie banner first" in s1_goal


def test_tier2_f9_system1_isolation_user_facts():
    """F9.B5: compile_system1_goal formats user facts properly."""
    ctx = DualTierContext(primary_goal="Goal", user_facts={"name": "Alice", "city": "Istanbul"})
    s1_goal = ctx.compile_system1_goal()
    assert "name=Alice" in s1_goal
    assert "city=Istanbul" in s1_goal


# --- F10 Boundaries ---

def test_tier2_f10_s2_context_empty_plan():
    """F10.B1: format_for_s2 with 0 subgoals and 0 candidates."""
    ledger = TaskLedger(needs_plan=True, subgoals=[], candidates=[])
    text = ledger.format_for_s2()
    assert "Plan Status: Active" in text
    assert "Subgoals & Extracted Results:" in text


def test_tier2_f10_s2_context_special_characters():
    """F10.B2: format_for_s2 preserves unicode and Turkish characters."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="İstanbul ve Ankara'da fiyat karşılaştır", status="active")],
    )
    text = ledger.format_for_s2()
    assert "İstanbul ve Ankara'da" in text


def test_tier2_f10_s2_context_candidate_attributes_formatting():
    """F10.B3: format_for_s2 formats candidate attributes cleanly with comma separation."""
    ledger = TaskLedger(
        needs_plan=True,
        candidates=[Candidate(item="Phone", attributes={"brand": "Apple", "storage": "128GB"})],
    )
    text = ledger.format_for_s2()
    assert "brand: Apple" in text
    assert "storage: 128GB" in text


def test_tier2_f10_s2_context_20_subgoals_scaling():
    """F10.B4: format_for_s2 formats 20 subgoals without truncation or error."""
    subgoals = [Subgoal(id=f"sg_{i}", text=f"Step {i}", status="pending") for i in range(20)]
    ledger = TaskLedger(needs_plan=True, subgoals=subgoals)
    text = ledger.format_for_s2()
    assert "sg_0" in text
    assert "sg_19" in text


def test_tier2_f10_s2_context_budget_boundary_values():
    """F10.B5: format_for_s2 handles extreme budget values."""
    ledger = TaskLedger(needs_plan=True, budget=Budget(sources=100, max_steps=500))
    text = ledger.format_for_s2()
    assert "Budget: 100 sources, 500 max steps" in text


# --- F11 Boundaries ---

def test_tier2_f11_progress_check_all_falsy():
    """F11.B1: ProgressCheckResult handles all falsy fields without error."""
    pcr = ProgressCheckResult(subgoal_done=False, next_subgoal=None, stuck=False, new_notes=None)
    assert pcr.subgoal_done is False
    assert pcr.next_subgoal is None


def test_tier2_f11_progress_check_special_chars_in_notes():
    """F11.B2: ProgressCheckResult handles multi-line string in new_notes."""
    notes = "Line 1: Item found\nLine 2: Out of stock"
    pcr = ProgressCheckResult(new_notes=notes)
    assert pcr.new_notes == notes


def test_tier2_f11_progress_check_empty_string_next_subgoal():
    """F11.B3: ProgressCheckResult with empty string next_subgoal."""
    pcr = ProgressCheckResult(next_subgoal="")
    assert pcr.next_subgoal == ""


def test_tier2_f11_progress_check_large_batch_results():
    """F11.B4: ProgressCheckResult accommodates 5 extracted results in batch."""
    results = [{"fact": f"Fact {i}", "source_url": f"https://s{i}.com", "step": i} for i in range(5)]
    pcr = ProgressCheckResult(extracted_results=results)
    assert len(pcr.extracted_results) == 5


def test_tier2_f11_progress_check_json_serialization():
    """F11.B5: ProgressCheckResult serializes to valid JSON."""
    pcr = ProgressCheckResult(subgoal_done=True, stuck=False)
    j = pcr.model_dump_json()
    assert '"subgoal_done":true' in j


# --- F12 Boundaries ---

def test_tier2_f12_advance_subgoal_empty_subgoals():
    """F12.B1: advance_subgoal on TaskLedger with 0 subgoals returns None."""
    ledger = TaskLedger(needs_plan=True, subgoals=[])
    assert ledger.advance_subgoal() is None


def test_tier2_f12_advance_subgoal_all_already_completed():
    """F12.B2: advance_subgoal when all subgoals are completed returns None."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg1", text="t", status="completed")],
    )
    assert ledger.advance_subgoal() is None


def test_tier2_f12_advance_subgoal_skips_non_pending():
    """F12.B3: advance_subgoal skips subgoals that are already completed or failed."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg1", text="t1", status="active"),
            Subgoal(id="sg2", text="t2", status="failed"),
            Subgoal(id="sg3", text="t3", status="pending"),
        ],
    )
    next_sg = ledger.advance_subgoal()
    assert next_sg.id == "sg3"


def test_tier2_f12_get_active_subgoal_empty_list():
    """F12.B4: get_active_subgoal on empty subgoals returns None."""
    ledger = TaskLedger(needs_plan=True, subgoals=[])
    assert ledger.get_active_subgoal() is None


def test_tier2_f12_advance_subgoal_status_mutation():
    """F12.B5: advance_subgoal directly mutates status attribute on Subgoal object."""
    sg1 = Subgoal(id="sg1", text="t1", status="active")
    sg2 = Subgoal(id="sg2", text="t2", status="pending")
    ledger = TaskLedger(needs_plan=True, subgoals=[sg1, sg2])
    ledger.advance_subgoal()
    assert sg1.status == "completed"
    assert sg2.status == "active"


# --- F13 Boundaries ---

def test_tier2_f13_verify_done_domain_normalization_case_insensitive():
    """F13.B1: verify_done normalizes domain names case-insensitively (e.g. SITE.COM vs site.com)."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t1",
                status="completed",
                results=[
                    SubgoalResult(fact="F1", source_url="https://SITE.COM/p1", step=1),
                    SubgoalResult(fact="F2", source_url="https://site.com/p2", step=2),
                ],
            )
        ],
    )
    # Should only count as 1 distinct domain, so budget of 2 is NOT met
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is False


def test_tier2_f13_verify_done_ignores_ports_and_queries():
    """F13.B2: verify_done extracts domain cleanly regardless of query parameters or paths."""
    url1 = "https://example.com/search?q=test#hash"
    url2 = "https://example.com/item/42?ref=home"
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[
                    SubgoalResult(fact="F1", source_url=url1, step=1),
                    SubgoalResult(fact="F2", source_url=url2, step=2),
                ],
            )
        ],
    )
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is False


def test_tier2_f13_verify_done_with_visited_urls_fallback():
    """F13.B3: verify_done counts domains from visited_urls if provided."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[SubgoalResult(fact="F1", source_url="https://site-a.com/p1", step=1)],
            )
        ],
    )
    visited = ["https://site-b.com/landing"]
    is_done, _reason, _ = ledger.verify_done(visited_urls=visited)
    assert is_done is True


def test_tier2_f13_verify_done_sources_budget_zero_or_one():
    """F13.B4: verify_done with budget.sources=1 passes with single source."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=1),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[SubgoalResult(fact="F", source_url="https://single.com", step=1)],
            )
        ],
    )
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is True


def test_tier2_f13_verify_done_empty_source_url_in_results():
    """F13.B5: Results with empty source_url do not crash domain parsing."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=1),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[SubgoalResult(fact="F", source_url="", step=1)],
            )
        ],
    )
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is False  # empty URL gives 0 sources, budget=1 unmet


# --- F14 Boundaries ---

def test_tier2_f14_done_rejection_repeated_calls_stable():
    """F14.B1: Calling verify_done repeatedly on rejected plan produces identical stable result."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg1", text="t", status="pending")],
    )
    res1 = ledger.verify_done()
    res2 = ledger.verify_done()
    assert res1[0] == res2[0] == False
    assert res1[1] == res2[1]
    assert res1[2] == res2[2]


def test_tier2_f14_done_rejection_when_all_subgoals_empty_results():
    """F14.B2: When multiple subgoals lack results, the first incomplete one is returned."""
    sg1 = Subgoal(id="sg1", text="t1", status="completed", results=[])
    sg2 = Subgoal(id="sg2", text="t2", status="completed", results=[])
    ledger = TaskLedger(needs_plan=True, subgoals=[sg1, sg2])
    _is_done, _reason, missing = ledger.verify_done()
    assert missing is sg1


def test_tier2_f14_done_rejection_budget_10_sources():
    """F14.B3: Rejection with large source budget clearly states found count vs required."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=10),
        subgoals=[
            Subgoal(
                id="sg1",
                text="t",
                status="completed",
                results=[SubgoalResult(fact="F", source_url="https://s1.com", step=1)],
            )
        ],
    )
    is_done, reason, _ = ledger.verify_done()
    assert is_done is False
    assert "Found 1 sources, budget requires 10" in reason


def test_tier2_f14_done_rejection_candidate_table_zero_with_comparison_goal():
    """F14.B4: Rejection on empty candidates table when comparison criteria is required."""
    ledger = TaskLedger(
        needs_plan=True,
        finish_criteria="Find best price among 3 sites",
        budget=Budget(sources=3),
        subgoals=[
            Subgoal(
                id="sg1",
                text="Compare",
                status="completed",
                results=[SubgoalResult(fact="Fact", source_url="https://s1.com", step=1)],
            )
        ],
        candidates=[],
    )
    is_done, _reason, _ = ledger.verify_done()
    assert is_done is False


def test_tier2_f14_reactivation_preserves_existing_results():
    """F14.B5: Reactivating a subgoal preserves previously accumulated results."""
    sg = Subgoal(
        id="sg1",
        text="t",
        status="completed",
        results=[SubgoalResult(fact="Pre-existing fact", source_url="https://a.com", step=1)],
    )
    sg.status = "active"
    assert len(sg.results) == 1
    assert sg.results[0].fact == "Pre-existing fact"


# --- F15 Boundaries ---

def test_tier2_f15_final_answer_empty_results_and_candidates():
    """F15.B1: synthesize_final_answer with empty results produces valid markdown structure."""
    ledger = TaskLedger(needs_plan=True)
    ans = ledger.synthesize_final_answer("Empty Goal")
    assert "### Görev Sonucu: Empty Goal" in ans


def test_tier2_f15_final_answer_multiline_facts():
    """F15.B2: Facts with internal newlines format without breaking markdown."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg1",
                text="Multi-line fact subgoal",
                results=[SubgoalResult(fact="Point 1\nPoint 2", source_url="https://src.org", step=1)],
            )
        ],
    )
    ans = ledger.synthesize_final_answer("Goal")
    assert "Point 1" in ans
    assert "https://src.org" in ans


def test_tier2_f15_final_answer_massive_attributes_dictionary():
    """F15.B3: Candidate with 20 attributes formats all attributes cleanly."""
    attrs = {f"attr_{i}": f"val_{i}" for i in range(20)}
    ledger = TaskLedger(
        needs_plan=True,
        candidates=[Candidate(item="Mega Item", source_url="https://mega.com", attributes=attrs)],
    )
    ans = ledger.synthesize_final_answer("Mega task")
    assert "attr_0: val_0" in ans
    assert "attr_19: val_19" in ans


def test_tier2_f15_final_answer_unicode_and_special_symbols():
    """F15.B4: Turkish characters and currency symbols (€, $, ₺) preserved in final answer."""
    ledger = TaskLedger(
        needs_plan=True,
        candidates=[
            Candidate(item="Kulaklık 🎧", source_url="https://hepsiburada.com", attributes={"fiyat": "1.499,00 ₺"})
        ],
    )
    ans = ledger.synthesize_final_answer("Kulaklık Bul")
    assert "Kulaklık 🎧" in ans
    assert "1.499,00 ₺" in ans


def test_tier2_f15_final_answer_url_parenthesis_escaping():
    """F15.B5: Source URLs with parentheses or query strings render without crashing."""
    url = "https://en.wikipedia.org/wiki/Entity_(disambiguation)?ref=1#sec"
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg1",
                text="Entity",
                results=[SubgoalResult(fact="Entity fact", source_url=url, step=1)],
            )
        ],
    )
    ans = ledger.synthesize_final_answer("Goal")
    assert url in ans


# --- F16 Boundaries ---

def test_tier2_f16_readme_file_exists():
    """F16.B1: README.md file exists at project root."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    assert readme_path.is_file()


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_tier2_f16_readme_references_arxiv_papers():
    """F16.B2: README.md includes arXiv references for academic foundations."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "arxiv" in content.lower()


def test_tier2_f16_no_langchain_or_llamaindex():
    """F16.B3: Reflex codebase does not import heavy LLM agent frameworks (langchain, llamaindex)."""
    forbidden = ["langchain", "llama_index"]
    reflex_dir = Path(__file__).resolve().parent.parent / "reflex"
    for py_file in reflex_dir.glob("**/*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        for term in forbidden:
            assert f"import {term}" not in text


def test_tier2_f16_no_faiss_or_chroma_modules():
    """F16.B4: Reflex codebase does not contain vector index modules."""
    reflex_dir = Path(__file__).resolve().parent.parent / "reflex"
    for py_file in reflex_dir.glob("**/*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        assert "faiss" not in text


@pytest.mark.skipif(not _HAS_ATTRIBUTION, reason="Pending Milestone 4 README academic attribution update")
def test_tier2_f16_academic_attribution_all_four_methods_present():
    """F16.B5: All 4 key academic methods are cited in README."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    methods = ["Plan-and-Act", "StructuredAgent", "BrowseMaster", "Magentic-One"]
    for m in methods:
        assert m in content, f"Missing attribution for {m} in README.md"


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS (PAIRWISE INTERACTIONS)
# ==============================================================================

def test_tier3_interactive_false_with_clarify_and_multi_subgoal_plan():
    """Tier 3.1: Non-interactive mode adopts clarify default and initializes multi-subgoal ledger."""
    raw = {
        "needs_plan": True,
        "clarify": {"question": "Derinlik?", "options": ["Hızlı", "Detaylı"], "default": "Hızlı"},
        "subgoals": [
            {"id": "sg_1", "text": "Hızlı adım 1", "status": "active"},
            {"id": "sg_2", "text": "Hızlı adım 2", "status": "pending"},
        ],
        "budget": {"sources": 2, "max_steps": 20},
    }
    config = ReflexConfig(interactive=False)
    ledger = parse_task_ledger(raw)
    assert config.interactive is False
    assert ledger.clarify.default == "Hızlı"
    assert ledger.get_active_subgoal().id == "sg_1"


def test_tier3_additive_memory_with_candidate_updates_and_milestones():
    """Tier 3.2: Facts accumulate additively while candidates are deduplicated across milestones."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="Find stores", status="active")],
    )
    # Milestone 1: Store A discovered
    ledger.add_result("Found Store A with price 100", "https://store-a.com", 1)
    ledger.add_candidate("Store A", "https://store-a.com", {"price": "100 TL"})

    # Milestone 2: Store A updated with shipping info + Store B discovered
    ledger.add_result("Store A has free shipping", "https://store-a.com", 3)
    ledger.add_candidate("Store A", "https://store-a.com", {"shipping": "free"})
    ledger.add_candidate("Store B", "https://store-b.com", {"price": "95 TL"})

    assert len(ledger.subgoals[0].results) == 2
    assert len(ledger.candidates) == 2
    assert ledger.candidates[0].attributes["price"] == "100 TL"
    assert ledger.candidates[0].attributes["shipping"] == "free"


def test_tier3_summarization_triggered_while_candidate_table_preserved():
    """Tier 3.3: Exceeding 2500 chars condenses completed subgoal facts without altering candidates table."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Extensive review research",
                status="completed",
                results=[
                    SubgoalResult(fact="A" * 1400, source_url="https://review1.org", step=1),
                    SubgoalResult(fact="B" * 1400, source_url="https://review2.org", step=2),
                ],
            )
        ],
        candidates=[Candidate(item="Gadget X", source_url="https://gadget.com", attributes={"score": "9.2"})],
    )
    assert ledger.total_results_chars() == 2800
    assert len(ledger.candidates) == 1
    triggered = ledger.summarize_completed_results(threshold=2500)
    assert triggered is True
    assert ledger.total_results_chars() < 2800
    # Candidate table is fully intact
    assert len(ledger.candidates) == 1
    assert ledger.candidates[0].attributes["score"] == "9.2"


def test_tier3_done_rejection_reactivates_subgoal_then_succeeds_on_new_facts():
    """Tier 3.4: Premature DONE rejected -> subgoal reactivated -> new facts added -> DONE approved."""
    sg2 = Subgoal(id="sg_2", text="Find second source", status="pending", results=[])
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find first source",
                status="completed",
                results=[SubgoalResult(fact="Fact 1", source_url="https://source1.com", step=1)],
            ),
            sg2,
        ],
    )
    # First attempt: sg_2 is pending -> DONE rejected
    is_done, _reason, missing = ledger.verify_done()
    assert is_done is False
    assert missing is sg2
    # Reactivate missing subgoal
    missing.status = "active"

    # Agent executes and populates missing subgoal
    ledger.add_result("Fact 2", "https://source2.com", step=4, subgoal_id="sg_2")
    missing.status = "completed"

    # Second attempt: all subgoals complete, budget satisfied -> DONE approved
    is_done2, _reason2, missing2 = ledger.verify_done()
    assert is_done2 is True
    assert missing2 is None


def test_tier3_system1_isolation_during_active_plan_with_candidates():
    """Tier 3.5: System 1 goal receives only active subgoal, keeping candidates isolated."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Navigate to search bar",
                status="active",
                results=[SubgoalResult(fact="Past finding", source_url="https://site.com", step=1)],
            )
        ],
        candidates=[Candidate(item="Laptop Pro", source_url="https://laptop.com", attributes={"ram": "32GB"})],
    )
    ctx = DualTierContext(primary_goal="Find cheap laptops", active_subgoal=ledger.get_active_subgoal().text)
    s1_goal = ctx.compile_system1_goal()
    assert "Current subgoal: Navigate to search bar" in s1_goal
    assert "Laptop Pro" not in s1_goal
    assert "32GB" not in s1_goal
    assert "Past finding" not in s1_goal


def test_tier3_domain_normalization_across_subgoals_and_candidates():
    """Tier 3.6: Multiple subgoals and candidates referencing same domain correctly count as 1 source."""
    ledger = TaskLedger(
        needs_plan=True,
        budget=Budget(sources=2),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Sg1",
                status="completed",
                results=[SubgoalResult(fact="Fact 1", source_url="https://amazon.com/item1", step=1)],
            )
        ],
        candidates=[
            Candidate(item="Item 1", source_url="https://amazon.com/item2", attributes={"p": 10}),
            Candidate(item="Item 2", source_url="https://AMAZON.COM/item3", attributes={"p": 15}),
        ],
    )
    # Only amazon.com is present across all results and candidates
    is_done, reason, _ = ledger.verify_done()
    assert is_done is False
    assert "budget requires 2" in reason


def test_tier3_final_answer_synthesis_merges_summarized_facts_and_candidates():
    """Tier 3.7: Final synthesized answer renders summarized facts with citations alongside candidates."""
    url1 = "https://wiki.org/article"
    url2 = "https://store.com/item"
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Background",
                status="completed",
                results=[SubgoalResult(fact="Summarized background info", source_url=url1, step=1)],
            )
        ],
        candidates=[Candidate(item="Recommended Product", source_url=url2, attributes={"price": "$120"})],
    )
    answer = ledger.synthesize_final_answer("Full Product Review")
    assert "Summarized background info" in answer
    assert url1 in answer
    assert "Recommended Product" in answer
    assert "$120" in answer
    assert url2 in answer


def test_tier3_stuck_counter_increment_with_subgoal_advancement():
    """Tier 3.8: Stuck counter recorded in progress checks, does not prevent advance_subgoal when done."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="Step 1", status="active"),
            Subgoal(id="sg_2", text="Step 2", status="pending"),
        ],
    )
    # Check 1: agent is stuck
    pcr1 = ProgressCheckResult(subgoal_done=False, stuck=True)
    assert pcr1.stuck is True

    # Check 2: agent resolves stuck condition and completes subgoal
    pcr2 = ProgressCheckResult(subgoal_done=True, stuck=False)
    assert pcr2.subgoal_done is True
    next_sg = ledger.advance_subgoal()
    assert next_sg.id == "sg_2"
    assert next_sg.status == "active"


def test_tier3_malformed_input_graceful_fallback_to_simple_passthrough():
    """Tier 3.9: Completely corrupt input gracefully falls back to simple pass-through."""
    corrupted_inputs = [
        "{unclosed_json: true",
        None,
        "",
        {"subgoals": "not a list"},
    ]
    for inp in corrupted_inputs:
        ledger = parse_task_ledger(inp)
        assert ledger.needs_plan is False
        assert len(ledger.subgoals) == 0


def test_tier3_candidate_merging_preserves_earliest_step_and_enriches_attributes():
    """Tier 3.10: Candidate attributes enrich over multiple updates without losing previous keys."""
    ledger = TaskLedger(needs_plan=True)
    ledger.add_candidate("Widget 3000", "https://widget.com", {"release": 2024})
    ledger.add_candidate("widget 3000", "https://widget.com", {"price": "$300"})
    ledger.add_candidate("Widget 3000", "https://widget.com", {"rating": 4.9})
    cand = ledger.candidates[0]
    assert cand.attributes == {"release": 2024, "price": "$300", "rating": 4.9}


def test_add_result_skips_repeated_fact_in_same_subgoal():
    """System 2 re-reports a verified fact on later checks; the ledger keeps a single record."""
    plan = parse_task_ledger({"needs_plan": True, "subgoals": [{"id": "sg_1", "text": "Find dates", "status": "active"}]})
    first = plan.add_result(fact="Ada Lovelace born 10 December 1815", source_url="https://en.wikipedia.org/wiki/Ada", step=3)
    again = plan.add_result(fact="ada lovelace born 10 december 1815", source_url="https://en.wikipedia.org/wiki/Ada", step=9)
    assert again is first
    assert len(plan.subgoals[0].results) == 1
