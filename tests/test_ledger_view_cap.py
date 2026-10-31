"""Tests for Task Ledger S2 view capping (Milestone 3 / R3).

Verifies deterministic view-only truncation when task ledger exceeds LEDGER_S2_CHAR_LIMIT (6000),
candidate table capping at MAX_CANDIDATES_IN_S2_VIEW (30), non-destructive preservation of
full underlying data for verify_done and synthesize_final_answer, and backward compatibility.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.planner import (
    LEDGER_S2_CHAR_LIMIT,
    MAX_CANDIDATES_IN_S2_VIEW,
    Budget,
    Candidate,
    Subgoal,
    SubgoalResult,
    TaskLedger,
)
from reflex.supervisor import SupervisorDecision


def _build_standard_expected(ledger: TaskLedger) -> str:
    """Helper to independently compute the un-truncated standard S2 output."""
    lines: list[str] = [
        f"Plan Status: {'Active' if ledger.needs_plan else 'No Plan'}",
        f"Finish Criteria: {ledger.finish_criteria or 'Not specified'}",
        f"Budget: {ledger.budget.sources} sources, {ledger.budget.max_steps} max steps",
        "Subgoals & Extracted Results:",
    ]
    for sg in ledger.subgoals:
        status_tag = f"[{sg.status.upper()}]"
        done_when = f" (Done when: {sg.done_when})" if sg.done_when else ""
        lines.append(f"  {status_tag} {sg.id}: {sg.text}{done_when}")
        for r in sg.results:
            src = f" (Source: {r.source_url})" if r.source_url else ""
            lines.append(f"    - [Step {r.step}] {r.fact}{src}")
    if ledger.candidates:
        lines.append("Candidates Table:")
        for c in ledger.candidates:
            attr_str = ", ".join(f"{k}: {v}" for k, v in c.attributes.items())
            src = f" (Source: {c.source_url})" if c.source_url else ""
            lines.append(f"  - {c.item}: {attr_str}{src}")
    return "\n".join(lines)


def test_constants_defined():
    """Verify LEDGER_S2_CHAR_LIMIT and MAX_CANDIDATES_IN_S2_VIEW constants."""
    assert LEDGER_S2_CHAR_LIMIT == 6000
    assert MAX_CANDIDATES_IN_S2_VIEW == 30


def test_format_for_s2_below_limit_unchanged():
    """Ledger below 6000 chars produces byte-for-byte identical output to legacy format."""
    ledger = TaskLedger(
        needs_plan=True,
        finish_criteria="Find 2 comparison products and verify prices",
        budget=Budget(sources=2, max_steps=20),
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Search laptop prices on Amazon",
                status="completed",
                done_when="Found 2 laptops",
                results=[
                    SubgoalResult(
                        fact="Found Laptop A for $999",
                        source_url="https://amazon.com/a",
                        step=2,
                    ),
                    SubgoalResult(
                        fact="Found Laptop B for $1199",
                        source_url="https://amazon.com/b",
                        step=3,
                    ),
                ],
            ),
            Subgoal(
                id="sg_2",
                text="Search laptop prices on BestBuy",
                status="active",
                done_when="Found 2 matching models",
                results=[
                    SubgoalResult(
                        fact="Found Laptop A for $980 on sale",
                        source_url="https://bestbuy.com/a",
                        step=5,
                    ),
                ],
            ),
        ],
        candidates=[
            Candidate(
                item="Laptop A",
                source_url="https://amazon.com/a",
                attributes={"price": "$999", "brand": "BrandX"},
            ),
            Candidate(
                item="Laptop B",
                source_url="https://amazon.com/b",
                attributes={"price": "$1199", "brand": "BrandY"},
            ),
        ],
    )

    expected = _build_standard_expected(ledger)
    assert len(expected) < LEDGER_S2_CHAR_LIMIT

    output = ledger.format_for_s2()

    # Must be strictly byte-for-byte identical
    assert output == expected
    assert output.encode("utf-8") == expected.encode("utf-8")


def test_format_for_s2_above_limit_shrinks():
    """Ledger with many completed subgoals and >30 candidates exceeds 6000 chars, view shrinks below limit."""
    completed_subgoals = []
    for i in range(1, 4):
        results = []
        for src_idx in (1, 2):
            for fact_idx in range(1, 5):
                results.append(
                    SubgoalResult(
                        fact=f"Detailed research finding for phase {i} source {src_idx} fact {fact_idx}: " + ("x" * 150),
                        source_url=f"https://source{src_idx}.com/research",
                        step=i * 10 + src_idx * 5 + fact_idx,
                    )
                )
        completed_subgoals.append(
            Subgoal(
                id=f"sg_{i}",
                text=f"Deep research phase {i}",
                status="completed",
                done_when=f"Phase {i} completed",
                results=results,
            )
        )

    active_subgoal = Subgoal(
        id="sg_active",
        text="Active live investigation",
        status="active",
        done_when="Live step completed",
        results=[
            SubgoalResult(
                fact="Crucial active finding that must not be truncated: " + ("ACTIVEDATA" * 20),
                source_url="https://active-source.com/live",
                step=40,
            )
        ],
    )

    candidates = [
        Candidate(
            item=f"Product Candidate {idx:02d}",
            source_url=f"https://store.com/item/{idx}",
            attributes={"price": f"${idx * 10}", "score": f"{idx}.5"},
        )
        for idx in range(1, 46)  # 45 candidates > MAX_CANDIDATES_IN_S2_VIEW (30)
    ]

    ledger = TaskLedger(
        needs_plan=True,
        finish_criteria="Comprehensive multi-store product comparison",
        budget=Budget(sources=10, max_steps=80),
        subgoals=[*completed_subgoals, active_subgoal],
        candidates=candidates,
    )

    standard_uncapped = _build_standard_expected(ledger)
    assert len(standard_uncapped) > LEDGER_S2_CHAR_LIMIT, (
        f"Test fixture should exceed limit, got {len(standard_uncapped)}"
    )

    capped_view = ledger.format_for_s2()

    # View shrinks significantly
    assert len(capped_view) < len(standard_uncapped)
    assert len(capped_view) <= LEDGER_S2_CHAR_LIMIT

    # Active subgoal results remain intact in full
    assert "Crucial active finding that must not be truncated: " in capped_view
    assert ("ACTIVEDATA" * 20) in capped_view

    # Candidate table displays at most 30 rows + omitted notice
    assert "+ 15 candidates omitted" in capped_view
    assert "Product Candidate 01" in capped_view
    assert "Product Candidate 30" in capped_view
    assert "Product Candidate 31" not in capped_view
    assert "Product Candidate 45" not in capped_view


def test_active_and_pending_subgoals_remain_full():
    """Active and pending subgoals are never truncated even when capped view is active."""
    long_active_fact = "Active Fact " + ("A" * 600)
    long_pending_fact = "Pending Fact " + ("P" * 600)

    # Completed subgoal with lots of facts to push total > 6000
    completed_results = [
        SubgoalResult(
            fact=f"Completed finding {k}: " + ("C" * 280),
            source_url=f"https://source-comp-{k}.com",
            step=k,
        )
        for k in range(25)
    ]

    ledger = TaskLedger(
        needs_plan=True,
        finish_criteria="Goal",
        budget=Budget(sources=5, max_steps=30),
        subgoals=[
            Subgoal(id="sg_done", text="Done", status="completed", results=completed_results),
            Subgoal(
                id="sg_act",
                text="Active Subgoal",
                status="active",
                results=[SubgoalResult(fact=long_active_fact, source_url="https://act.com", step=30)],
            ),
            Subgoal(
                id="sg_pen",
                text="Pending Subgoal",
                status="pending",
                results=[SubgoalResult(fact=long_pending_fact, source_url="https://pen.com", step=31)],
            ),
        ],
    )

    assert len(_build_standard_expected(ledger)) > LEDGER_S2_CHAR_LIMIT
    capped_view = ledger.format_for_s2()

    assert long_active_fact in capped_view
    assert long_pending_fact in capped_view


def test_completed_subgoals_summarized_at_300_chars_per_source():
    """In capped view, completed subgoals merge multiple facts per source and truncate at 300 chars."""
    url1 = "https://example.com/site1"
    url2 = "https://example.com/site2"

    completed_results = [
        SubgoalResult(fact="First finding for site1. " + ("1" * 150), source_url=url1, step=1),
        SubgoalResult(fact="Second finding for site1. " + ("2" * 160), source_url=url1, step=2),
        SubgoalResult(fact="Only finding for site2 short", source_url=url2, step=3),
    ]

    # Add lots of candidates to ensure ledger > 6000
    candidates = [
        Candidate(item=f"Item {i}", attributes={"desc": "d" * 150})
        for i in range(40)
    ]

    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(id="sg_1", text="Search", status="completed", results=completed_results),
        ],
        candidates=candidates,
    )

    assert len(_build_standard_expected(ledger)) > LEDGER_S2_CHAR_LIMIT
    capped = ledger.format_for_s2()

    # url1 had ~350 chars total across two facts -> condensed to <= 300 chars ending with "..."
    assert f"(Source: {url1})" in capped
    assert "First finding for site1." in capped
    assert "..." in capped
    # Site 2 had short finding, remains intact without trailing "..."
    assert "Only finding for site2 short" in capped


def test_candidate_table_capped_at_30_with_omitted_marker():
    """Candidates table shows first 30 items and appends '+ N candidates omitted' when > 30."""
    candidates = [
        Candidate(item=f"Candidate {i:03d}", attributes={"a": "b" * 120})
        for i in range(1, 51)  # 50 candidates
    ]
    ledger = TaskLedger(
        needs_plan=True,
        candidates=candidates,
    )
    assert len(_build_standard_expected(ledger)) > LEDGER_S2_CHAR_LIMIT

    view = ledger.format_for_s2()
    assert "+ 20 candidates omitted" in view
    assert "Candidate 001" in view
    assert "Candidate 030" in view
    assert "Candidate 031" not in view
    assert "Candidate 050" not in view


def test_candidate_table_under_30_no_omitted_marker_when_capped():
    """When subgoals push output over limit but candidates <= 30, no omitted marker is appended."""
    completed_results = [
        SubgoalResult(
            fact=f"Long fact {k} " + ("L" * 300),
            source_url=f"https://site-{k}.com",
            step=k,
        )
        for k in range(25)
    ]
    candidates = [Candidate(item=f"Item {i}", attributes={"k": "v"}) for i in range(10)]

    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="Sg", status="completed", results=completed_results)],
        candidates=candidates,
    )
    assert len(_build_standard_expected(ledger)) > LEDGER_S2_CHAR_LIMIT

    view = ledger.format_for_s2()
    assert "candidates omitted" not in view
    for i in range(10):
        assert f"Item {i}" in view


def test_underlying_data_preserved_for_final_answer_and_verify_done():
    """format_for_s2() preserves raw subgoals & candidates for synthesize_final_answer and verify_done."""
    long_fact_1 = "Fact 1: " + ("F1" * 800)
    long_fact_2 = "Fact 2: " + ("F2" * 800)

    subgoals = [
        Subgoal(
            id="sg_1",
            text="Examine Source 1",
            status="completed",
            results=[
                SubgoalResult(fact=long_fact_1, source_url="https://site1.com/item", step=1),
                SubgoalResult(fact=long_fact_2, source_url="https://site1.com/item", step=2),
            ],
        ),
        Subgoal(
            id="sg_2",
            text="Examine Source 2",
            status="completed",
            results=[
                SubgoalResult(fact="Valid site2 fact", source_url="https://site2.com/item", step=3),
            ],
        ),
    ]

    candidates = [
        Candidate(
            item=f"Product Model {i:02d}",
            source_url=f"https://site1.com/item/{i}",
            attributes={"price": f"${i * 100}", "detail": f"Detail {i}"},
        )
        for i in range(1, 41)  # 40 candidates
    ]

    ledger = TaskLedger(
        needs_plan=True,
        finish_criteria="Evaluate 40 models across 2 sources",
        budget=Budget(sources=2, max_steps=40),
        subgoals=subgoals,
        candidates=candidates,
    )

    orig_subgoals = copy.deepcopy(ledger.subgoals)
    orig_candidates = copy.deepcopy(ledger.candidates)

    assert len(_build_standard_expected(ledger)) > LEDGER_S2_CHAR_LIMIT

    # Call format_for_s2()
    s2_view = ledger.format_for_s2()
    assert len(s2_view) <= LEDGER_S2_CHAR_LIMIT

    # 1. State invariant check: TaskLedger was NOT modified in place
    assert ledger.subgoals == orig_subgoals
    assert ledger.candidates == orig_candidates
    assert len(ledger.subgoals[0].results) == 2
    assert ledger.subgoals[0].results[0].fact == long_fact_1
    assert ledger.subgoals[0].results[1].fact == long_fact_2
    assert len(ledger.candidates) == 40

    # 2. verify_done check: inspects full underlying data
    is_done, _reason, missing = ledger.verify_done(
        visited_urls=["https://site1.com/item", "https://site2.com/item"]
    )
    assert is_done is True
    assert missing is None

    # 3. synthesize_final_answer check: includes ALL 40 candidate items and full facts
    final_answer = ledger.synthesize_final_answer("Comparison of 40 Models")
    for i in range(1, 41):
        assert f"Product Model {i:02d}" in final_answer
    assert long_fact_1 in final_answer
    assert long_fact_2 in final_answer


def test_determinism_identical_byte_output():
    """Calling format_for_s2() multiple times on the same ledger yields 100% identical byte output."""
    candidates = [
        Candidate(item=f"Item {i}", attributes={"price": f"${i}", "info": f"info-{i}" * 10})
        for i in range(35)
    ]
    subgoals = [
        Subgoal(
            id=f"sg_{k}",
            text=f"Subgoal {k}",
            status="completed" if k % 2 == 0 else "active",
            results=[
                SubgoalResult(fact=f"Fact {k}-{m} " + ("M" * 100), source_url=f"https://s{k}.com", step=m)
                for m in range(4)
            ],
        )
        for k in range(10)
    ]
    ledger = TaskLedger(needs_plan=True, subgoals=subgoals, candidates=candidates)

    out1 = ledger.format_for_s2()
    out2 = ledger.format_for_s2()
    out3 = ledger.format_for_s2()

    assert out1 == out2 == out3
    assert out1.encode("utf-8") == out2.encode("utf-8") == out3.encode("utf-8")


def test_custom_max_chars_threshold_override():
    """format_for_s2 accepts custom max_chars argument without breaking default."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Small subgoal",
                status="completed",
                results=[
                    SubgoalResult(fact="A" * 200, source_url="https://site.com", step=1),
                    SubgoalResult(fact="B" * 200, source_url="https://site.com", step=2),
                ],
            )
        ],
    )
    full = ledger.format_for_s2()  # Default limit 6000 -> not capped
    assert len(full) < LEDGER_S2_CHAR_LIMIT
    assert "..." not in full

    # Override limit to 200 chars -> triggers capping
    capped = ledger.format_for_s2(max_chars=200)
    assert len(capped) < len(full)
    assert "..." in capped


def test_agent_escalation_does_not_mutate_task_ledger():
    """ReflexSession escalation does not mutate TaskLedger.subgoals in place."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Non-mutating test")

        # Create a ledger with completed results > 2500 chars (legacy threshold)
        long_fact_1 = "Alpha " + ("1" * 1500)
        long_fact_2 = "Beta " + ("2" * 1500)
        session.ctx.plan = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(
                    id="sg_1",
                    text="Completed search",
                    status="completed",
                    results=[
                        SubgoalResult(fact=long_fact_1, source_url="https://site.com", step=1),
                        SubgoalResult(fact=long_fact_2, source_url="https://site.com", step=2),
                    ],
                ),
                Subgoal(
                    id="sg_2",
                    text="Active next step",
                    status="active",
                ),
            ],
        )

        assert session.ctx.plan.total_results_chars() > 2500

        sup = SupervisorDecision(
            action_type="UPDATE_SUBGOAL",
            memory="escalation memory note",
            reasoning="step reasoning",
        )
        with patch("reflex.agent.resolve_escalation", return_value=sup):
            session._handle_escalation(reason="test", trigger_type="gate_uncertain", s1_decision=None)

        # Previously, summarize_completed_results(threshold=2500) would have collapsed sg_1.results
        # into a single condensed result with truncated fact ending in "..."
        # Now, TaskLedger underlying data MUST remain un-mutated with both original facts!
        assert len(session.ctx.plan.subgoals[0].results) == 2
        assert session.ctx.plan.subgoals[0].results[0].fact == long_fact_1
        assert session.ctx.plan.subgoals[0].results[1].fact == long_fact_2
