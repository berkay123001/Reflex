"""Empirical Adversarial Stress Test Suite for Milestone M1 (challenger_m1_2_run2).

Tests:
1. verify_done() behavior under zero subgoals, missing results, candidate counts, visited URL lists.
2. advance_subgoal() state transitions and empty/all-completed lists.
3. Serialization / deserialization cycles (model_dump -> JSON -> parse_task_ledger).
4. Concurrency / thread safety simulation on TaskLedger and DualTierContext.
"""

from __future__ import annotations

import json
import threading

from reflex.context import DualTierContext
from reflex.planner import (
    Budget,
    Candidate,
    Subgoal,
    SubgoalResult,
    TaskLedger,
    _extract_domain,
    parse_task_ledger,
)

# ==============================================================================
# AREA 1: verify_done() Behavior
# ==============================================================================


def test_verify_done_zero_subgoals_no_plan():
    """Verify verify_done passes immediately when needs_plan=False and subgoals is empty."""
    ledger = TaskLedger(needs_plan=False, subgoals=[])
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is True
    assert "No plan required" in reason
    assert missing_sg is None


def test_verify_done_zero_subgoals_needs_plan_fails_without_sources():
    """Verify verify_done fails when needs_plan=True and subgoals is empty (0 sources < budget 1)."""
    ledger = TaskLedger(needs_plan=True, subgoals=[], budget=Budget(sources=1))
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is False
    assert "budget requires 1" in reason
    assert missing_sg is None


def test_verify_done_missing_results_rejected():
    """Verify verify_done rejects completed subgoals with empty results."""
    sg1 = Subgoal(id="sg_1", text="Find info", status="completed", results=[])
    ledger = TaskLedger(needs_plan=True, subgoals=[sg1], budget=Budget(sources=1))
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is False
    assert "has no verified results" in reason
    assert missing_sg is sg1


def test_verify_done_candidate_count_deficit_rejected():
    """Verify verify_done rejects when candidates exist but count < budget.sources."""
    sg1 = Subgoal(
        id="sg_1",
        text="Find candidate",
        status="completed",
        results=[SubgoalResult(fact="Found phone", source_url="https://store1.com", step=1)],
    )
    c1 = Candidate(item="Phone X", source_url="https://store1.com")
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[sg1],
        candidates=[c1],
        budget=Budget(sources=3),
    )
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is False
    assert "Candidate table has 1 items, expected at least 3" in reason
    assert missing_sg is sg1


def test_verify_done_candidate_count_sufficient_passes():
    """Verify verify_done passes when candidate count and distinct sources meet budget."""
    sg1 = Subgoal(
        id="sg_1",
        text="Find candidates",
        status="completed",
        results=[SubgoalResult(fact="Found phones", source_url="https://store1.com", step=1)],
    )
    candidates = [
        Candidate(item="Phone X", source_url="https://store1.com"),
        Candidate(item="Phone Y", source_url="https://store2.com"),
        Candidate(item="Phone Z", source_url="https://store3.com"),
    ]
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[sg1],
        candidates=candidates,
        budget=Budget(sources=3),
    )
    is_done, reason, missing_sg = ledger.verify_done()
    assert is_done is True
    assert "All completion criteria satisfied" in reason
    assert missing_sg is None


def test_verify_done_visited_urls_domain_normalization():
    """Verify verify_done collapses visited URLs with www, port, and scheme to distinct domains."""
    sg1 = Subgoal(
        id="sg_1",
        text="Search docs",
        status="completed",
        results=[SubgoalResult(fact="Found docs", source_url="https://docs.python.org/3/", step=1)],
    )
    visited = [
        "https://www.docs.python.org/3/library/",
        "http://docs.python.org:8080/3/tutorial/",
        "https://pypi.org/project/reflex/",
    ]
    ledger = TaskLedger(needs_plan=True, subgoals=[sg1], budget=Budget(sources=2))
    is_done, reason, _ = ledger.verify_done(visited_urls=visited)
    assert is_done is True
    assert "All completion criteria satisfied" in reason


def test_verify_done_pseudo_url_leakage():
    """Adversarial check: browser pseudo-schemes like about:blank return empty string."""
    domain_about = _extract_domain("about:blank")
    domain_js = _extract_domain("javascript:void(0)")
    assert domain_about == ""
    assert domain_js == ""


# ==============================================================================
# AREA 2: advance_subgoal() State Transitions
# ==============================================================================


def test_advance_subgoal_empty_list():
    """Verify advance_subgoal returns None safely when subgoals is empty."""
    ledger = TaskLedger(needs_plan=True, subgoals=[])
    res = ledger.advance_subgoal()
    assert res is None


def test_advance_subgoal_all_completed_is_idempotent():
    """Verify advance_subgoal is idempotent and returns None when all subgoals are completed."""
    s1 = Subgoal(id="1", text="S1", status="completed")
    s2 = Subgoal(id="2", text="S2", status="completed")
    ledger = TaskLedger(needs_plan=True, subgoals=[s1, s2])
    res = ledger.advance_subgoal()
    assert res is None
    assert s1.status == "completed"
    assert s2.status == "completed"


def test_advance_subgoal_standard_progression():
    """Verify step-by-step advance transitions from first to last subgoal."""
    s1 = Subgoal(id="1", text="Step 1", status="active")
    s2 = Subgoal(id="2", text="Step 2", status="pending")
    s3 = Subgoal(id="3", text="Step 3", status="pending")
    ledger = TaskLedger(needs_plan=True, subgoals=[s1, s2, s3])

    # First advance: completes s1, activates s2
    res1 = ledger.advance_subgoal()
    assert res1 is s2
    assert s1.status == "completed"
    assert s2.status == "active"
    assert s3.status == "pending"

    # Second advance: completes s2, activates s3
    res2 = ledger.advance_subgoal()
    assert res2 is s3
    assert s2.status == "completed"
    assert s3.status == "active"

    # Third advance: completes s3, returns None (all completed)
    res3 = ledger.advance_subgoal()
    assert res3 is None
    assert s3.status == "completed"


def test_advance_subgoal_skips_pending_when_no_active_subgoal():
    """Adversarial check: if no active subgoal exists, advance_subgoal activates pending instead of completing it."""
    s1 = Subgoal(id="1", text="Step 1", status="completed")
    s2 = Subgoal(id="2", text="Step 2", status="pending")
    ledger = TaskLedger(needs_plan=True, subgoals=[s1, s2])

    res = ledger.advance_subgoal()
    assert res is s2
    assert s2.status == "active"


# ==============================================================================
# AREA 3: Serialization / Deserialization Cycles
# ==============================================================================


def test_serialization_cycle_completed_status_reset_bug():
    """Adversarial check: model_dump -> parse_task_ledger preserves completed status without resetting."""
    ledger = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Step 1",
                status="completed",
                results=[SubgoalResult(fact="Done 1", source_url="https://a.com")],
            ),
            Subgoal(
                id="sg_2",
                text="Step 2",
                status="completed",
                results=[SubgoalResult(fact="Done 2", source_url="https://b.com")],
            ),
        ],
        budget=Budget(sources=2),
    )
    # Before serialization: verify_done passes
    is_done_before, _, _ = ledger.verify_done()
    assert is_done_before is True

    # Roundtrip through JSON
    dumped = ledger.model_dump()
    json_str = json.dumps(dumped)
    restored = parse_task_ledger(json_str)

    # Status preservation: restored subgoals remain 'completed'
    assert restored.subgoals[0].status == "completed"
    assert restored.subgoals[1].status == "completed"
    is_done_after, reason, missing_sg = restored.verify_done()
    assert is_done_after is True
    assert "All completion criteria satisfied" in reason
    assert missing_sg is None


def test_parse_task_ledger_crashes_on_non_numeric_step():
    """Adversarial check: parse_task_ledger safely handles non-numeric step without raising."""
    payload = {
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Step 1",
                "results": [{"fact": "Discovered fact", "step": "second_step"}],
            }
        ],
    }
    ledger = parse_task_ledger(payload)
    assert ledger.needs_plan is True
    assert ledger.subgoals[0].results[0].step == 0


def test_parse_task_ledger_crashes_on_malformed_candidate_attributes():
    """Adversarial check: parse_task_ledger safely handles non-dict attributes without raising."""
    payload_str_attr = {
        "needs_plan": True,
        "candidates": [{"item": "Gadget", "attributes": "inexpensive"}],
    }
    ledger1 = parse_task_ledger(payload_str_attr)
    assert ledger1.candidates[0].attributes == {}

    payload_list_attr = {
        "needs_plan": True,
        "candidates": [{"item": "Gadget", "attributes": ["tag1", "tag2"]}],
    }
    ledger2 = parse_task_ledger(payload_list_attr)
    assert ledger2.candidates[0].attributes == {}


def test_parse_task_ledger_crashes_on_non_iterable_options():
    """Adversarial check: parse_task_ledger safely handles non-iterable options without raising."""
    payload_none = {
        "needs_plan": True,
        "clarify": {"question": "How many?", "options": None},
    }
    ledger1 = parse_task_ledger(payload_none)
    assert ledger1.clarify.options == []

    payload_int = {
        "needs_plan": True,
        "clarify": {"question": "How many?", "options": 42},
    }
    ledger2 = parse_task_ledger(payload_int)
    assert ledger2.clarify.options == []


# ==============================================================================
# AREA 4: Concurrency / Thread Safety Simulation
# ==============================================================================


def test_concurrency_advance_subgoal_rapid_progression():
    """Stress test: concurrent threads executing advance_subgoal() sequentially drain all subgoals."""
    subgoals = [Subgoal(id=f"sg_{i}", text=f"Subgoal {i}", status="pending") for i in range(20)]
    ledger = TaskLedger(needs_plan=True, subgoals=subgoals)

    returned_subgoals: list[str | None] = []
    lock = threading.Lock()

    def advance_worker():
        sg = ledger.advance_subgoal()
        with lock:
            returned_subgoals.append(sg.id if sg else None)

    threads = [threading.Thread(target=advance_worker) for _ in range(21)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    completed_count = sum(1 for s in ledger.subgoals if s.status == "completed")
    assert completed_count == 20


def test_concurrency_dual_tier_context_snapshot_and_writes():
    """Stress test: concurrent reader (snapshot_dict) and writers (add_user_message, plan edits)."""
    ctx = DualTierContext(primary_goal="Concurrent Execution")
    ctx.plan = TaskLedger(
        needs_plan=True,
        subgoals=[Subgoal(id="sg_1", text="Subgoal 1", status="active")],
    )

    errors: list[tuple[str, str]] = []

    def reader():
        for _ in range(200):
            try:
                snap = ctx.snapshot_dict()
                if "primary_goal" not in snap or snap["plan"] is None:
                    errors.append(("reader", "invalid snapshot"))
            except (KeyError, RuntimeError, ValueError) as exc:
                errors.append(("reader", str(exc)))

    def writer_messages():
        for i in range(200):
            try:
                ctx.add_user_message(f"Message {i}")
            except (RuntimeError, ValueError) as exc:
                errors.append(("writer_messages", str(exc)))

    def writer_plan():
        for i in range(200):
            try:
                if ctx.plan is not None:
                    ctx.plan.add_result(fact=f"Fact {i}", source_url=f"https://source{i}.com", step=i)
                    ctx.plan.add_candidate(item=f"Item {i % 10}", source_url=f"https://store{i % 5}.com")
            except (RuntimeError, ValueError) as exc:
                errors.append(("writer_plan", str(exc)))

    threads = [
        threading.Thread(target=reader),
        threading.Thread(target=writer_messages),
        threading.Thread(target=writer_plan),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrent errors: {errors}"
