"""Tests for Milestone 1 (R1): Cache-friendly S2 prompt ordering and cache_hit_ratio metric.

Verifies:
1. Exact byte-for-byte prefix invariance between consecutive steps for static & slow-changing context.
2. Dynamic tail changes when tactical hint, trajectory, trigger, candidate, and DOM change.
3. All original context sections remain present in the supervisor prompt in correct tiered order.
4. ReflexSession.snapshot() properly computes cache_hit_ratio, handling division by zero safely.
"""

from __future__ import annotations

import json
from typing import Any

from reflex.agent import ReflexConfig, ReflexSession
from reflex.context import ChatMessage, DualTierContext
from reflex.planner import Candidate, Subgoal, SubgoalResult, TaskLedger
from reflex.safety import wrap_untrusted_dom


def _build_full_user_prompt(
    ctx: DualTierContext,
    history: list[dict[str, Any]],
    trigger_type: str,
    escalation_reason: str,
    s1_decision: dict[str, Any] | None,
    page: dict[str, Any],
    elements: list[dict[str, Any]],
) -> str:
    """Helper recreating supervisor prompt assembly exactly as done in resolve_escalation."""
    s1_summary = "None (forced supervisor escalation)"
    if s1_decision:
        s1_summary = json.dumps(
            {
                "operation": s1_decision.get("operation"),
                "target": s1_decision.get("target"),
                "op_confidence": s1_decision.get("confidence"),
                "target_confidence": s1_decision.get("target_confidence"),
                "operation_probabilities": s1_decision.get("operation_probabilities"),
            },
            ensure_ascii=False,
        )

    focus = (ctx.active_subgoal or "").strip() or (ctx.primary_goal or "").strip()
    return (
        f"{ctx.build_supervisor_context_block(history)}\n\n"
        f"Escalation Trigger: {trigger_type} — {escalation_reason}\n"
        f"System 1 Candidate Decision: {s1_summary}\n\n"
        f"{wrap_untrusted_dom(page, elements, focus=focus)}"
    )


def test_consecutive_steps_identical_cacheable_prefix() -> None:
    """Step 1 and Step 2 with same goal/ledger/pages share an identical byte-for-byte cacheable prefix."""
    plan = TaskLedger(
        needs_plan=True,
        finish_criteria="Found founding year and verified",
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find official history page",
                status="completed",
                results=[SubgoalResult(fact="Established in 1989", source_url="https://example.org/about", step=1)],
            ),
            Subgoal(id="sg_2", text="Verify registration charter", status="active"),
        ],
        candidates=[
            Candidate(
                item="Charter Document",
                source_url="https://example.org/charter",
                attributes={"status": "official"},
            )
        ],
    )

    ctx = DualTierContext(
        primary_goal="Find founding year of organization",
        active_subgoal="Verify registration charter",
        constraints=["do not submit payment forms", "stay within example.org"],
        user_present=False,
        plan=plan,
        working_memory="Organization founded late 20th century.",
    )
    ctx.user_facts = {"org_name": "Example Foundation", "jurisdiction": "UK"}
    ctx.visited_urls = ["https://example.org", "https://example.org/about"]
    ctx.visit_queue = ["https://example.org/charter", "https://example.org/team"]
    ctx.subgoal_queue = ["Download charter PDF"]
    ctx.completed_subgoals = ["Find official history page"]
    ctx.rejected_links = ["https://example.org/careers"]
    ctx.ineffective_clicks = [("https://example.org/about", "Careers Button", "link")]
    ctx.record_dom_fact(key="heading", value="About Example Foundation", source_url="https://example.org/about", step=1)
    ctx.record_dom_fact(key="est_year", value="1989", source_url="https://example.org/about", step=2)
    ctx.older_chat_summary = "User requested verification of 1989 founding year."
    ctx.add_user_message("Please check 1989 founding year.", kind="chat")
    ctx.add_agent_message("Navigating to about page.", kind="chat")

    # Step 1: dynamic state
    ctx.tactical_hint = "Look for charter link in navigation menu"
    history_step1: list[dict[str, Any]] = [
        {
            "step": 1,
            "actor": "S1",
            "operation": "CLICK",
            "action": "About Us",
            "text": "",
            "effective_progress": True,
            "url": "https://example.org",
        }
    ]
    trigger_1 = "confidence_drop"
    reason_1 = "Confidence 0.45 fell below threshold 0.70"
    s1_decision_1 = {"operation": "CLICK", "target": 12, "confidence": 0.45}
    page_1 = {"url": "https://example.org/about", "title": "About Us", "text": "About content"}
    elements_1 = [{"index": 12, "role": "link", "name": "Charter"}]

    prompt_step1 = _build_full_user_prompt(
        ctx=ctx,
        history=history_step1,
        trigger_type=trigger_1,
        escalation_reason=reason_1,
        s1_decision=s1_decision_1,
        page=page_1,
        elements=elements_1,
    )

    # Step 2: same static and slow-changing context, but different tactical hint, trajectory, trigger, decision, and DOM
    ctx.tactical_hint = "Scroll down to check footer links"
    history_step2: list[dict[str, Any]] = history_step1 + [
        {
            "step": 2,
            "actor": "S2",
            "operation": "SCROLL_DOWN",
            "action": "Footer",
            "text": "",
            "effective_progress": False,
            "url": "https://example.org/about",
        }
    ]
    trigger_2 = "stuck_loop"
    reason_2 = "Repeated action with no progress"
    s1_decision_2 = {"operation": "SCROLL_DOWN", "target": None, "confidence": 0.30}
    page_2 = {"url": "https://example.org/about#footer", "title": "About Us - Footer", "text": "Footer text"}
    elements_2 = [{"index": 99, "role": "link", "name": "Back to top"}]

    prompt_step2 = _build_full_user_prompt(
        ctx=ctx,
        history=history_step2,
        trigger_type=trigger_2,
        escalation_reason=reason_2,
        s1_decision=s1_decision_2,
        page=page_2,
        elements=elements_2,
    )

    split_marker = "Current Tactical Hint:"
    assert split_marker in prompt_step1, "Tactical hint must be present in prompt 1"
    assert split_marker in prompt_step2, "Tactical hint must be present in prompt 2"

    prefix_1, _ = prompt_step1.split(split_marker, 1)
    prefix_2, _ = prompt_step2.split(split_marker, 1)

    # Prefix must be non-empty and substantial
    assert len(prefix_1) > 200, "Prefix should contain full static + slow-changing tiers"

    # Byte-for-byte exact identity
    assert prefix_1 == prefix_2
    assert prefix_1.encode("utf-8") == prefix_2.encode("utf-8")


def test_dynamic_tail_changes_when_tactical_hint_and_dom_change() -> None:
    """The dynamic tail (tactical hint, trajectory, trigger, candidate, DOM) diverges while prefix remains stable."""
    ctx = DualTierContext(primary_goal="Compare product prices", active_subgoal="Find product A")
    ctx.user_facts = {"budget": "$100"}

    # Step 1
    ctx.tactical_hint = "Hint Step 1: Click first result"
    history_1: list[dict[str, Any]] = [
        {"step": 1, "actor": "S1", "operation": "CLICK", "action": "Item 1", "effective_progress": True, "url": "https://shop.com"}
    ]
    prompt_1 = _build_full_user_prompt(
        ctx=ctx,
        history=history_1,
        trigger_type="trigger_alpha",
        escalation_reason="reason_alpha",
        s1_decision={"operation": "CLICK", "target": 1, "confidence": 0.5},
        page={"url": "https://shop.com/1", "title": "Product 1"},
        elements=[{"index": 1, "role": "button", "name": "Buy"}],
    )

    # Step 2
    ctx.tactical_hint = "Hint Step 2: Check alternative seller"
    history_2: list[dict[str, Any]] = history_1 + [
        {"step": 2, "actor": "S2", "operation": "CLICK", "action": "Item 2", "effective_progress": True, "url": "https://shop.com/2"}
    ]
    prompt_2 = _build_full_user_prompt(
        ctx=ctx,
        history=history_2,
        trigger_type="trigger_beta",
        escalation_reason="reason_beta",
        s1_decision={"operation": "GOTO_URL", "target": None, "confidence": 0.8},
        page={"url": "https://shop.com/2", "title": "Product 2"},
        elements=[{"index": 2, "role": "button", "name": "Compare"}],
    )

    split_marker = "Current Tactical Hint:"
    prefix_1, tail_1 = prompt_1.split(split_marker, 1)
    prefix_2, tail_2 = prompt_2.split(split_marker, 1)

    # Prefix is identical
    assert prefix_1 == prefix_2

    # Dynamic tail is different
    assert tail_1 != tail_2

    # Tail contains dynamic items
    assert "Hint Step 1" in tail_1
    assert "trigger_alpha" in tail_1
    assert "Product 1" in tail_1

    assert "Hint Step 2" in tail_2
    assert "trigger_beta" in tail_2
    assert "Product 2" in tail_2


def test_all_original_sections_present() -> None:
    """When all context fields are populated, all section headers are present in strict tiered cache order."""
    plan = TaskLedger(
        needs_plan=True,
        finish_criteria="Finish all subgoals",
        subgoals=[Subgoal(id="sg_1", text="Subgoal One", status="completed")],
    )
    ctx = DualTierContext(
        primary_goal="Complete research task",
        active_subgoal="Subgoal Two",
        constraints=["do not close tabs", "no purchases"],
        user_present=False,
        plan=plan,
        working_memory="Additional research notes.",
    )
    ctx.user_facts = {"user_pref": "concise"}
    ctx.visited_urls = ["https://example.com/start"]
    ctx.visit_queue = ["https://example.com/next"]
    ctx.subgoal_queue = ["Subgoal Three"]
    ctx.completed_subgoals = ["Subgoal One"]
    ctx.rejected_links = ["https://example.com/ads"]
    ctx.ineffective_clicks = [("https://example.com/start", "Dead Button", "button")]
    ctx.record_dom_fact(key="meta", value="verified", source_url="https://example.com/start", step=1)
    ctx.older_chat_summary = "Earlier conversation summary."
    ctx.chat_log = [ChatMessage(role="user", content="Start the task", kind="chat")]
    ctx.tactical_hint = "Current tactical suggestion."

    history = [
        {
            "step": 1,
            "actor": "S1",
            "operation": "CLICK",
            "action": "Next",
            "effective_progress": True,
            "url": "https://example.com/start",
        }
    ]

    prompt = _build_full_user_prompt(
        ctx=ctx,
        history=history,
        trigger_type="test_trigger",
        escalation_reason="test_reason",
        s1_decision={"operation": "CLICK", "target": 1, "confidence": 0.6},
        page={"url": "https://example.com/start", "title": "Start Page"},
        elements=[{"index": 1, "role": "button", "name": "Next"}],
    )

    expected_headers = [
        # Tier 1: Static
        "Primary Goal:",
        "Active Constraints:",
        "User Availability:",
        "User Facts (verified from human user):",
        # Tier 2: Slow-changing
        "Structured Task Ledger:",
        "Cumulative Working Memory:",
        "Already Visited & Rejected Links (DO NOT CLICK THESE AGAIN):",
        "Ineffective Clicks on Current Page — bu buton işe yaramadı (DO NOT CLICK THESE AGAIN):",
        "Visited Pages (already opened; re-open only if needed):",
        "Queued URLs (collected earlier; open with GOTO_URL directly, no need to return to the source page):",
        "Active Subgoal:",
        "Remaining Subgoals:",
        "Completed Subgoals:",
        "Verified DOM Observations:",
        "Earlier Chat Summary:",
        "Recent User-Agent Conversation:",
        # Tier 3: Dynamic / step-changing
        "Current Tactical Hint:",
        "Recent Execution Trajectory (most recent last):",
        "Escalation Trigger:",
        "System 1 Candidate Decision:",
        "<untrusted_page_data>",
    ]

    for header in expected_headers:
        assert header in prompt, f"Expected header '{header}' missing from supervisor prompt"

    # Verify strictly monotonic ordering
    indices = [prompt.index(header) for header in expected_headers]
    assert indices == sorted(indices), (
        f"Headers not in strictly monotonic tiered order. Indices: {dict(zip(expected_headers, indices))}"
    )


def test_snapshot_usage_includes_cache_hit_ratio() -> None:
    """ReflexSession.snapshot() correctly calculates cache_hit_ratio in metrics['usage']."""
    cfg = ReflexConfig()

    # Case 1: Zero prompt tokens -> ratio is 0.0 (safe division)
    session_zero = ReflexSession(url="https://example.com", goal="Test zero tokens", config=cfg)
    snap_zero = session_zero.snapshot()
    usage_zero = snap_zero["metrics"]["usage"]
    assert "cache_hit_ratio" in usage_zero
    assert usage_zero["prompt_tokens"] == 0
    assert usage_zero["cached_tokens"] == 0
    assert usage_zero["cache_hit_ratio"] == 0.0

    # Case 2: Decisions with cached_tokens via prompt_tokens_details
    session_details = ReflexSession(url="https://example.com", goal="Test details tokens", config=cfg)
    session_details.decisions.append(
        {
            "usage": {
                "prompt_tokens": 1520,
                "completion_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 1393},
                "total_tokens": 1600,
            }
        }
    )
    snap_details = session_details.snapshot()
    usage_details = snap_details["metrics"]["usage"]
    assert usage_details["prompt_tokens"] == 1520
    assert usage_details["cached_tokens"] == 1393
    # 1393 / 1520 = 0.916447... -> round(..., 4) == 0.9164
    assert usage_details["cache_hit_ratio"] == 0.9164

    # Case 3: Decisions with direct top-level cached_tokens
    session_direct = ReflexSession(url="https://example.com", goal="Test direct cached", config=cfg)
    session_direct.decisions.append(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "cached_tokens": 750,
                "total_tokens": 1050,
            }
        }
    )
    snap_direct = session_direct.snapshot()
    usage_direct = snap_direct["metrics"]["usage"]
    assert usage_direct["prompt_tokens"] == 1000
    assert usage_direct["cached_tokens"] == 750
    assert usage_direct["cache_hit_ratio"] == 0.75

    # Case 4: Multiple calls across decisions, escalations, text_calls
    session_multi = ReflexSession(url="https://example.com", goal="Test multi calls", config=cfg)
    session_multi.decisions.append(
        {"usage": {"prompt_tokens": 500, "cached_tokens": 200, "completion_tokens": 20}}
    )
    session_multi.escalations.append(
        {"usage": {"prompt_tokens": 500, "cached_tokens": 300, "completion_tokens": 40}}
    )
    snap_multi = session_multi.snapshot()
    usage_multi = snap_multi["metrics"]["usage"]
    assert usage_multi["prompt_tokens"] == 1000
    assert usage_multi["cached_tokens"] == 500
    assert usage_multi["cache_hit_ratio"] == 0.50
