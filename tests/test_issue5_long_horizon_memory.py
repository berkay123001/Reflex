"""Tests for Milestone M2 - Requirement R2 / GitHub Issue #5.

Verifies:
1. R2a: Visited Pages in S2 context (last 40 unique URLs, max 150 chars/URL),
   target_url navigations and initial URLs recorded in ctx.visited_urls.
   compile_system1_goal strictly isolated and free of URLs.
2. R2b: Working memory fallback when plan is None (additive notebook: working_memory_notes,
   casefold deduplication against last note, max 12 notes, max 1500 chars total,
   dropping oldest notes; single note equality preserved). Plan format preserved when plan exists.
   Reset on NEW_GOAL.
3. R2c: CONTINUE_OR_STEER plan preservation and merge: adopt if None,
   preserve completed subgoals and results and append new subgoals if existing,
   do not reset on needs_plan=False.
4. R2d: Loop verifier novel page progression: hub-and-spoke (A->B->A->C->A) not flagged;
   repeated transition without novel page in between flagged;
   2-step and 3-step oscillations continue to be caught.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.context import DualTierContext
from reflex.planner import Subgoal, TaskLedger
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation
from reflex.verifier import verify_step

# =====================================================================
# R2a Tests: Visited Pages in S2 Context & System 1 Isolation
# =====================================================================


def test_visited_urls_s2_context_capped_at_40_and_isolated_from_s1():
    """45 distinct URLs visited: S2 context has last 40; compile_system1_goal has none."""
    ctx = DualTierContext(primary_goal="Survey academic papers on arXiv")

    # Visit 45 distinct URLs
    urls = [f"https://arxiv.org/abs/2610.{i:05d}" for i in range(1, 46)]
    # Add one extra long URL > 150 chars to test length truncation
    long_url = "https://arxiv.org/abs/2610.99999?query=" + "long_param_" * 20
    urls.append(long_url)

    ctx.visited_urls.extend(urls)
    assert len(ctx.visited_urls) == 46

    # S2 Context block check
    s2_block = ctx.build_supervisor_context_block([])
    assert "Visited Pages (already opened; re-open only if needed):" in s2_block

    # First 6 URLs should have dropped off (we keep last 40)
    for u in urls[:6]:
        assert u not in s2_block

    # Last 40 URLs must be present
    for u in urls[6:45]:
        assert u in s2_block

    # Long URL must be truncated to <= 150 chars
    assert len(long_url) > 150
    assert long_url[:140] in s2_block
    assert long_url not in s2_block  # full long URL truncated

    # System 1 goal compilation must strictly isolate S1 from visited URLs
    s1_goal = ctx.compile_system1_goal()
    assert "Visited Pages" not in s1_goal
    assert "arxiv.org" not in s1_goal
    for u in urls:
        assert u not in s1_goal


def test_visited_urls_deduplication_preserving_visit_order():
    """Duplicate visits should not inflate S2 context count; visit order preserved."""
    ctx = DualTierContext(primary_goal="Browse web shop")
    ctx.visited_urls = [
        "https://example.com/home",
        "https://example.com/catalog",
        "https://example.com/home",  # revisit
        "https://example.com/item/1",
        "https://example.com/catalog",  # revisit
        "https://example.com/item/2",
    ]

    s2_block = ctx.build_supervisor_context_block([])
    assert "Visited Pages (already opened; re-open only if needed):" in s2_block
    # Verify each URL is listed once in order of first appearance
    lines = [line.strip() for line in s2_block.splitlines() if line.strip().startswith("- https://example.com")]
    assert lines == [
        "- https://example.com/home",
        "- https://example.com/catalog",
        "- https://example.com/item/1",
        "- https://example.com/item/2",
    ]


def test_reply_user_records_target_url_in_visited_urls():
    """reply_user navigations to target_url must be recorded in ctx.visited_urls."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {
            "url": "https://example.com/initial",
            "actions": [],
        }
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(
            url="https://example.com/initial",
            goal="Test navigation tracking",
        )
        assert "https://example.com/initial" in session.ctx.visited_urls

        mock_interp = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            target_url="https://example.com/target_page",
        )

        def fake_observe(*args, **kwargs):
            return {"url": "https://example.com/target_page", "actions": []}

        mock_browser.observe.side_effect = fake_observe

        with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
            session.reply_user("Go to the target page")

        assert "https://example.com/target_page" in session.ctx.visited_urls


def test_reply_user_new_goal_records_target_url_in_visited_urls():
    """NEW_GOAL navigations to target_url must also be recorded in ctx.visited_urls."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {
            "url": "https://example.com/start",
            "actions": [],
        }
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(
            url="https://example.com/start",
            goal="Initial goal",
        )

        mock_interp = UserMessageInterpretation(
            intent="NEW_GOAL",
            new_goal="Start new search",
            target_url="https://example.com/search",
        )

        def fake_observe(*args, **kwargs):
            return {"url": "https://example.com/search", "actions": []}

        mock_browser.observe.side_effect = fake_observe

        with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
            session.reply_user("Search for new products")

        assert "https://example.com/search" in session.ctx.visited_urls


# =====================================================================
# R2b Tests: Working Memory Fallback (Additive Notebook)
# =====================================================================


def test_additive_notebook_when_plan_is_none():
    """When plan is None: consecutive distinct notes appended; 13th drops oldest; 1500 char cap."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Explore site")
        session.ctx.plan = None

        # Add 3 distinct notes and 1 casefold duplicate
        notes = [
            "Found overview section",
            "found overview section",  # duplicate, must be ignored
            "Checked pricing table",
            "Found contact details",
        ]

        for note in notes:
            sup = SupervisorDecision(
                action_type="UPDATE_SUBGOAL",
                memory=note,
                reasoning="exploration note",
            )
            with patch("reflex.agent.resolve_escalation", return_value=sup):
                session._handle_escalation(reason="test", trigger_type="gate_uncertain", s1_decision=None)

        s2_block = session.ctx.build_supervisor_context_block([])
        assert "Found overview section" in s2_block
        assert "Checked pricing table" in s2_block
        assert "Found contact details" in s2_block
        assert len(session.ctx.working_memory_notes) == 3

        # Push 10 more distinct notes (total 13 distinct notes)
        for i in range(1, 11):
            sup = SupervisorDecision(
                action_type="UPDATE_SUBGOAL",
                memory=f"Additional research finding {i}",
                reasoning=f"finding {i}",
            )
            with patch("reflex.agent.resolve_escalation", return_value=sup):
                session._handle_escalation(reason="test", trigger_type="gate_uncertain", s1_decision=None)

        assert len(session.ctx.working_memory_notes) == 12
        s2_block_13 = session.ctx.build_supervisor_context_block([])
        # 13th note caused the 1st note ("Found overview section") to drop
        assert "Found overview section" not in s2_block_13
        assert "Checked pricing table" in s2_block_13
        assert "Additional research finding 10" in s2_block_13


def test_additive_notebook_character_limit_drops_oldest():
    """Total character cap of 1500 enforced by dropping oldest notes."""
    ctx = DualTierContext(primary_goal="Test char limit")
    # Add 10 notes of 200 characters each (total 2000 chars)
    for i in range(10):
        note = f"Note {i:02d}: " + "X" * 190
        ctx.add_working_memory_note(note)

    assert len(ctx.working_memory_notes) < 10
    total_len = sum(len(n) for n in ctx.working_memory_notes)
    assert total_len <= 1500
    # The oldest notes should have been dropped first
    assert "Note 00:" not in ctx.working_memory
    assert "Note 09:" in ctx.working_memory


def test_additive_notebook_single_note_verbatim_compatibility():
    """Single note must equal note verbatim for legacy assertions."""
    ctx = DualTierContext(primary_goal="Legacy compatibility")
    legacy_note = "Legacy 15-word note: inspected search results"
    ctx.add_working_memory_note(legacy_note)
    assert ctx.working_memory == legacy_note
    assert ctx.working_memory_notes == [legacy_note]


def test_additive_notebook_empty_or_whitespace_ignored():
    """Empty or whitespace notes must not be added to working_memory_notes."""
    ctx = DualTierContext(primary_goal="Whitespace test")
    ctx.add_working_memory_note("")
    ctx.add_working_memory_note("   ")
    assert ctx.working_memory_notes == []
    assert ctx.working_memory == ""


def test_additive_notebook_cleared_on_new_goal():
    """Starting a NEW_GOAL must clear working_memory and working_memory_notes."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Initial task")
        session.ctx.add_working_memory_note("Note to be cleared")
        assert len(session.ctx.working_memory_notes) == 1

        mock_interp = UserMessageInterpretation(
            intent="NEW_GOAL",
            new_goal="Fresh task",
        )
        with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
            session.reply_user("Start over with a fresh task")

        assert session.ctx.working_memory == ""
        assert session.ctx.working_memory_notes == []


def test_plan_format_for_s2_preserved_when_plan_active():
    """When a plan is active, working_memory uses plan.format_for_s2()."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Plan active task")
        session.ctx.plan = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="Search items", status="active"),
            ],
        )

        sup = SupervisorDecision(
            action_type="UPDATE_SUBGOAL",
            memory="escalation memory note",
            reasoning="step reasoning",
        )
        with patch("reflex.agent.resolve_escalation", return_value=sup):
            session._handle_escalation(reason="test", trigger_type="gate_uncertain", s1_decision=None)

        assert "Plan Status: Active" in session.ctx.working_memory
        assert "Search items" in session.ctx.working_memory


# =====================================================================
# R2c Tests: CONTINUE_OR_STEER Plan Preservation and Merge
# =====================================================================


def test_continue_or_steer_plan_preservation_and_merge():
    """CONTINUE_OR_STEER with needs_plan=True adopts if None, merges pending if exists, preserves if False."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Multi-step task")

        # 1. Plan is None -> adopts interp.plan
        session.ctx.plan = None
        new_plan = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="Find papers", status="active"),
                Subgoal(id="sg_2", text="Extract code links", status="pending"),
            ],
        )
        interp1 = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            plan=new_plan,
        )
        with patch("reflex.agent.interpret_user_message", return_value=interp1):
            session.reply_user("Continue with plan")

        assert session.ctx.plan is not None
        assert session.ctx.plan.needs_plan is True
        assert len(session.ctx.plan.subgoals) == 2
        assert session.ctx.active_subgoal == "Find papers"

        # Complete sg_1 and record result
        session.ctx.plan.subgoals[0].status = "completed"
        session.ctx.plan.add_result(fact="Paper 1 has code at github.com/test", source_url="https://example.com/p1", step=1, subgoal_id="sg_1")

        # 2. Existing plan -> merges new pending subgoals, preserves completed subgoals and results
        steer_plan = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="find papers", status="active"),  # overlap (casefold)
                Subgoal(id="sg_2", text="extract code links", status="active"),  # overlap (casefold)
                Subgoal(id="sg_3", text="Compare dataset licenses", status="pending"),  # novel!
            ],
        )
        interp2 = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            plan=steer_plan,
        )
        with patch("reflex.agent.interpret_user_message", return_value=interp2):
            session.reply_user("Also check dataset licenses")

        assert session.ctx.plan.subgoals[0].status == "completed"
        assert len(session.ctx.plan.subgoals[0].results) == 1
        assert "github.com/test" in session.ctx.plan.subgoals[0].results[0].fact
        # Novel subgoal added as pending
        assert len(session.ctx.plan.subgoals) == 3
        assert session.ctx.plan.subgoals[2].text == "Compare dataset licenses"
        assert session.ctx.plan.subgoals[2].status == "pending"

        # 3. needs_plan=False does NOT reset existing plan
        interp3 = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            plan=TaskLedger(needs_plan=False),
        )
        with patch("reflex.agent.interpret_user_message", return_value=interp3):
            session.reply_user("Just keep going")

        assert session.ctx.plan is not None
        assert len(session.ctx.plan.subgoals) == 3


def test_continue_or_steer_with_updated_subgoal():
    """CONTINUE_OR_STEER with explicit updated_subgoal sets active_subgoal directly."""
    with patch("reflex.agent.Browser") as mock_browser_cls:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://example.com", "actions": []}
        mock_browser_cls.return_value = mock_browser

        session = ReflexSession(url="https://example.com", goal="Multi-step task")
        session.ctx.plan = TaskLedger(
            needs_plan=True,
            subgoals=[
                Subgoal(id="sg_1", text="Find papers", status="active"),
            ],
        )

        interp = UserMessageInterpretation(
            intent="CONTINUE_OR_STEER",
            updated_subgoal="Focus exclusively on arxiv papers from 2026",
            plan=TaskLedger(needs_plan=False),
        )
        with patch("reflex.agent.interpret_user_message", return_value=interp):
            session.reply_user("Filter for 2026")

        assert session.ctx.active_subgoal == "Focus exclusively on arxiv papers from 2026"
        assert session.ctx.plan is not None  # not reset


# =====================================================================
# R2d Tests: Loop Verifier Hub-and-Spoke Progression & Cycle Detection
# =====================================================================


def test_verifier_hub_and_spoke_progress_not_flagged():
    """A -> B -> A -> C -> A is hub-and-spoke progress, NOT an oscillation loop."""
    # Step 1: Click paper 1 from list (A -> B)
    # Step 2: Go back to list (B -> A)
    # Step 3: Click paper 2 from list (A -> C)
    # Step 4: Go back to list (C -> A)
    history = [
        {"kind": "click", "action": "Paper 1", "text": None, "after_fingerprint": "fp_B", "effective_progress": True},
        {"kind": "go_back", "action": "Go back", "text": None, "after_fingerprint": "fp_A", "effective_progress": True},
        {"kind": "click", "action": "Paper 2", "text": None, "after_fingerprint": "fp_C", "effective_progress": True},
    ]

    # Now step 4: returning to list A from page C
    res = verify_step(
        action={"kind": "go_back", "label": "Go back"},
        before_page={"url": "https://arxiv.org/abs/paper2", "fingerprint": "fp_C", "actions": []},
        after_page={"url": "https://arxiv.org/search", "fingerprint": "fp_A", "actions": []},
        history=history,
    )

    assert res.oscillation_detected is False
    assert res.loop_detected is False


def test_verifier_repeated_transition_without_novel_page_flagged():
    """A -> B -> A -> B -> A has no novel page in between, so it IS flagged as oscillation."""
    history = [
        {"kind": "click", "action": "Paper 1", "text": None, "after_fingerprint": "fp_B", "effective_progress": True},
        {"kind": "go_back", "action": "Go back", "text": None, "after_fingerprint": "fp_A", "effective_progress": True},
        {"kind": "click", "action": "Paper 1", "text": None, "after_fingerprint": "fp_B", "effective_progress": True},
    ]

    res = verify_step(
        action={"kind": "go_back", "label": "Go back"},
        before_page={"url": "https://arxiv.org/abs/paper1", "fingerprint": "fp_B", "actions": []},
        after_page={"url": "https://arxiv.org/search", "fingerprint": "fp_A", "actions": []},
        history=history,
    )

    assert res.oscillation_detected is True
    assert "Navigation cycle" in res.reason or "oscillation" in res.reason


def test_verifier_multi_step_novel_exploration_chain():
    """Hub -> Spoke1 -> Detail1 -> Hub -> Spoke2 -> Hub is valid exploration."""
    history = [
        {"kind": "click", "action": "Spoke 1", "text": None, "after_fingerprint": "fp_spoke1", "effective_progress": True},
        {"kind": "click", "action": "Detail 1", "text": None, "after_fingerprint": "fp_detail1", "effective_progress": True},
        {"kind": "go_back", "action": "Go Hub", "text": None, "after_fingerprint": "fp_hub", "effective_progress": True},
        {"kind": "click", "action": "Spoke 2", "text": None, "after_fingerprint": "fp_spoke2", "effective_progress": True},
    ]

    # Step 5: Returning to hub from spoke2
    res = verify_step(
        action={"kind": "go_back", "label": "Go Hub"},
        before_page={"url": "https://example.com/spoke2", "fingerprint": "fp_spoke2", "actions": []},
        after_page={"url": "https://example.com/hub", "fingerprint": "fp_hub", "actions": []},
        history=history,
    )

    assert res.oscillation_detected is False
    assert res.loop_detected is False
