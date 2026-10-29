"""Unit and Integration Tests for Milestone M2: Additive Memory & Context Integration.

Covers:
1. UserMessageInterpretation & interpret_user_message TaskLedger support (R1).
2. Session startup plan ingestion & config.interactive clarify handling (R1, R2).
3. Anti-clobbering in _handle_escalation (Plan-as-Memory, candidate tracking, 0-3 distilled facts, 2500-char summarization) (R2).
4. System 1 input isolation in compile_system1_goal (R2).
5. System 2 structured plan injection in build_supervisor_context_block (R2).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig
from reflex.context import DualTierContext
from reflex.planner import Candidate, Subgoal, SubgoalResult, TaskLedger, parse_task_ledger
from reflex.supervisor import (
    SupervisorDecision,
    UserMessageInterpretation,
    interpret_user_message,
)

# ==============================================================================
# 1. Supervisor UserMessageInterpretation & interpret_user_message tests
# ==============================================================================

def test_user_message_interpretation_plan_field_default():
    """UserMessageInterpretation defaults plan to None and accepts TaskLedger."""
    interp = UserMessageInterpretation(intent="CHAT_ONLY")
    assert interp.plan is None

    ledger = TaskLedger(needs_plan=True, finish_criteria="Find 2 sources")
    interp_with_plan = UserMessageInterpretation(intent="NEW_GOAL", plan=ledger)
    assert interp_with_plan.plan is not None
    assert interp_with_plan.plan.needs_plan is True
    assert interp_with_plan.plan.finish_criteria == "Find 2 sources"


def test_interpret_user_message_parses_flat_plan():
    """interpret_user_message parses TaskLedger when root JSON contains needs_plan and subgoals."""
    mock_parsed = {
        "intent": "CONTINUE_OR_STEER",
        "needs_plan": True,
        "subgoals": [
            {"id": "sg_1", "text": "Subgoal 1", "done_when": "Done 1", "status": "active", "results": []},
            {"id": "sg_2", "text": "Subgoal 2", "done_when": "Done 2", "status": "pending", "results": []},
        ],
        "candidates": [{"item": "Item A", "source_url": "https://a.com", "attributes": {"price": "$10"}}],
        "finish_criteria": "Verify items",
        "budget": {"sources": 2, "max_steps": 20},
    }
    with patch("reflex.supervisor._chat_completion", return_value=(mock_parsed, 100, 0.001, "test-model")):
        ctx = DualTierContext(primary_goal="Compare products")
        interp = interpret_user_message(
            user_message="Compare products",
            ctx=ctx,
            page={"url": "https://example.com"},
            history=[],
            pending_fact_key=None,
            supervisor_model="test-model",
        )

    assert interp.plan is not None
    assert interp.plan.needs_plan is True
    assert len(interp.plan.subgoals) == 2
    assert interp.plan.subgoals[0].id == "sg_1"
    assert len(interp.plan.candidates) == 1
    assert interp.plan.candidates[0].item == "Item A"
    assert interp.plan.budget.sources == 2


def test_interpret_user_message_parses_nested_plan_key():
    """interpret_user_message parses TaskLedger when returned under a 'plan' dictionary."""
    mock_parsed = {
        "intent": "NEW_GOAL",
        "new_goal": "Investigate AI trends",
        "plan": {
            "needs_plan": True,
            "subgoals": [{"id": "sg_1", "text": "Read arXiv papers", "status": "active"}],
            "budget": {"sources": 3, "max_steps": 25},
        },
    }
    with patch("reflex.supervisor._chat_completion", return_value=(mock_parsed, 80, 0.001, "test-model")):
        ctx = DualTierContext(primary_goal="Old goal")
        interp = interpret_user_message(
            user_message="görevi değiştir ve güncel gelişmeleri araştır",
            ctx=ctx,
            page={"url": "https://example.com"},
            history=[],
            pending_fact_key=None,
            supervisor_model="test-model",
        )

    assert interp.intent == "NEW_GOAL"
    assert interp.plan is not None
    assert interp.plan.needs_plan is True
    assert interp.plan.subgoals[0].text == "Read arXiv papers"
    assert interp.plan.budget.sources == 3


def test_interpret_user_message_omits_plan_for_simple_tasks():
    """interpret_user_message leaves plan as None when model response lacks planning keys."""
    mock_parsed = {
        "intent": "CHAT_ONLY",
        "direct_reply": "Sayfa başlığı: Example Domain",
    }
    with patch("reflex.supervisor._chat_completion", return_value=(mock_parsed, 50, 0.0005, "test-model")):
        ctx = DualTierContext(primary_goal="Simple goal")
        interp = interpret_user_message(
            user_message="ekranda ne var?",
            ctx=ctx,
            page={"url": "https://example.com"},
            history=[],
            pending_fact_key=None,
            supervisor_model="test-model",
        )

    assert interp.intent == "CHAT_ONLY"
    assert interp.plan is None


# ==============================================================================
# 2. Session startup plan ingestion & config.interactive tests
# ==============================================================================

def _create_mock_session(
    interp_plan: TaskLedger | None = None,
    interactive: bool = False,
    extra_interp_kwargs: dict | None = None,
) -> ReflexSession:
    """Helper to instantiate ReflexSession with mocked browser and supervisor initialization."""
    interp_kwargs = {
        "intent": "CONTINUE_OR_STEER",
        "allow_arbitrary_choices": False,
        "constraints": [],
        "plan": interp_plan,
    }
    if extra_interp_kwargs:
        interp_kwargs.update(extra_interp_kwargs)
    mock_interp = UserMessageInterpretation(**interp_kwargs)

    with (
        patch("reflex.agent.Browser") as mock_browser_cls,
        patch("reflex.agent.interpret_user_message", return_value=mock_interp),
    ):
        mock_browser_inst = MagicMock()
        mock_browser_inst.observe.return_value = {
            "url": "https://example.com",
            "title": "Example Page",
            "text": "Hello world",
            "actions": [],
        }
        mock_browser_cls.return_value = mock_browser_inst

        config = ReflexConfig(decision_backend="rule", interactive=interactive)
        session = ReflexSession(
            goal="Research historical figures",
            url="https://example.com",
            config=config,
        )
        return session


def test_startup_ingests_plan_and_activates_first_subgoal():
    """Session startup ingests active TaskLedger and sets active_subgoal."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [
            {"id": "sg_1", "text": "Find Ada Lovelace details", "status": "active"},
            {"id": "sg_2", "text": "Find Charles Babbage details", "status": "pending"},
        ],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    assert session.ctx.plan is not None
    assert session.ctx.plan.needs_plan is True
    assert session.ctx.active_subgoal == "Find Ada Lovelace details"
    assert session.status == "ready"


def test_startup_non_interactive_clarify_auto_adopts_default():
    """When interactive=False, clarify default is auto-adopted without pausing or asking."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "clarify": {
            "question": "Kaç kaynak taransın?",
            "options": ["2", "3", "5"],
            "default": "3",
        },
        "subgoals": [{"id": "sg_1", "text": "Scan literature", "status": "active"}],
        "budget": {"sources": 1, "max_steps": 25},
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    assert session.status == "ready"
    assert session.pending_fact_key is None
    assert session.ctx.user_facts.get("clarify_scope") == "3"
    assert session.ctx.plan.budget.sources == 3
    # Chat message was added informing about non-interactive auto-adoption
    assert any("Varsayılan kapsam kabul edildi" in m.content for m in session.ctx.chat_log)


def test_startup_interactive_clarify_pauses_session_and_asks_user():
    """When interactive=True, clarify pauses session (waiting_user) and prompts user."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "clarify": {
            "question": "Kaç kaynak taransın?",
            "options": ["2", "3", "5"],
            "default": "3",
        },
        "subgoals": [{"id": "sg_1", "text": "Scan literature", "status": "active"}],
    })
    session = _create_mock_session(interp_plan=plan, interactive=True)

    assert session.status == "waiting_user"
    assert session.pending_fact_key == "clarify_scope"
    assert session.last_asked_question is not None
    assert "Kaç kaynak taransın?" in session.last_asked_question
    assert "3" in session.last_asked_question
    ask_messages = [m for m in session.ctx.chat_log if m.kind == "ask_user"]
    assert len(ask_messages) == 1
    assert "Kaç kaynak taransın?" in ask_messages[0].content


def test_startup_simple_task_needs_plan_false_bypasses_planning():
    """When needs_plan=False, session plan remains None and normal ready flow holds."""
    plan = parse_task_ledger({"needs_plan": False})
    session = _create_mock_session(interp_plan=plan, interactive=False)

    assert session.ctx.plan is None
    assert session.status == "ready"


# ==============================================================================
# 3. Anti-clobbering in _handle_escalation tests
# ==============================================================================

def test_escalation_anti_clobbering_preserves_plan_as_memory():
    """_handle_escalation uses plan format as memory, never clobbering with 15-word sup.memory."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "finish_criteria": "Gather 2 items",
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Lovelace",
                "status": "active",
                "results": [{"fact": "Born 1815", "source_url": "https://wiki.org/ada", "step": 1}],
            }
        ],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    short_ephemeral_memory = "Only 5 words here"
    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="SCROLL_DOWN",
        memory=short_ephemeral_memory,
        extracted_results=[{"fact": "Died 1852", "source_url": "https://wiki.org/ada", "step": 2}],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="uncertain_action",
            trigger_type="gate_uncertain",
            s1_decision={"action": "SCROLL_DOWN", "confidence": 0.4},
        )

    # working_memory is NOT the 15-word scratchpad!
    assert session.ctx.working_memory != short_ephemeral_memory
    # working_memory is the formatted structured plan!
    assert "Plan Status: Active" in session.ctx.working_memory
    assert "Born 1815" in session.ctx.working_memory
    assert "Died 1852" in session.ctx.working_memory
    # Active subgoal accumulated results additively
    active_sg = session.ctx.plan.get_active_subgoal()
    assert len(active_sg.results) == 2
    assert active_sg.results[0].fact == "Born 1815"
    assert active_sg.results[1].fact == "Died 1852"


def test_escalation_legacy_memory_preserved_when_no_plan():
    """When no plan is active, legacy behavior sets ctx.working_memory = sup.memory."""
    session = _create_mock_session(interp_plan=None, interactive=False)

    legacy_memory = "Legacy 15-word note: inspected search results"
    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="SCROLL_DOWN",
        memory=legacy_memory,
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="uncertain_action",
            trigger_type="gate_uncertain",
            s1_decision={"action": "SCROLL_DOWN", "confidence": 0.4},
        )

    assert session.ctx.working_memory == legacy_memory


def test_escalation_rate_limits_distilled_facts_to_max_3():
    """_handle_escalation appends at most 3 facts per milestone call."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "active", "results": []}],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="WAIT",
        extracted_results=[
            {"fact": f"Fact {i}", "source_url": "https://site.com", "step": i}
            for i in range(1, 6)
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="stuck_loop",
            trigger_type="loop_detected",
            s1_decision=None,
        )

    active_sg = session.ctx.plan.get_active_subgoal()
    assert len(active_sg.results) == 3
    assert [r.fact for r in active_sg.results] == ["Fact 1", "Fact 2", "Fact 3"]


def test_escalation_rejects_raw_dom_blobs_from_results():
    """_handle_escalation rejects raw DOM/HTML strings from being added as facts."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [{"id": "sg_1", "text": "Subgoal 1", "status": "active", "results": []}],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    raw_html = "<div class='table-responsive'><table><tr><td>$450</td></tr></table></div>"
    valid_fact = "Flight price is $450 on Pegasus Airlines"
    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="WAIT",
        extracted_results=[
            {"fact": raw_html, "source_url": "https://flights.com", "step": 1},
            {"fact": valid_fact, "source_url": "https://flights.com", "step": 1},
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="check",
            trigger_type="gate_uncertain",
            s1_decision=None,
        )

    active_sg = session.ctx.plan.get_active_subgoal()
    assert len(active_sg.results) == 1
    assert active_sg.results[0].fact == valid_fact


def test_escalation_updates_candidate_table_with_attributes():
    """_handle_escalation updates candidate comparison table with attributes and deduplication."""
    plan = parse_task_ledger({
        "needs_plan": True,
        "candidates": [{"item": "Anker Q30", "source_url": "https://store1.com", "attributes": {"price": "$79"}}],
        "subgoals": [{"id": "sg_1", "text": "Compare", "status": "active"}],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)

    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="SCROLL_DOWN",
        extracted_candidates=[
            {"item": "Anker Q30", "source_url": "https://store2.com", "attributes": {"rating": "4.6"}},
            {"item": "Sony XM4", "source_url": "https://store3.com", "attributes": {"price": "$248"}},
        ],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="check",
            trigger_type="gate_uncertain",
            s1_decision=None,
        )

    assert len(session.ctx.plan.candidates) == 2
    anker = next(c for c in session.ctx.plan.candidates if c.item == "Anker Q30")
    assert anker.attributes["price"] == "$79"
    assert anker.attributes["rating"] == "4.6"
    assert "https://store2.com" in anker.source_urls

    sony = next(c for c in session.ctx.plan.candidates if c.item == "Sony XM4")
    assert sony.attributes["price"] == "$248"


def test_escalation_triggers_2500_char_summarization():
    """_handle_escalation triggers 2500-char summarization on completed subgoals while preserving URLs."""
    long_fact_1 = "Observation details " * 80  # ~1600 chars
    long_fact_2 = "Additional findings " * 80  # ~1600 chars
    plan = parse_task_ledger({
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Completed research",
                "status": "completed",
                "results": [
                    {"fact": long_fact_1, "source_url": "https://source-a.org/paper1", "step": 1},
                    {"fact": long_fact_2, "source_url": "https://source-b.org/paper2", "step": 2},
                ],
            },
            {
                "id": "sg_2",
                "text": "Current step",
                "status": "active",
                "results": [],
            },
        ],
    })
    session = _create_mock_session(interp_plan=plan, interactive=False)
    assert session.ctx.plan.total_results_chars() > 2500

    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="WAIT",
        extracted_results=[{"fact": "Brief fact for active sg", "source_url": "https://active.org", "step": 3}],
    )

    with patch("reflex.agent.resolve_escalation", return_value=sup_decision):
        session._handle_escalation(
            reason="check",
            trigger_type="gate_uncertain",
            s1_decision=None,
        )

    # Completed subgoal was summarized to under 3000 chars while preserving both URLs
    completed_sg = session.ctx.plan.subgoals[0]
    surviving_urls = {r.source_url for r in completed_sg.results}
    assert "https://source-a.org/paper1" in surviving_urls
    assert "https://source-b.org/paper2" in surviving_urls
    assert session.ctx.plan.total_results_chars() < 2500


# ==============================================================================
# 4. System 1 input isolation tests
# ==============================================================================

def test_system1_input_isolation_active_subgoal_only():
    """compile_system1_goal isolates S1 from candidate table and extracted past results."""
    plan = TaskLedger(
        needs_plan=True,
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find Ada Lovelace birth year",
                status="completed",
                results=[SubgoalResult(fact="Born 1815", source_url="https://wiki.org/ada", step=1)],
            ),
            Subgoal(
                id="sg_2",
                text="Find Charles Babbage birth year",
                status="active",
                results=[],
            ),
        ],
        candidates=[Candidate(item="Babbage Engine", source_url="https://museum.org", attributes={"built": "yes"})],
    )
    ctx = DualTierContext(
        primary_goal="Research Ada Lovelace and Charles Babbage",
        active_subgoal="Find Charles Babbage birth year",
        plan=plan,
    )

    s1_prompt = ctx.compile_system1_goal()
    assert "Research Ada Lovelace and Charles Babbage" in s1_prompt
    assert "Current subgoal: Find Charles Babbage birth year" in s1_prompt

    # Strictly excluded from System 1 prompt
    assert "Born 1815" not in s1_prompt
    assert "https://wiki.org/ada" not in s1_prompt
    assert "Babbage Engine" not in s1_prompt
    assert "Candidates Table" not in s1_prompt
    assert "Subgoals & Extracted Results" not in s1_prompt


# ==============================================================================
# 5. System 2 context injection tests
# ==============================================================================

def test_system2_context_block_injects_structured_task_ledger():
    """build_supervisor_context_block injects full structured Task Ledger for S2."""
    plan = TaskLedger(
        needs_plan=True,
        finish_criteria="All birth years verified",
        subgoals=[
            Subgoal(
                id="sg_1",
                text="Find Lovelace",
                status="completed",
                results=[SubgoalResult(fact="Born 1815", source_url="https://wiki.org/ada", step=1)],
            ),
            Subgoal(id="sg_2", text="Find Babbage", status="active"),
        ],
        candidates=[Candidate(item="Analytical Engine", source_url="https://sciencemuseum.org.uk", attributes={"type": "mechanical"})],
    )
    ctx = DualTierContext(
        primary_goal="History of Computing",
        active_subgoal="Find Babbage",
        plan=plan,
    )

    s2_block = ctx.build_supervisor_context_block([])
    assert "Structured Task Ledger:" in s2_block
    assert "Plan Status: Active" in s2_block
    assert "Finish Criteria: All birth years verified" in s2_block
    assert "[COMPLETED] sg_1: Find Lovelace" in s2_block
    assert "[ACTIVE] sg_2: Find Babbage" in s2_block
    assert "Born 1815" in s2_block
    assert "https://wiki.org/ada" in s2_block
    assert "Candidates Table:" in s2_block
    assert "Analytical Engine" in s2_block
    assert "https://sciencemuseum.org.uk" in s2_block
