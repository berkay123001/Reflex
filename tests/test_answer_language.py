"""Unit tests for Milestone 4 / Requirement R3: Answer Language & Internal Neutrality.

Verifies:
1. Supervisor prompt construction with English Primary Goal instructs matching English,
   and no Turkish phrases leak into S2 context or bias S2 response.
2. Supervisor prompt construction with Turkish Primary Goal instructs matching Turkish.
3. Neutral internal context formatting: verify context notes, failure notes, and planner headers
   do not contain Turkish strings ("bu buton işe yaramadı", etc.).
4. Simulated S2 response parser/validation where reasoning and summary match goal language.
5. All tests run completely offline without external network or LLM calls in <1s.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflex.agent import ReflexAgent
from reflex.config import ReflexConfig
from reflex.context import DualTierContext
from reflex.planner import TaskLedger, parse_task_ledger
from reflex.supervisor import (
    SupervisorDecision,
    get_escalation_system_prompt,
    parse_supervisor_decision,
    resolve_escalation,
)

# ==============================================================================
# Helper Fixtures & Mock Agent
# ==============================================================================


def _create_mock_agent(
    goal: str = "Test Primary Goal",
    plan: TaskLedger | None = None,
    initial_url: str = "https://www.google.com/search?q=test",
) -> ReflexAgent:
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
        agent = ReflexAgent(url=initial_url, goal=goal, config=config)
        agent.browser = mock_browser
        return agent


FORBIDDEN_TURKISH_INTERNAL_PHRASES = [
    "bu buton işe yaramadı",
    "işe yaramadı",
]


# ==============================================================================
# 1. Supervisor Prompt Construction: English Primary Goal
# ==============================================================================


def test_supervisor_prompt_construction_english_goal_instructs_english():
    """Supervisor prompt for an English goal must instruct answering in the user's language (English) without Turkish leakage."""
    english_goal = "Find the earliest available cholesterol test appointment on Healthline"
    ctx = DualTierContext(primary_goal=english_goal)
    ctx.ineffective_clicks.append(("btn_1", "Submit Appointment", "button"))

    history = [
        {"step": 1, "actor": "S1", "operation": "CLICK", "action": "Submit Appointment", "effective_progress": False}
    ]

    s2_block = ctx.build_supervisor_context_block(history)

    # 1. Primary goal is reflected in context
    assert f"Primary Goal: {english_goal}" in s2_block

    # 2. Ineffective clicks block must be neutral English without "bu buton işe yaramadı"
    assert "Ineffective Clicks on Current Page" in s2_block
    for phrase in FORBIDDEN_TURKISH_INTERNAL_PHRASES:
        assert phrase not in s2_block, f"Found forbidden Turkish phrase '{phrase}' in S2 context block"

    # 3. System prompt instructs answering in user's language
    sys_prompt = get_escalation_system_prompt(package_mode=False)
    assert "user's language" in sys_prompt.lower() or "language of the primary goal" in sys_prompt.lower()
    assert "summary_for_user" in sys_prompt
    assert "reasoning" in sys_prompt


def test_supervisor_prompt_construction_english_goal_unattended_mode_neutral():
    """Unattended mode note in English goal context must be neutral and contain zero Turkish text."""
    ctx = DualTierContext(
        primary_goal="Compare flight prices from New York to London",
        user_present=False,
    )
    s2_block = ctx.build_supervisor_context_block([])

    assert "User Availability: the user is NOT available during this run" in s2_block
    for turkish_word in ["kullanıcı", "oturum", "gözetimsiz", "yok", "bulunamadı"]:
        assert turkish_word not in s2_block.lower()


def test_resolve_escalation_constructs_neutral_user_prompt():
    """resolve_escalation must build a user prompt free of hardcoded Turkish phrases when goal is in English."""
    english_goal = "Find traditional IRA calculator on Chase website"
    ctx = DualTierContext(primary_goal=english_goal)
    ctx.ineffective_clicks.append(("4", "Calculate Button", "button"))

    captured_prompt = {}

    def fake_chat_comp(model, system_prompt, user_prompt, **kwargs):
        captured_prompt["system_prompt"] = system_prompt
        captured_prompt["user_prompt"] = user_prompt
        dummy_decision = {
            "action_type": "COMPLETE_TASK",
            "reasoning": "Located the calculator.",
            "summary_for_user": "IRA calculator is available on Chase.",
        }
        return dummy_decision, 50, 0.001, model

    with patch("reflex.supervisor._chat_completion", side_effect=fake_chat_comp):
        decision = resolve_escalation(
            ctx=ctx,
            page={"url": "https://www.chase.com", "title": "Chase IRA", "actions": []},
            elements=[],
            history=[],
            escalation_reason="Target 'Calculate Button' (button) had no effect on this page",
            trigger_type="no_effect_repeat",
            s1_decision=None,
            supervisor_model="muse-spark-1.3-contributor",
        )

        assert decision.action_type == "COMPLETE_TASK"
        user_p = captured_prompt.get("user_prompt", "")
        assert f"Primary Goal: {english_goal}" in user_p
        for phrase in FORBIDDEN_TURKISH_INTERNAL_PHRASES:
            assert phrase not in user_p, f"Found '{phrase}' in escalation user prompt"


# ==============================================================================
# 2. Supervisor Prompt Construction: Turkish Primary Goal
# ==============================================================================


def test_supervisor_prompt_construction_turkish_goal_instructs_matching_turkish():
    """Supervisor prompt construction for a Turkish goal conveys the Turkish goal without hardcoded language conflict."""
    turkish_goal = "Chase sitesinde geleneksel IRA hesaplayıcıyı bul ve 65 yaş için hesapla"
    ctx = DualTierContext(primary_goal=turkish_goal)

    s2_block = ctx.build_supervisor_context_block([])
    assert f"Primary Goal: {turkish_goal}" in s2_block

    sys_prompt = get_escalation_system_prompt()
    # Schema requires user's language for both reasoning and summary
    assert "user's language" in sys_prompt.lower() or "language of the primary goal" in sys_prompt.lower()


def test_supervisor_prompt_dynamic_language_support_not_rigid_whitelist():
    """Language matching principle works dynamically for non-Turkish/non-English goals as well."""
    spanish_goal = "Buscar calculadora de jubilación tradicional en el sitio de Chase"
    ctx = DualTierContext(primary_goal=spanish_goal)
    s2_block = ctx.build_supervisor_context_block([])
    assert f"Primary Goal: {spanish_goal}" in s2_block


# ==============================================================================
# 3. Neutral Internal Context Formatting
# ==============================================================================


def test_neutral_internal_context_ineffective_clicks_no_turkish():
    """ctx.build_supervisor_context_block must format ineffective_clicks neutrally without Turkish text."""
    ctx = DualTierContext(primary_goal="Book appointment")
    ctx.ineffective_clicks.append(("elem_42", "Next Step", "button"))
    ctx.ineffective_clicks.append(("elem_99", "Submit", "input"))

    block = ctx.build_supervisor_context_block([])

    assert "Ineffective Clicks on Current Page" in block
    assert "Next Step (button)" in block
    assert "Submit (input)" in block
    assert "bu buton işe yaramadı" not in block


def test_agent_ineffective_clicks_escalation_reason_is_neutral():
    """When ReflexAgent encounters an ineffective click loop, escalation reason does not include Turkish."""
    agent = _create_mock_agent(goal="Complete checkout")
    page_data = {
        "url": "https://store.com/cart",
        "title": "Cart",
        "text": "Items",
        "actions": [{"id": "btn_pay", "kind": "click", "role": "button", "label": "Pay Now", "node": 10}],
        "fingerprint": "fp_cart",
    }
    agent.browser.observe.return_value = page_data
    agent.page = page_data

    # Simulate ineffective click recorded
    agent._record_ineffective_click(
        {"id": "btn_pay", "kind": "click", "role": "button", "label": "Pay Now"},
        current_url="https://store.com/cart",
        page_fp="fp_cart",
    )

    s1_click = {
        "operation": "CLICK",
        "target": "btn_pay",
        "choice": "btn_pay",
        "confidence": 0.90,
        "target_confidence": 0.90,
        "latency_ms": 10,
        "model": "turnike",
        "usage": {},
    }

    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="SCROLL_DOWN",
        reasoning="Button ineffective, scrolling down.",
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_click),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision) as mock_s2,
    ):
        agent.step()
        assert mock_s2.called
        kwargs = mock_s2.call_args.kwargs
        reason = kwargs.get("escalation_reason", "")
        assert "bu buton işe yaramadı" not in reason
        assert "Target 'Pay Now' (button) had no effect on this page" in reason


def test_planner_synthesize_final_answer_adapts_to_goal_language():
    """TaskLedger.synthesize_final_answer uses English headers for English goals and Turkish for Turkish goals."""
    raw = {
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Find Ada Lovelace birth year",
                "status": "completed",
                "results": [{"fact": "Born in 1815", "source_url": "https://en.wikipedia.org/wiki/Ada", "step": 1}],
            }
        ],
        "candidates": [
            {
                "item": "Ada Lovelace",
                "source_url": "https://en.wikipedia.org/wiki/Ada",
                "attributes": {"year": "1815"},
            }
        ],
    }
    ledger = parse_task_ledger(raw)

    # 1. English goal
    english_res = ledger.synthesize_final_answer("Find Ada Lovelace birth year")
    # Must contain English labels or neutral headers
    assert "1815" in english_res
    assert "https://en.wikipedia.org/wiki/Ada" in english_res
    # Should not contain Turkish hardcoded labels
    assert "Görev Sonucu:" not in english_res
    assert "Doğrulanan Bulgular:" not in english_res

    # 2. Turkish goal
    turkish_res = ledger.synthesize_final_answer("Ada Lovelace doğum yılını bul")
    assert "1815" in turkish_res
    assert "Görev Sonucu: Ada Lovelace doğum yılını bul" in turkish_res
    assert "Doğrulanan Bulgular:" in turkish_res


def test_full_context_block_english_goal_contains_zero_turkish_tokens():
    """A fully populated DualTierContext with an English goal contains no Turkish system strings."""
    ctx = DualTierContext(
        primary_goal="Investigate quantum computing milestones",
        active_subgoal="Find 2026 quantum supremacy papers",
        tactical_hint="Focus on arXiv search results",
        working_memory="Inspected IBM quantum blog, looking for Nature article",
        user_present=False,
    )
    ctx.user_facts["topic"] = "quantum"
    ctx.visited_urls.append("https://ibm.com/quantum")
    ctx.rejected_links.append("https://spam.com/ad (ad)")
    ctx.ineffective_clicks.append(("btn_search", "Submit Search", "button"))
    ctx.record_dom_fact("paper_count", "12 papers found", "https://arxiv.org", 2)

    history = [
        {"step": 1, "actor": "S1", "operation": "GOTO_URL", "action": "https://arxiv.org", "effective_progress": True}
    ]

    s2_block = ctx.build_supervisor_context_block(history)

    # Specific Turkish leakage markers to forbid
    turkish_forbidden = [
        "bu buton işe yaramadı",
        "işe yaramadı",
        "görev sonucu",
        "doğrulanan bulgular",
        "bulgu kaydedilmedi",
        "detay yok",
    ]
    for pattern in turkish_forbidden:
        assert pattern not in s2_block.lower(), f"Unexpected Turkish pattern '{pattern}' in S2 block"


# ==============================================================================
# 4. Simulated S2 Response Parser & Validation
# ==============================================================================


def test_parse_supervisor_decision_preserves_english_reasoning_and_summary():
    """parse_supervisor_decision cleanly parses English reasoning and summary."""
    raw_decision = {
        "action_type": "COMPLETE_TASK",
        "evaluation_previous_goal": "Found requested test page",
        "memory": "Test is no longer offered on Healthline",
        "next_goal": "Report findings to user",
        "reasoning": "The requested cholesterol lab test is obsolete and no longer offered on Healthline.",
        "summary_for_user": "We searched Healthline comprehensively: this lab test is discontinued.",
    }

    decision = parse_supervisor_decision(raw_decision)
    assert decision.action_type == "COMPLETE_TASK"
    assert decision.reasoning == "The requested cholesterol lab test is obsolete and no longer offered on Healthline."
    assert decision.summary_for_user == "We searched Healthline comprehensively: this lab test is discontinued."


def test_parse_supervisor_decision_preserves_turkish_reasoning_and_summary():
    """parse_supervisor_decision cleanly parses Turkish reasoning and summary."""
    raw_decision = {
        "action_type": "COMPLETE_TASK",
        "evaluation_previous_goal": "İstenen sayfaya ulaşıldı",
        "memory": "Test artık sitede sunulmuyor",
        "next_goal": "Kullanıcıya bildir",
        "reasoning": "İstenen kolesterol testi siteden kaldırılmıştır.",
        "summary_for_user": "Healthline üzerinde yapılan aramada testin artık sunulmadığı tespit edildi.",
    }

    decision = parse_supervisor_decision(raw_decision)
    assert decision.action_type == "COMPLETE_TASK"
    assert decision.reasoning == "İstenen kolesterol testi siteden kaldırılmıştır."
    assert decision.summary_for_user == "Healthline üzerinde yapılan aramada testin artık sunulmadığı tespit edildi."


def test_agent_complete_task_preserves_english_goal_response():
    """When S2 provides COMPLETE_TASK for an English goal, agent records the English summary without corruption."""
    english_goal = "Find the latest release version of Python"
    agent = _create_mock_agent(goal=english_goal)

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

    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="The latest Python version 3.14 was found on python.org.",
        summary_for_user="The latest release version is Python 3.14.",
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()
        assert snap["status"] == "done"
        assert agent.status == "done"

        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert len(chat_msgs) > 0
        final_msg = chat_msgs[-1]["message"]
        assert final_msg == "The latest release version is Python 3.14."


def test_agent_complete_task_fallback_synthesis_respects_english_goal():
    """When S2 completes task without summary_for_user, synthesized plan fallback uses English for English goals."""
    raw = {
        "needs_plan": True,
        "subgoals": [
            {
                "id": "sg_1",
                "text": "Identify primary author of the paper",
                "status": "completed",
                "results": [{"fact": "Turing is the author", "source_url": "https://example.com/paper", "step": 1}],
            }
        ],
    }
    plan = parse_task_ledger(raw)
    agent = _create_mock_agent(goal="Identify primary author of the paper", plan=plan)

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

    # S2 returns COMPLETE_TASK with summary_for_user=None, triggering ledger.synthesize_final_answer
    sup_decision = SupervisorDecision(
        action_type="COMPLETE_TASK",
        reasoning="Task verified complete",
        summary_for_user=None,
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.resolve_escalation", return_value=sup_decision),
    ):
        snap = agent.step()
        assert snap["status"] == "done"

        chat_msgs = [m for m in agent.ctx.messages if m.get("kind") == "chat"]
        assert len(chat_msgs) > 0
        final_msg = chat_msgs[-1]["message"]
        assert "Turing is the author" in final_msg
        assert "Görev Sonucu:" not in final_msg
