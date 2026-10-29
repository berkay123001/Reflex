from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig
from reflex.context import DualTierContext
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation


def _make_fake_browser(page: dict[str, Any] | None = None) -> MagicMock:
    mock_b = MagicMock()
    if page is None:
        page = {
            "url": "https://example.com/shop",
            "title": "Shop",
            "text": "Product selection page",
            "fingerprint": "fp_shop_1",
            "actions": [
                {
                    "id": "btn_add_to_cart",
                    "node": 10,
                    "kind": "click",
                    "role": "button",
                    "label": "Add to Cart",
                    "submits_form": False,
                },
                {
                    "id": "btn_checkout",
                    "node": 11,
                    "kind": "click",
                    "role": "button",
                    "label": "Proceed to Checkout",
                    "submits_form": False,
                },
            ],
        }
    mock_b.observe.return_value = page
    mock_b.fresh.return_value = True
    return mock_b


def test_unattended_first_ask_converts_to_default_and_continues_with_override() -> None:
    """When ctx.user_present=False, the first ASK_USER does not halt, sets tactical_hint,

    status remains 'ready', and the next step proceeds to System 2 with OVERRIDE_ACTION.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(interactive=False, s2_every_step=True)

    question_text = "What size shirt do you prefer?"
    expected_hint = f"User is unavailable: choose a sensible default for: {question_text}, state it in the final answer"

    sup_ask = SupervisorDecision(
        action_type="ASK_USER",
        question_for_user=question_text,
        reasoning="Shirt size preference not specified by user",
        model="supervisor-test",
        latency_ms=10,
    )

    sup_override = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Defaulting to Medium and proceeding with add to cart",
        model="supervisor-test",
        latency_ms=12,
    )

    s2_context_blocks: list[str] = []

    def fake_resolve_escalation(**kwargs: Any) -> SupervisorDecision:
        ctx: DualTierContext = kwargs["ctx"]
        history = kwargs.get("history", [])
        s2_context_blocks.append(ctx.build_supervisor_context_block(history))
        if len(s2_context_blocks) == 1:
            return sup_ask
        return sup_override

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.resolve_escalation", side_effect=fake_resolve_escalation),
    ):
        session = ReflexSession(
            url="https://example.com/shop",
            goal="Buy a shirt",
            config=cfg,
        )
        assert session.ctx.user_present is False
        assert len(session.ctx.unattended_questions) == 0

        # Step 1: Fake S2 returns ASK_USER
        snap1 = session.step()

        # Must NOT halt in waiting_user; must remain ready
        assert session.status == "ready"
        assert snap1["status"] == "ready"
        assert session.ctx.tactical_hint == expected_hint
        assert session.pending_escalation_reason == (
            "ask_user_default",
            f"Unattended default for: {question_text}",
        )
        assert session.last_asked_question is None
        assert session.pending_fact_key is None
        assert session.ctx.unattended_questions == [question_text]
        assert session.unattended_questions == [question_text]
        assert session._unattended_questions == [question_text]

        # Verify escalation note message logged
        esc_notes = [m.content for m in session.ctx.chat_log if m.kind == "escalation_note"]
        assert any(expected_hint in note for note in esc_notes)

        # Step 2: Next step dispatches to S2 via pending_escalation_reason and executes OVERRIDE_ACTION
        snap2 = session.step()
        assert session.status == "ready"
        assert snap2["status"] == "ready"
        assert len(s2_context_blocks) == 2
        # Verify S2 received tactical hint in its context block
        assert f"Current Tactical Hint: {expected_hint}" in s2_context_blocks[1]

        # Verify action was executed
        assert len(session.history) == 1
        assert session.history[0]["actor"] == "S2"
        assert session.history[0]["operation"] == "CLICK"


def test_unattended_duplicate_question_triggers_waiting_user() -> None:
    """When ctx.user_present=False and the supervisor asks the exact same question twice

    in the session, the second occurrence transitions to 'waiting_user'.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(interactive=False, s2_every_step=True)

    question_text = "Which shipping speed do you want: Standard or Express?"

    sup_ask = SupervisorDecision(
        action_type="ASK_USER",
        question_for_user=question_text,
        reasoning="Shipping method required",
        model="supervisor-test",
        latency_ms=10,
    )

    call_count = 0

    def fake_resolve_escalation(**kwargs: Any) -> SupervisorDecision:
        nonlocal call_count
        call_count += 1
        return sup_ask

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.resolve_escalation", side_effect=fake_resolve_escalation),
    ):
        session = ReflexSession(
            url="https://example.com/checkout",
            goal="Checkout cart",
            config=cfg,
        )

        # Step 1: First time asking -> converted to default, remains ready
        session.step()
        assert session.status == "ready"
        assert len(session.ctx.unattended_questions) == 1
        assert session.pending_escalation_reason is not None

        # Step 2: Second time asking the same question -> transitions to waiting_user
        snap = session.step()
        assert session.status == "waiting_user"
        assert snap["status"] == "waiting_user"
        assert session.last_asked_question == question_text
        assert snap["asked_question"] == question_text
        assert session.pending_fact_key == "user_clarification"

        # Verify ask_user message logged
        ask_msgs = [m.content for m in session.ctx.chat_log if m.kind == "ask_user"]
        assert question_text in ask_msgs


def test_unattended_duplicate_question_case_and_whitespace_insensitive() -> None:
    """Duplicate detection is case-insensitive and trims whitespace."""
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(interactive=False, s2_every_step=True)

    q1 = "  Which color do you prefer?  "
    q2 = "which color do you prefer?"

    call_count = 0

    def fake_resolve_escalation(**kwargs: Any) -> SupervisorDecision:
        nonlocal call_count
        call_count += 1
        q = q1 if call_count == 1 else q2
        return SupervisorDecision(
            action_type="ASK_USER",
            question_for_user=q,
            reasoning="Color preference",
            model="supervisor-test",
        )

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.resolve_escalation", side_effect=fake_resolve_escalation),
    ):
        session = ReflexSession(
            url="https://example.com/item",
            goal="Pick a color",
            config=cfg,
        )

        session.step()
        assert session.status == "ready"

        session.step()
        assert session.status == "waiting_user"
        assert session.last_asked_question == q2


def test_unattended_max_two_defaults_third_distinct_question_triggers_waiting_user() -> None:
    """When ctx.user_present=False, at most 2 ASK_USER questions per session are converted

    to defaults. Even with distinct question strings, the 3rd ASK_USER transitions to waiting_user.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(interactive=False, s2_every_step=True)

    questions = [
        "Question 1: Choose size (S, M, L)",
        "Question 2: Choose gift wrapping (Yes, No)",
        "Question 3: Enter recipient delivery instructions",
    ]

    call_count = 0

    def fake_resolve_escalation(**kwargs: Any) -> SupervisorDecision:
        nonlocal call_count
        q = questions[min(call_count, len(questions) - 1)]
        call_count += 1
        return SupervisorDecision(
            action_type="ASK_USER",
            question_for_user=q,
            reasoning=f"Reason for {q}",
            model="supervisor-test",
        )

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.resolve_escalation", side_effect=fake_resolve_escalation),
    ):
        session = ReflexSession(
            url="https://example.com/order",
            goal="Complete order",
            config=cfg,
        )

        # Question 1: Converted (count = 1) -> ready
        session.step()
        assert session.status == "ready"
        assert len(session.ctx.unattended_questions) == 1
        assert session.ctx.unattended_questions[0] == questions[0]
        assert "Question 1" in session.ctx.tactical_hint

        # Question 2: Converted (count = 2) -> ready
        session.step()
        assert session.status == "ready"
        assert len(session.ctx.unattended_questions) == 2
        assert session.ctx.unattended_questions[1] == questions[1]
        assert "Question 2" in session.ctx.tactical_hint

        # Question 3: 3rd ASK_USER call in session -> must halt in waiting_user!
        snap = session.step()
        assert session.status == "waiting_user"
        assert snap["status"] == "waiting_user"
        assert session.last_asked_question == questions[2]
        assert len(session.ctx.unattended_questions) == 2  # Not converted

        ask_msgs = [m.content for m in session.ctx.chat_log if m.kind == "ask_user"]
        assert questions[2] in ask_msgs


def test_interactive_user_present_true_immediately_transitions_to_waiting_user() -> None:
    """When ctx.user_present=True, the first ASK_USER call immediately transitions to

    'waiting_user', preserving standard interactive behavior.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(interactive=True, s2_every_step=True)

    question_text = "What is your username?"

    sup_ask = SupervisorDecision(
        action_type="ASK_USER",
        question_for_user=question_text,
        reasoning="Username required for login",
        model="supervisor-test",
        latency_ms=15,
    )

    with (
        patch("reflex.agent.Browser", return_value=fake_browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.resolve_escalation", return_value=sup_ask),
    ):
        session = ReflexSession(
            url="https://example.com/login",
            goal="Log in to account",
            config=cfg,
        )
        assert session.ctx.user_present is True

        snap = session.step()

        # Must immediately halt in waiting_user on first call
        assert session.status == "waiting_user"
        assert snap["status"] == "waiting_user"
        assert session.last_asked_question == question_text
        assert session.pending_fact_key == "user_clarification"
        assert snap["asked_question"] == question_text
        assert len(session.ctx.unattended_questions) == 0

        # Tactical hint must NOT have been converted to unattended default message
        assert "User is unavailable" not in session.ctx.tactical_hint

        ask_msgs = [m.content for m in session.ctx.chat_log if m.kind == "ask_user"]
        assert question_text in ask_msgs


def test_system1_goal_compilation_with_tactical_hint() -> None:
    """Verify compile_system1_goal incorporates tactical_hint cleanly without schema disruption."""
    ctx = DualTierContext(primary_goal="Book a flight")
    ctx.tactical_hint = "User is unavailable: choose a sensible default for: seat class, state it in the final answer"

    compiled = ctx.compile_system1_goal()
    assert "Book a flight" in compiled
    assert "Note: User is unavailable: choose a sensible default for: seat class, state it in the final answer" in compiled
