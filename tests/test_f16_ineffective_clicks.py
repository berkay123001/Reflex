"""Tests for F16: Prevent repeating ineffective button clicks (`no_effect_repeat`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from reflex.agent import ReflexSession
from reflex.browser.cdp import canonical_url
from reflex.config import ReflexConfig
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation
from reflex.system1 import action_space
from tests.test_form_submit_gate import chromium_port
from tests.test_overlay_choices import _chromium_browser

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ETKISIZ_BUTON_FIXTURE = FIXTURES / "etkisiz_buton.html"
SONRAKI_QUIZ_FIXTURE = FIXTURES / "sonraki_quiz.html"


def _make_session_with_browser(browser: Any, goal: str) -> ReflexSession:
    """Create a ReflexSession bound to an already-open test `Browser` instance."""
    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
    ):
        cfg = ReflexConfig(screenshots=False, enable_site_memory=False)
        session = ReflexSession(url=browser.observe(screenshot=False)["url"], goal=goal, config=cfg)
    return session


def test_f16_dead_button_escalates_on_second_selection_and_never_clicks_third_time(
    chromium_port: int,
) -> None:
    """F16: Dead button + working link fixture. Fake S1 always selects the dead button ->
    escalates to S2 on the 2nd selection ('no_effect_repeat') with 'bu buton işe yaramadı' in S2 context,
    and the button is never clicked a 3rd time.
    """
    uri = ETKISIZ_BUTON_FIXTURE.resolve().as_uri()
    with _chromium_browser(chromium_port, uri) as browser:
        with patch.object(browser, "close"):
            session = _make_session_with_browser(browser, "Sonuç sayfasına ulaş")

            def fake_s1_always_picks_dead_button(
                page: dict[str, Any],
                goal: str,
                history: list[dict[str, Any]],
                config: ReflexConfig,
                **kwargs: Any,
            ) -> dict[str, Any]:
                _, targets, _ = action_space(page.get("actions", []))
                dead_target = next(
                    idx
                    for idx, act in targets.get("CLICK", {}).items()
                    if act.get("label") == "Bir pim kaydet"
                )
                dead_act = targets["CLICK"][dead_target]
                return {
                    "operation": "CLICK",
                    "target": dead_target,
                    "choice": dead_act["id"],
                    "confidence": 0.96,
                    "target_confidence": 0.94,
                    "target_margin": 0.88,
                    "operation_probabilities": {"CLICK": 0.96},
                    "target_probabilities": {dead_target: 0.94},
                    "latency_ms": 10,
                    "model": "fake-s1",
                    "usage": {"cost": 0.0},
                }

            captured_s2_calls: list[dict[str, Any]] = []

            def fake_s2_resolve_escalation(**kwargs: Any) -> SupervisorDecision:
                ctx = kwargs["ctx"]
                history = kwargs["history"]
                s2_block = ctx.build_supervisor_context_block(history)
                captured_s2_calls.append(
                    {
                        "trigger_type": kwargs["trigger_type"],
                        "escalation_reason": kwargs["escalation_reason"],
                        "s2_context_block": s2_block,
                    }
                )
                # On the first escalation (step 2), return UPDATE_SUBGOAL so step 3 also tests
                # that S1 selecting the dead button a 3rd time still does not click it;
                # on the second escalation (step 3), override by clicking the working link.
                if len(captured_s2_calls) == 1:
                    return SupervisorDecision(
                        action_type="UPDATE_SUBGOAL",
                        updated_subgoal="Buton çalışmıyor, linki kullan",
                        tactical_hint="Sonuç Sayfasına Git linkine tıkla",
                        reasoning="Dead button detected",
                        model="fake-s2",
                        latency_ms=12,
                    )
                _, targets, _ = action_space(kwargs["page"].get("actions", []))
                link_target = next(
                    idx
                    for idx, act in targets.get("CLICK", {}).items()
                    if act.get("label") == "Sonuç Sayfasına Git"
                )
                return SupervisorDecision(
                    action_type="OVERRIDE_ACTION",
                    operation="CLICK",
                    target=link_target,
                    reasoning="Clicking working link instead of dead button",
                    model="fake-s2",
                    latency_ms=12,
                )

            with (
                patch("reflex.agent.choose_action", side_effect=fake_s1_always_picks_dead_button),
                patch("reflex.agent.resolve_escalation", side_effect=fake_s2_resolve_escalation),
            ):
                # 1st selection: S1 clicks "Bir pim kaydet" -> no progress -> recorded in ineffective_clicks
                session.step()
                assert browser.evaluate("window.__dead_clicks || 0") == 1
                assert len(session.history) == 1
                assert session.history[0]["action"] == "Bir pim kaydet"
                assert session.history[0]["effective_progress"] is False
                assert len(session.escalations) == 0
                assert (canonical_url(uri), "Bir pim kaydet", "button") in session.ineffective_clicks

                # 2nd selection: S1 selects "Bir pim kaydet" again -> NOT executed, escalates with "no_effect_repeat"
                session.step()
                assert browser.evaluate("window.__dead_clicks || 0") == 1
                assert len(session.history) == 1  # Still only 1 click executed!
                assert len(session.escalations) == 1
                assert session.escalations[0]["trigger_type"] == "no_effect_repeat"
                assert len(captured_s2_calls) == 1
                assert captured_s2_calls[0]["trigger_type"] == "no_effect_repeat"
                assert "bu buton işe yaramadı" in captured_s2_calls[0]["s2_context_block"]
                assert "Bir pim kaydet" in captured_s2_calls[0]["s2_context_block"]

                # 3rd selection: S1 selects "Bir pim kaydet" a 3rd time -> still NOT clicked; S2 clicks working link
                session.step()
                assert int(browser.evaluate("Number(window.name || '0')")) == 1
                dead_btn_clicks = [h for h in session.history if h.get("action") == "Bir pim kaydet"]
                assert len(dead_btn_clicks) == 1  # Never clicked a 2nd or 3rd time!
                assert session.history[-1]["action"] == "Sonuç Sayfasına Git"
                assert session.history[-1]["url_changed"] is True
                # URL changed -> ineffective_clicks cleared
                assert session.ineffective_clicks == []


def test_f16_sonraki_fixture_five_clicks_not_blocked(chromium_port: int) -> None:
    """F16: 'Sonraki' button that changes page content on every click is NOT blocked across 5 clicks."""
    uri = SONRAKI_QUIZ_FIXTURE.resolve().as_uri()
    with _chromium_browser(chromium_port, uri) as browser:
        with patch.object(browser, "close"):
            session = _make_session_with_browser(browser, "Quiz sorularını ilerlet")

            def fake_s1_always_picks_sonraki(
                page: dict[str, Any],
                goal: str,
                history: list[dict[str, Any]],
                config: ReflexConfig,
                **kwargs: Any,
            ) -> dict[str, Any]:
                _, targets, _ = action_space(page.get("actions", []))
                next_target = next(
                    idx
                    for idx, act in targets.get("CLICK", {}).items()
                    if act.get("label") == "Sonraki"
                )
                next_act = targets["CLICK"][next_target]
                return {
                    "operation": "CLICK",
                    "target": next_target,
                    "choice": next_act["id"],
                    "confidence": 0.97,
                    "target_confidence": 0.95,
                    "target_margin": 0.90,
                    "operation_probabilities": {"CLICK": 0.97},
                    "target_probabilities": {next_target: 0.95},
                    "latency_ms": 8,
                    "model": "fake-s1",
                    "usage": {"cost": 0.0},
                }

            with patch("reflex.agent.choose_action", side_effect=fake_s1_always_picks_sonraki):
                for _ in range(5):
                    session.step()

            assert len(session.history) == 5
            assert all(h["action"] == "Sonraki" for h in session.history)
            assert all(h["effective_progress"] is True for h in session.history)
            assert len(session.escalations) == 0
            assert session.ineffective_clicks == []
            assert browser.evaluate("window.__quiz_step") == 6
            assert "Soru 6:" in session.page.get("text", "")


def test_f16_ineffective_clicks_cleared_when_fingerprint_meaningfully_changes() -> None:
    """F16: Ineffective click record is cleared when page fingerprint meaningfully changes on the same URL."""
    sess = ReflexSession.__new__(ReflexSession)
    from reflex.context import DualTierContext

    sess.ctx = DualTierContext(primary_goal="Test")
    sess.history = []
    sess._ineffective_url = ""
    sess._ineffective_fp = ""
    sess.page = {
        "url": "https://example.com/quiz",
        "fingerprint": "fp_initial",
        "actions": [{"id": "e1", "kind": "click", "role": "button", "label": "Hot or Cold"}],
    }

    act = sess.page["actions"][0]
    sess._record_ineffective_click(act, "https://example.com/quiz", "fp_initial")
    assert ("https://example.com/quiz", "Hot or Cold", "button") in sess.ineffective_clicks
    assert sess._is_ineffective_click(act, "https://example.com/quiz") is True

    # Meaningfully changing page fingerprint clears the ineffective click record
    sess.page = {
        "url": "https://example.com/quiz",
        "fingerprint": "fp_updated_content",
        "actions": [{"id": "e1", "kind": "click", "role": "button", "label": "Hot or Cold"}],
    }
    assert sess._is_ineffective_click(act, "https://example.com/quiz") is False
    assert sess.ineffective_clicks == []
