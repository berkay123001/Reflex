"""Home-site guard for guessed GOTO_URLs and the idle supervisor-call guard (offline)."""

from __future__ import annotations

from reflex.agent import MAX_IDLE_STEP_CALLS, _home_site_for, _site_key
from reflex.context import DualTierContext
from reflex.supervisor import SupervisorDecision
from tests.test_stuck_finish import _make_mock_session


def _goto(url: str) -> SupervisorDecision:
    return SupervisorDecision(action_type="OVERRIDE_ACTION", operation="GOTO_URL", url=url, reasoning="try")


def _session_on(url: str, goal: str = "Find theatre events in Las Vegas"):
    sess = _make_mock_session(url=url, goal=goal)
    sess.ctx.home_site = _home_site_for(url)
    return sess


def test_site_key_uses_registrable_domain() -> None:
    assert _site_key("https://webapps.akc.org/event-search/#/search") == "akc.org"
    assert _site_key("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert _site_key("https://www.liderpatronlar.com.tr/uyelerimiz") == "liderpatronlar.com.tr"
    assert _site_key("about:blank") == ""


def test_search_engine_start_has_no_home_site() -> None:
    assert _home_site_for("https://www.google.com") == ""
    assert _home_site_for("https://www.google.com/search?q=quiz") == ""
    assert _home_site_for("https://duckduckgo.com/?q=x") == ""
    assert _home_site_for("https://search.yahoo.com/search?p=x") == ""
    assert _home_site_for("https://www.stubhub.com/") == "stubhub.com"


def test_product_of_a_search_company_is_the_task_website() -> None:
    """full300 2026-11-04: tasks on Google Shopping guessed amazon/bestbuy/walmart URLs because google.com counted as open web."""
    assert _home_site_for("https://www.google.com/shopping?udm=28") == "google.com"
    assert _home_site_for("https://www.google.com/travel/flights") == "google.com"
    assert _home_site_for("https://www.google.com/finance/") == "google.com"
    assert _home_site_for("https://www.google.com/maps/") == "google.com"
    assert _home_site_for("https://finance.yahoo.com/") == "yahoo.com"


def test_guess_off_google_shopping_is_rejected() -> None:
    sess = _session_on("https://www.google.com/shopping?udm=28", goal="Look for reviews of a Nest Hello Video Doorbell")
    assert sess._resolve_supervisor_action(_goto("https://www.amazon.com/s?k=Nest+Hello"), {}, {}) is None
    assert any("amazon.com" in r for r in sess.ctx.rejected_links)


def test_guessed_other_site_is_rejected() -> None:
    sess = _session_on("https://www.stubhub.com/")
    assert sess._resolve_supervisor_action(_goto("https://www.ticketmaster.com/discover/theatre"), {}, {}) is None
    assert any("ticketmaster.com" in r for r in sess.ctx.rejected_links)
    assert "stubhub.com" in sess.ctx.tactical_hint


def test_same_site_and_subdomain_are_allowed() -> None:
    sess = _session_on("https://www.akc.org/")
    action = sess._resolve_supervisor_action(_goto("https://webapps.akc.org/event-search/"), {}, {})
    assert action is not None and action["kind"] == "goto_url"


def test_link_present_on_page_is_allowed() -> None:
    sess = _session_on("https://www.liderpatronlar.com.tr/uyelerimiz")
    sess.page["actions"] = [{"kind": "click", "label": "Firma", "href": "https://www.firma-ornek.com/"}]
    assert sess._resolve_supervisor_action(_goto("https://www.firma-ornek.com/"), {}, {}) is not None


def test_site_named_in_goal_is_allowed() -> None:
    sess = _session_on("https://www.stubhub.com/", goal="Compare StubHub and Ticketmaster prices for a show")
    assert sess._resolve_supervisor_action(_goto("https://www.ticketmaster.com/"), {}, {}) is not None


def test_site_reached_by_following_a_link_stays_allowed() -> None:
    sess = _session_on("https://www.traderjoes.com/")
    sess.ctx.visited_urls.append("https://hosted.where2getit.com/traderjoes/locator.html")
    assert sess._resolve_supervisor_action(_goto("https://hosted.where2getit.com/traderjoes/x"), {}, {}) is not None


def test_open_web_run_is_not_restricted() -> None:
    sess = _session_on("https://www.google.com")
    assert sess._resolve_supervisor_action(_goto("https://www.quizzei.com/"), {}, {}) is not None


def test_guard_off_when_home_site_empty() -> None:
    sess = _make_mock_session(url="https://www.stubhub.com/")
    assert sess.ctx.home_site == ""
    assert sess._resolve_supervisor_action(_goto("https://www.ticketmaster.com/"), {}, {}) is not None


def test_context_block_names_task_website() -> None:
    ctx = DualTierContext(primary_goal="g", active_subgoal="g", home_site="stubhub.com")
    assert "Task Website: stubhub.com" in ctx.build_supervisor_context_block([])
    open_ctx = DualTierContext(primary_goal="g", active_subgoal="g")
    assert "Task Website" not in open_ctx.build_supervisor_context_block([])


def _idle_session():
    sess = _make_mock_session()
    sess._idle_step_calls = 0
    sess._step_once = lambda: {"status": sess.status}
    sess._build_honest_failure_summary = lambda cause=None: f"honest summary: {cause}"
    return sess


def test_idle_calls_trigger_stuck_finish_then_honest_end() -> None:
    sess = _idle_session()
    for _ in range(MAX_IDLE_STEP_CALLS - 1):
        sess.step()
    assert sess.pending_escalation_reason is None
    sess.step()
    assert sess.pending_escalation_reason[0] == "stuck_finish"
    assert sess.status == "ready"
    for _ in range(MAX_IDLE_STEP_CALLS):
        sess.step()
    assert sess.status == "done"
    assert sess.last_error.startswith("honest summary: The last")


def test_recorded_step_resets_idle_counter() -> None:
    sess = _idle_session()
    for _ in range(MAX_IDLE_STEP_CALLS - 1):
        sess.step()

    def _recorded():
        sess.history.append({"step": len(sess.history) + 1})
        return {"status": sess.status}

    sess._step_once = _recorded
    sess.step()
    assert sess._idle_step_calls == 0
    assert sess.pending_escalation_reason is None


def test_site_linked_from_an_earlier_page_is_allowed() -> None:
    sess = _session_on("https://www.liderpatronlar.com.tr/uyelerimiz")
    sess._remember_linked_sites({"actions": [], "page_links": [{"href": "https://prestij.com.tr/", "label": "Prestij"}]})
    sess.page = {"url": "https://www.muratgold.com/", "actions": []}
    sess.ctx.visited_urls.append("https://www.muratgold.com/")
    assert sess._resolve_supervisor_action(_goto("https://prestij.com.tr/"), {}, {}) is not None
    assert sess._resolve_supervisor_action(_goto("https://www.unlinked-example.com/"), {}, {}) is None


def test_covered_target_for_s2_is_reported_back() -> None:
    from reflex.browser.cdp import TargetCoveredError

    sess = _make_mock_session()
    sess.browser.act.side_effect = TargetCoveredError("Target 'Search' is covered by an overlay/modal or offscreen.")
    sess._execute_and_verify(
        action={"id": "27", "kind": "fill", "label": "Search", "node": 27},
        actor="S2", operation="TYPE_TEXT", target="27", confidence=1.0, target_confidence=None,
        joint_confidence=1.0, target_margin=0.0, latency_ms=0, model_name="m", usage={},
        preset_text="fever", escalation_note=None,
    )
    trigger, reason = sess.pending_escalation_reason
    assert trigger == "target_covered"
    assert "TYPE_TEXT" in reason and "Search" in reason
    assert sess.status == "ready"


def test_unattended_noop_escalation_reports_rejected_link_instead_of_waiting_user() -> None:
    from unittest.mock import patch

    sess = _session_on("https://www.stubhub.com/")
    sess.ctx.user_present = False
    sess.page["fingerprint"] = "fp_stubhub"
    bad_goto = _goto("https://www.ticketmaster.com/discover/theatre")

    with patch("reflex.agent.resolve_escalation", return_value=bad_goto):
        sess._handle_escalation(trigger_type="s2_every_step", reason="step 1", s1_decision=None)
        assert sess.status == "ready"
        assert sess.pending_escalation_reason is None

        sess._handle_escalation(trigger_type="s2_every_step", reason="step 2", s1_decision=None)
        assert sess.status == "ready"
        assert sess.pending_fact_key is None
        assert sess.last_asked_question is None
        assert sess.pending_escalation_reason is not None
        trig, reason = sess.pending_escalation_reason
        assert trig == "action_rejected"
        assert sess.ctx.rejected_links[-1] in reason


def test_unattended_noop_escalation_reports_target_not_found_and_respects_user_present() -> None:
    from unittest.mock import patch

    # 1. Unattended OVERRIDE_ACTION with missing target -> target not found, stays ready
    sess = _make_mock_session()
    sess.ctx.user_present = False
    sess.page["fingerprint"] = "fp_1"
    bad_click = SupervisorDecision(action_type="OVERRIDE_ACTION", operation="CLICK", target="99", reasoning="click 99")

    with patch("reflex.agent.resolve_escalation", return_value=bad_click):
        sess._handle_escalation(trigger_type="s2_every_step", reason="step 1", s1_decision=None)
        sess._handle_escalation(trigger_type="s2_every_step", reason="step 2", s1_decision=None)
        assert sess.status == "ready"
        assert sess.pending_escalation_reason is not None
        assert "target not found" in sess.pending_escalation_reason[1]

    # 2. Unattended UPDATE_SUBGOAL twice -> stays ready and sets pending_escalation_reason
    sess_sub = _make_mock_session()
    sess_sub.ctx.user_present = False
    sess_sub.page["fingerprint"] = "fp_2"
    upd = SupervisorDecision(action_type="UPDATE_SUBGOAL", updated_subgoal="new sg", reasoning="replan")

    with patch("reflex.agent.resolve_escalation", return_value=upd):
        sess_sub._handle_escalation(trigger_type="low_op_conf", reason="step 1", s1_decision=None)
        sess_sub._handle_escalation(trigger_type="low_op_conf", reason="step 2", s1_decision=None)
        assert sess_sub.status == "ready"
        assert sess_sub.pending_escalation_reason is not None
        assert "target not found" in sess_sub.pending_escalation_reason[1]

    # 3. Interactive (user_present=True) -> still transitions to waiting_user
    sess_int = _make_mock_session()
    sess_int.ctx.user_present = True
    sess_int.page["fingerprint"] = "fp_3"
    with patch("reflex.agent.resolve_escalation", return_value=bad_click):
        sess_int._handle_escalation(trigger_type="s2_every_step", reason="step 1", s1_decision=None)
        sess_int._handle_escalation(trigger_type="s2_every_step", reason="step 2", s1_decision=None)
        assert sess_int.status == "waiting_user"
        assert sess_int.pending_fact_key == "user_guidance"


def _run_offline_slider_act(
    *,
    now: int,
    min_v: int,
    max_v: int,
    target_text: str,
    step_delta: int,
) -> tuple[int, list[str], list[str]]:
    from reflex.browser.cdp import MARKER, Browser

    node = 12
    guard = [node, "slider", "Age", None, None, None, None, False, None, None, None, None, None, ""]
    action = {"id": "e1", "node": node, "kind": "fill", "role": "slider", "label": "Age"}
    page = {
        "url": "https://example.test/calc",
        "marker": "m1",
        "page_key": ["o1", "https://example.test/calc"],
        "guards": {str(node): guard, node: guard},
        "actions": [action],
    }
    hit = {"x": 50, "y": 20, "rx": 10, "ry": 10, "rw": 80, "rh": 20, "jsClicked": False, "prevOrigin": 1}
    state = {"phase": 0, "val": now}
    eval_scripts: list[str] = []
    key_downs: list[str] = []

    def evaluate(expr: str):
        if expr == MARKER:
            return "m1"
        eval_scripts.append(expr)
        state["phase"] += 1
        p = state["phase"]
        if p == 1:
            return [page["page_key"], guard]
        if p == 2:
            return dict(hit)
        if p == 3:
            return None  # clear_js
        if p == 4:
            return {"now": str(state["val"]), "min": str(min_v), "max": str(max_v)}
        if "getAttribute('aria-valuenow')" in expr:
            return str(state["val"])
        return False

    def call(method: str, **params):
        if method == "Input.dispatchKeyEvent" and params.get("type") == "keyDown":
            k = params["key"]
            key_downs.append(k)
            if k == "ArrowRight":
                state["val"] = min(max_v, state["val"] + step_delta)
            elif k == "ArrowLeft":
                state["val"] = max(min_v, state["val"] - step_delta)
        return {}

    b = Browser.__new__(Browser)
    b.after_input = None
    b.last_action_rect = None
    b.evaluate = evaluate  # type: ignore[method-assign]
    b.call = call  # type: ignore[method-assign]
    b.act(action, page, text=target_text)
    return state["val"], key_downs, eval_scripts


def test_aria_slider_dispatches_arrow_keys_and_stops_when_stuck_or_capped() -> None:
    # 1. Step up with ArrowRight
    val_up, keys_up, scripts_up = _run_offline_slider_act(now=30, min_v=18, max_v=100, target_text="35", step_delta=1)
    assert val_up == 35
    assert keys_up == ["ArrowRight"] * 5
    assert not any("setAttribute('aria-valuenow'" in s for s in scripts_up)

    # 2. Step down with ArrowLeft
    val_dn, keys_dn, _ = _run_offline_slider_act(now=65, min_v=18, max_v=100, target_text="62", step_delta=1)
    assert val_dn == 62
    assert keys_dn == ["ArrowLeft"] * 3

    # 3. Value does not change -> stops after 1st key press
    val_st, keys_st, _ = _run_offline_slider_act(now=40, min_v=0, max_v=100, target_text="80", step_delta=0)
    assert val_st == 40
    assert keys_st == ["ArrowRight"]

    # 4. Large distance -> capped at 200 presses
    val_cap, keys_cap, _ = _run_offline_slider_act(now=0, min_v=0, max_v=1000, target_text="500", step_delta=1)
    assert val_cap == 200
    assert len(keys_cap) == 200

