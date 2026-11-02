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
    assert _home_site_for("https://www.google.com/travel/flights") == ""
    assert _home_site_for("https://duckduckgo.com/?q=x") == ""
    assert _home_site_for("https://www.stubhub.com/") == "stubhub.com"


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
