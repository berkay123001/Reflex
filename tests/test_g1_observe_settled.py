"""Tests for Milestone G1: Graceful page navigation handling via _observe_settled."""

from __future__ import annotations

import http.server
import inspect
import threading
import time
import urllib.request
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexSession, _observe_settled
from reflex.browser import StalePage
from reflex.config import ReflexConfig
from reflex.supervisor import UserMessageInterpretation


def _make_session(
    browser: Any,
    config: ReflexConfig | None = None,
    deadline_s: float | None = None,
) -> ReflexSession:
    cfg = config or ReflexConfig(screenshots=False)
    initial_page = {"url": "https://example.com/test", "actions": [], "marker": "init_marker"}

    # Temporarily ensure observe succeeds during ReflexSession.__init__
    saved_side_effect = getattr(getattr(browser, "observe", None), "side_effect", None)
    if saved_side_effect is not None:
        browser.observe.side_effect = None
        browser.observe.return_value = initial_page
    elif hasattr(browser, "observe") and isinstance(browser.observe, MagicMock) and not browser.observe.return_value:
        browser.observe.return_value = initial_page

    with (
        patch("reflex.agent.Browser") as mock_b_cls,
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_b_cls.return_value = browser
        interp_res = MagicMock(spec=UserMessageInterpretation)
        interp_res.allow_arbitrary_choices = False
        interp_res.constraints = []
        interp_res.plan = None
        interp_res.target_url = None
        interp_res.direct_reply = None
        interp_res.new_goal = None
        interp_res.updated_subgoal = None
        interp_res.tactical_hint = None
        interp_res.model_dump.return_value = {}
        mock_interp.return_value = interp_res

        session = ReflexSession(
            url="https://example.com/test",
            goal="Test observe settled",
            config=cfg,
        )
        session.browser = browser
        if deadline_s is not None:
            session._observe_deadline_s = deadline_s

    if saved_side_effect is not None:
        browser.observe.side_effect = saved_side_effect

    return session


def test_observe_settled_signature_and_default_deadline():
    """_observe_settled must accept deadline_s parameter with default value 8."""
    sig = inspect.signature(ReflexSession._observe_settled)
    assert "deadline_s" in sig.parameters
    default = sig.parameters["deadline_s"].default
    assert default == 8 or default == 8.0

    sig_mod = inspect.signature(_observe_settled)
    assert "deadline_s" in sig_mod.parameters
    assert sig_mod.parameters["deadline_s"].default == 8 or sig_mod.parameters["deadline_s"].default == 8.0


def test_observe_settled_immediate_settle():
    """When browser settles immediately, _observe_settled returns page without delay."""
    browser = MagicMock()
    page_data = {"url": "https://example.com", "actions": [], "marker": "m1"}
    browser.observe.return_value = page_data
    browser.fresh.return_value = True

    session = _make_session(browser)
    browser.observe.reset_mock()
    t0 = time.monotonic()
    res = session._observe_settled(deadline_s=2.0)
    dur = time.monotonic() - t0

    assert res == page_data
    assert dur < 0.1
    browser.observe.assert_called_once()


def test_observe_settled_recovers_after_stale():
    """When browser is navigating briefly, _observe_settled polls until settled."""
    browser = MagicMock()
    page_data = {"url": "https://example.com/ready", "actions": [], "marker": "m1"}
    browser.observe.side_effect = [
        StalePage("Document is navigating"),
        StalePage("Document is navigating"),
        page_data,
    ]
    browser.fresh.return_value = True

    session = _make_session(browser)
    t0 = time.monotonic()
    res = session._observe_settled(deadline_s=2.0)
    dur = time.monotonic() - t0

    assert res == page_data
    assert dur < 1.0


def test_observe_settled_configurable_deadline_and_timeout():
    """When deadline expires without settling: returns None, status='ready', escalation note added."""
    browser = MagicMock()
    browser.observe.side_effect = StalePage("Document is navigating")
    browser.fresh.return_value = False

    session = _make_session(browser)
    t0 = time.monotonic()
    res = session._observe_settled(deadline_s=0.2)
    dur = time.monotonic() - t0

    assert res is None
    assert dur >= 0.18
    assert dur < 0.8
    assert session.status == "ready"
    notes = [m["content"] for m in session.ctx.messages if m.get("kind") == "escalation_note"]
    assert "page_unsettled" in notes


def test_module_level_observe_settled_forwarder():
    """Module-level _observe_settled supports both ReflexSession and direct browser targets."""
    browser = MagicMock()
    page_data = {"url": "https://example.com/ok", "actions": []}
    browser.observe.return_value = page_data
    session = _make_session(browser)

    res = _observe_settled(session, deadline_s=1.0)
    assert res == page_data

    browser2 = MagicMock(spec=["observe", "evaluate"])
    browser2.observe.return_value = page_data
    res2 = _observe_settled(browser2, deadline_s=1.0)
    assert res2 == page_data


def test_invocation_point_1_step_start_unsettled_and_subsequent_recovery():
    """Location 1 (Step start): timeout records note and subsequent step recovers."""
    browser = MagicMock()
    init_page = {"url": "https://example.com/init", "actions": [], "marker": "m0"}
    browser.observe.return_value = init_page
    browser.fresh.return_value = False

    session = _make_session(browser, deadline_s=0.2)

    # Step 1: navigation in flight, observe and fresh both raise StalePage
    browser.observe.side_effect = StalePage("Document is navigating")
    browser.fresh.side_effect = StalePage("Document is navigating")

    snap1 = session.step()
    assert snap1["status"] == "ready"
    assert session.status == "ready"
    notes1 = [m["content"] for m in session.ctx.messages if m.get("kind") == "escalation_note"]
    assert "page_unsettled" in notes1

    # Step 2: page finishes loading and settles; step 2 recovers and acts
    settled_page = {
        "url": "https://example.com/settled",
        "actions": [{"id": "1", "kind": "click", "label": "Continue", "node": 10}],
        "marker": "m1",
        "fingerprint": "fp_settled",
    }
    browser.fresh.side_effect = None
    browser.fresh.return_value = False
    browser.observe.side_effect = None
    browser.observe.return_value = settled_page
    browser.act.return_value = {"rect": [10, 10, 50, 20]}

    s1_decision = {
        "operation": "CLICK",
        "target": "1",
        "choice": "1",
        "confidence": 0.95,
        "target_confidence": 0.95,
        "target_probabilities": {"1": 0.95},
        "latency_ms": 10,
        "model": "test",
        "usage": {},
    }
    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.verify_step") as mock_verify,
    ):
        mock_verify.return_value = MagicMock(
            effective_progress=True,
            page_changed=True,
            url_changed=False,
            element_state_changed=False,
            loop_detected=False,
            oscillation_detected=False,
            model_dump=lambda: {"effective_progress": True},
        )
        snap2 = session.step()

    assert snap2["status"] == "ready"
    assert len(session.history) == 1
    assert session.history[0]["action"] == "Continue"


def test_invocation_point_2_post_action_unsettled():
    """Location 2 (Post-action observation): StalePage after action times out gracefully."""
    browser = MagicMock()
    init_page = {
        "url": "https://example.com/start",
        "actions": [{"id": "1", "kind": "click", "label": "Submit", "node": 5}],
        "marker": "m0",
        "fingerprint": "fp0",
    }
    browser.observe.return_value = init_page
    browser.fresh.return_value = True

    session = _make_session(browser, deadline_s=0.2)
    session.page = init_page

    # Act succeeds, but post-action observe raises StalePage and times out
    browser.act.return_value = {"rect": [10, 10, 20, 20]}
    browser.observe.side_effect = StalePage("Document is navigating")

    s1_decision = {
        "operation": "CLICK",
        "target": "1",
        "choice": "1",
        "confidence": 0.9,
        "target_confidence": 0.9,
        "target_probabilities": {"1": 0.9},
        "latency_ms": 10,
        "model": "test",
        "usage": {},
    }
    with patch("reflex.agent.choose_action", return_value=s1_decision):
        snap = session.step()

    assert snap["status"] == "ready"
    assert session.status == "ready"
    notes = [m["content"] for m in session.ctx.messages if m.get("kind") == "escalation_note"]
    assert "page_unsettled" in notes


def test_invocation_point_3_stale_retry_unsettled():
    """Location 3 (Stale retry): StalePage during act retries observe and times out gracefully."""
    browser = MagicMock()
    init_page = {
        "url": "https://example.com/form",
        "actions": [{"id": "1", "kind": "click", "label": "Save", "node": 7}],
        "marker": "m0",
        "fingerprint": "fp0",
    }
    browser.observe.return_value = init_page
    browser.fresh.return_value = True

    session = _make_session(browser, deadline_s=0.2)
    session.page = init_page

    # browser.act raises StalePage, and stale retry observe also raises StalePage
    browser.act.side_effect = StalePage("DOM detached during click")
    browser.observe.side_effect = StalePage("Document is navigating")

    s1_decision = {
        "operation": "CLICK",
        "target": "1",
        "choice": "1",
        "confidence": 0.9,
        "target_confidence": 0.9,
        "target_probabilities": {"1": 0.9},
        "latency_ms": 10,
        "model": "test",
        "usage": {},
    }
    with patch("reflex.agent.choose_action", return_value=s1_decision):
        snap = session.step()

    assert snap["status"] == "ready"
    assert session.status == "ready"
    notes = [m["content"] for m in session.ctx.messages if m.get("kind") == "escalation_note"]
    assert "page_unsettled" in notes


@pytest.fixture
def http_server():
    """Lightweight local http.server fixture to test delayed page responses."""
    class Handler(http.server.BaseHTTPRequestHandler):
        state = {"delayed_requested": 0}

        def do_GET(self):
            if self.path == "/delayed":
                Handler.state["delayed_requested"] += 1
                if Handler.state["delayed_requested"] == 1:
                    time.sleep(0.5)
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Delayed Page Ready</h1><button>Next</button></body></html>")
            elif self.path == "/settled":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Settled</h1></body></html>")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://{host}:{port}"
    server.shutdown()


def test_delayed_page_load_local_http_server(http_server: str):
    """Integration test with local http.server fixture: small deadline_s, short delay, no crash, recovers on next step."""
    delayed_url = f"{http_server}/delayed"

    browser = MagicMock()
    init_page = {"url": delayed_url, "actions": [], "marker": "m0"}
    browser.observe.return_value = init_page
    browser.fresh.return_value = False

    session = _make_session(browser, deadline_s=0.3)

    # Step 1: querying the delayed endpoint on http_server simulates navigation in flight
    fetch_started = threading.Event()
    is_ready = threading.Event()

    def async_fetch():
        fetch_started.set()
        try:
            with urllib.request.urlopen(delayed_url, timeout=2.0) as resp:
                if resp.status == 200:
                    is_ready.set()
        except Exception:
            pass

    fetch_thread = threading.Thread(target=async_fetch, daemon=True)
    fetch_thread.start()
    fetch_started.wait(timeout=1.0)

    # While HTTP server is delaying, browser observe raises StalePage
    def dynamic_observe(*args, **kwargs):
        if not is_ready.is_set():
            raise StalePage("Document is navigating")
        return {
            "url": delayed_url,
            "actions": [{"id": "btn", "kind": "click", "label": "Next", "node": 2}],
            "marker": "m_delayed",
            "fingerprint": "fp_delayed",
        }

    browser.observe.side_effect = dynamic_observe

    t0 = time.monotonic()
    snap1 = session.step()
    dur1 = time.monotonic() - t0

    # Step 1 timed out within ~0.3-0.5s without crashing
    assert dur1 >= 0.28
    assert dur1 < 0.8
    assert snap1["status"] == "ready"
    assert session.status == "ready"
    notes1 = [m["content"] for m in session.ctx.messages if m.get("kind") == "escalation_note"]
    assert "page_unsettled" in notes1

    # Wait for HTTP server delay to complete
    fetch_thread.join(timeout=1.5)
    assert is_ready.is_set()

    # Step 2: page is now settled; agent recovers and performs action
    s1_decision = {
        "operation": "CLICK",
        "target": "btn",
        "choice": "btn",
        "confidence": 0.9,
        "target_confidence": 0.9,
        "target_probabilities": {"btn": 0.9},
        "latency_ms": 10,
        "model": "test",
        "usage": {},
    }
    browser.act.return_value = {"rect": [5, 5, 40, 20]}

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.verify_step") as mock_verify,
    ):
        mock_verify.return_value = MagicMock(
            effective_progress=True,
            page_changed=True,
            url_changed=False,
            element_state_changed=False,
            loop_detected=False,
            oscillation_detected=False,
            model_dump=lambda: {"effective_progress": True},
        )
        snap2 = session.step()

    assert snap2["status"] == "ready"
    assert len(session.history) == 1
    assert session.history[0]["action"] == "Next"
