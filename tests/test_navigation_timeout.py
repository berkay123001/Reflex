"""A site that never answers must not crash the session: navigation times out into a normal S2 escalation."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest
from browser_harness.helpers import _IPCResponseTimeout

from reflex.browser.cdp import NAVIGATE_RESPONSE_TIMEOUT_S, Browser, NavigationTimeout
from reflex.config import ReflexConfig, run_mode_flags
from reflex.context import DualTierContext

DEAD_URL = "http://dead-site.example/"


def _bare_browser() -> Browser:
    b = Browser.__new__(Browser)
    b.session = "s1"
    b.target = "t1"
    b._cdp_lock = threading.Lock()
    b._native_dialogs = []
    b.after_input = None
    b.evaluate = lambda expr: 1.0  # type: ignore[method-assign]
    b._dismiss_js_dialog = lambda: None  # type: ignore[method-assign]
    return b


def test_navigate_waits_longer_than_default_ipc_timeout_then_stops_loading_and_raises() -> None:
    b = _bare_browser()
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_cdp(method: str, **kw: Any) -> dict[str, Any]:
        calls.append((method, kw))
        if method == "Page.navigate":
            raise _IPCResponseTimeout("Page.navigate timed out")
        return {}

    with patch("reflex.browser.cdp.cdp", side_effect=fake_cdp), pytest.raises(NavigationTimeout, match="dead-site.example"):
        b.navigate(DEAD_URL)

    nav_kw = next(kw for m, kw in calls if m == "Page.navigate")
    assert nav_kw["_response_timeout"] == NAVIGATE_RESPONSE_TIMEOUT_S > 5
    assert "Page.stopLoading" in [m for m, _ in calls]


def test_navigate_success_still_waits_for_settle() -> None:
    b = _bare_browser()
    settled: list[Any] = []
    b._wait_for_navigation_and_settle = lambda prev_origin, max_wait: settled.append(prev_origin)  # type: ignore[method-assign]
    with patch("reflex.browser.cdp.cdp", return_value={"frameId": "f"}):
        b.navigate("https://ok.example/")
    assert settled == [1.0]


class _DeadSiteBrowser:
    def __init__(self, page: dict[str, Any]) -> None:
        self.page = page

    def act(self, action: dict[str, Any], page: dict[str, Any], text: str | None = None) -> dict[str, Any]:
        raise NavigationTimeout(f"{DEAD_URL} did not respond")

    def observe(self, screenshot: bool = False) -> dict[str, Any]:
        return self.page


def test_goto_dead_site_escalates_to_s2_and_marks_link_rejected_instead_of_crashing() -> None:
    from reflex.agent import ReflexSession

    page = {"url": "https://list.example/members", "actions": [], "fingerprint": "fp-list"}
    sess = ReflexSession.__new__(ReflexSession)
    sess.config = ReflexConfig(**run_mode_flags("s2_only"))
    sess.ctx = DualTierContext(primary_goal="g", active_subgoal="g")
    sess.page = page
    sess.history = []
    sess.browser = _DeadSiteBrowser(page)
    sess.status = "ready"
    sess.last_error = None
    sess.pending_escalation_reason = None
    sess._observe_settled = lambda: page  # type: ignore[method-assign]
    sess.snapshot = lambda: {"status": sess.status}  # type: ignore[method-assign]

    sess._execute_and_verify(
        action={"kind": "goto_url", "url": DEAD_URL, "label": f"GOTO_URL {DEAD_URL}"},
        actor="S2",
        operation="GOTO_URL",
        target=None,
        confidence=1.0,
        target_confidence=None,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=0,
        model_name="m",
        usage={},
        preset_text=DEAD_URL,
        escalation_note=None,
    )

    assert sess.status == "ready"
    assert sess.last_error is None
    assert sess.pending_escalation_reason is not None
    assert sess.pending_escalation_reason[0] == "navigation_timeout"
    assert DEAD_URL in sess.ctx.rejected_links[-1]
    assert DEAD_URL in sess.ctx.build_supervisor_context_block([])
