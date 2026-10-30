"""Tests for JavaScript dialog (alert/confirm/prompt/beforeunload) handling and IPC timeout resilience."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest

from reflex.browser.cdp import Browser, EvaluationTimeout, StalePage


def test_browser_init_enables_page_domain():
    """Browser.__init__ must call Page.enable on its attached session so headless Chrome never deadlocks on native dialogs."""
    called_methods: list[str] = []

    def fake_cdp(method: str, session_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        called_methods.append(method)
        if method == "Target.createTarget":
            return {"targetId": "t1"}
        if method == "Target.attachToTarget":
            return {"sessionId": "s1"}
        if method == "Runtime.evaluate":
            return {"result": {"value": "Mozilla/5.0"}}
        return {}

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        Browser("http://example.com")

    assert "Page.enable" in called_methods
    assert "Page.addScriptToEvaluateOnNewDocument" in called_methods


@pytest.mark.parametrize(
    ("dlg_type", "expected_accept"),
    [
        ("alert", True),
        ("beforeunload", True),
        ("confirm", False),
        ("prompt", False),
    ],
)
def test_call_dismisses_native_js_dialog_and_retries_on_timeout(dlg_type: str, expected_accept: bool):
    """When a native JS dialog freezes Runtime.evaluate with TimeoutError (_IPCResponseTimeout),
    Browser.call dismisses it via Page.handleJavaScriptDialog with safe accept policy and retries.
    """
    browser = Browser.__new__(Browser)
    browser._cdp_lock = threading.Lock()
    browser.session = "sess-dlg"
    browser.target = "target-dlg"
    browser._native_dialogs = []

    eval_attempts = 0
    handled_dialogs: list[dict[str, Any]] = []

    def fake_cdp(method: str, session_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        nonlocal eval_attempts
        if method == "Runtime.evaluate":
            eval_attempts += 1
            if eval_attempts == 1:
                raise TimeoutError("Runtime.evaluate timed out after 5s waiting for the daemon")
            return {"result": {"value": "ok_after_dismiss"}}
        if method == "Page.handleJavaScriptDialog":
            handled_dialogs.append({"session_id": session_id, **kwargs})
            return {}
        return {}

    fake_pending = {"dialog": {"type": dlg_type, "message": "Randevular yüklenirken bir hata oluştu"}}

    with (
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch("browser_harness.helpers._send", return_value=fake_pending),
    ):
        val = browser.evaluate("document.title")

    assert val == "ok_after_dismiss"
    assert eval_attempts == 2
    assert len(handled_dialogs) == 1
    assert handled_dialogs[0]["accept"] is expected_accept
    assert any("Randevular yüklenirken bir hata oluştu" in d for d in browser._native_dialogs)


def test_unresolved_timeout_raises_evaluation_timeout_without_10x_observe_spin():
    """If Runtime.evaluate still times out with no JS dialog open, EvaluationTimeout (a StalePage subclass)
    is raised immediately without spinning 10 retries in Browser.observe().
    """
    browser = Browser.__new__(Browser)
    browser._cdp_lock = threading.Lock()
    browser.session = "sess-hang"
    browser.target = "target-hang"
    browser.after_input = None
    browser.last_action_rect = None
    browser.last_jpeg_bytes = None
    browser._native_dialogs = []

    eval_calls = 0

    def fake_cdp(method: str, session_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        nonlocal eval_calls
        if method == "Runtime.evaluate":
            eval_calls += 1
            raise TimeoutError("Runtime.evaluate timed out after 5s waiting for the daemon")
        if method == "Page.handleJavaScriptDialog":
            raise RuntimeError({"code": -32602, "message": "No dialog is showing"})
        return {}

    with (
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch("browser_harness.helpers._send", return_value={"dialog": None}),
        pytest.raises(EvaluationTimeout) as exc_info,
    ):
        browser.observe(screenshot=False)

    assert isinstance(exc_info.value, StalePage)
    assert eval_calls == 1
