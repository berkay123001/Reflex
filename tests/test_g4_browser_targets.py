"""Unit tests for Task G4: Stop closing foreign tabs and warn on CDP URL mismatch."""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any
from unittest.mock import patch

import pytest

from reflex.browser.cdp import Browser, ensure_chromium_cdp


def _create_mock_cdp(target_id: str = "target_test_123", session_id: str = "session_test_456"):
    closed_targets: list[str] = []
    cdp_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_cdp(method: str, session_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        cdp_calls.append((method, kwargs))
        if method == "Target.createTarget":
            return {"targetId": target_id}
        if method == "Target.attachToTarget":
            return {"sessionId": session_id}
        if method == "Target.getTargets":
            return {
                "targetInfos": [
                    {"targetId": "foreign_tab_1", "type": "page"},
                    {"targetId": "foreign_tab_2", "type": "page"},
                    {"targetId": "foreign_ext_3", "type": "background_page"},
                    {"targetId": target_id, "type": "page"},
                ]
            }
        if method == "Target.closeTarget":
            tid = kwargs.get("targetId", "")
            closed_targets.append(tid)
            return {"success": True}
        if method == "Page.bringToFront":
            return {}
        if method == "Emulation.setDeviceMetricsOverride":
            return {}
        if method == "Emulation.setFocusEmulationEnabled":
            return {}
        if method == "Network.enable":
            return {}
        if method == "Network.setBlockedURLs":
            return {}
        if method == "Runtime.evaluate":
            return {"result": {"value": "Mozilla/5.0 Chrome/120.0.0.0"}}
        if method == "Page.addScriptToEvaluateOnNewDocument":
            return {"identifier": "script_1"}
        return {}

    return fake_cdp, closed_targets, cdp_calls


def test_browser_init_never_closes_foreign_targets():
    """Verify that Browser.__init__ never calls Target.closeTarget on preexisting foreign tabs."""
    fake_cdp, closed_targets, cdp_calls = _create_mock_cdp("my_browser_target")

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        browser = Browser("http://example.com")

        # Verify Target.closeTarget was NEVER called during initialization
        assert len(closed_targets) == 0, f"Expected 0 closed targets on startup, got {closed_targets}"
        assert "foreign_tab_1" not in closed_targets
        assert "foreign_tab_2" not in closed_targets

        # Verify Target.getTargets was never even invoked to close tabs
        get_targets_calls = [m for m, _ in cdp_calls if m == "Target.getTargets"]
        assert len(get_targets_calls) == 0

        # Verify target ownership
        assert browser.target == "my_browser_target"
        assert browser._owned_targets == {"my_browser_target"}

        # Now close browser and verify ONLY owned target is closed
        browser.close()
        assert closed_targets == ["my_browser_target"]
        assert browser.target is None
        assert browser._owned_targets == set()


def test_browser_close_only_closes_owned_targets_set():
    """Verify that Browser.close() only closes targets registered in _owned_targets."""
    fake_cdp, closed_targets, _ = _create_mock_cdp("main_target")

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        browser = Browser("http://example.com")
        # Add another owned target (e.g. from popup tracking)
        browser._owned_targets.add("popup_target")

        browser.close()
        assert set(closed_targets) == {"main_target", "popup_target"}
        assert "foreign_tab_1" not in closed_targets
        assert browser.target is None
        assert len(browser._owned_targets) == 0


def test_browser_close_handles_cdp_exception_gracefully():
    """Verify that errors from Target.closeTarget do not crash Browser.close()."""
    def broken_cdp(method: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if method == "Target.createTarget":
            return {"targetId": "err_target"}
        if method == "Target.attachToTarget":
            return {"sessionId": "err_session"}
        if method == "Target.closeTarget":
            raise RuntimeError("CDP connection closed")
        return {}

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=broken_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        browser = Browser("http://example.com")
        assert browser._owned_targets == {"err_target"}
        # Should not raise exception
        browser.close()
        assert browser.target is None
        assert len(browser._owned_targets) == 0


def test_browser_context_manager_cleanup():
    """Verify Browser context manager cleanly closes owned target on block exit."""
    fake_cdp, closed_targets, _ = _create_mock_cdp("ctx_target")

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        with Browser("http://example.com") as b:
            assert b.target == "ctx_target"
            assert b._owned_targets == {"ctx_target"}
            assert len(closed_targets) == 0

        assert closed_targets == ["ctx_target"]
        assert b.target is None
        assert len(b._owned_targets) == 0


def test_ensure_chromium_cdp_warns_on_url_mismatch(caplog):
    """When BU_CDP_URL is set to a different address than expected port, log a warning and emit UserWarning."""
    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9444"}, clear=False),
        patch("reflex.browser.cdp._port_open", return_value=True),
    ):
        with caplog.at_level(logging.WARNING), pytest.warns(UserWarning) as warn_records:
            ensure_chromium_cdp(port=9333)

        # Check logger.warning
        warning_logs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("BU_CDP_URL is already set to 'http://127.0.0.1:9444'" in msg for msg in warning_logs)
        assert any("http://127.0.0.1:9333" in msg for msg in warning_logs)

        # Check warnings.warn
        assert len(warn_records) >= 1
        assert "BU_CDP_URL is already set to 'http://127.0.0.1:9444'" in str(warn_records[0].message)


def test_ensure_chromium_cdp_no_warning_when_matching(caplog):
    """When BU_CDP_URL matches the expected port address, no warning is issued."""
    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9333"}, clear=False),
        patch("reflex.browser.cdp._port_open", return_value=True),
    ):
        with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as warn_records:
            warnings.simplefilter("always")
            ensure_chromium_cdp(port=9333)

        assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 0
        assert len(warn_records) == 0


def test_ensure_chromium_cdp_sets_default_when_unset(caplog):
    """When BU_CDP_URL is not set, setdefault sets it without issuing any warning."""
    env = os.environ.copy()
    env.pop("BU_CDP_URL", None)

    with (
        patch.dict(os.environ, env, clear=True),
        patch("reflex.browser.cdp._port_open", return_value=True),
    ):
        with caplog.at_level(logging.WARNING), warnings.catch_warnings(record=True) as warn_records:
            warnings.simplefilter("always")
            ensure_chromium_cdp(port=9555)

        assert os.environ.get("BU_CDP_URL") == "http://127.0.0.1:9555"
        assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 0
        assert len(warn_records) == 0
