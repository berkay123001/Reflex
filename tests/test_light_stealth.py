"""Tests for Milestone 2 / Requirement R1e: Light Automation Stealth.

Verifies:
1. Chromium launch flags: --disable-blink-features=AutomationControlled present; --enable-automation absent.
2. Clean User-Agent: HeadlessChrome stripped and replaced with standard Chrome version.
3. navigator.webdriver evaluates to undefined or false in page context.
4. REFLEX_HEADED=1 launches Chromium in headed mode (omits --headless=new).
5. mouseMoved event dispatch preceding click actions when light_stealth=True.
6. Randomized delays (50-250ms) between interactive actions when light_stealth=True.
7. Delays bypassed (0ms) when light_stealth=False for fast test execution.
8. Configuration model and environment variable precedence.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any
from unittest.mock import patch

import pytest

from reflex.browser.cdp import Browser, _stealth_delay
from reflex.browser.isolated import managed_isolated_browser
from reflex.config import ReflexConfig


def is_chromium_available() -> bool:
    """Check if Chromium or Chrome binary is present in PATH."""
    return bool(
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )


def test_chromium_launch_flags_clean_automation() -> None:
    """Verify that Chromium launch command includes --disable-blink-features=AutomationControlled and omits --enable-automation."""
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with patch("subprocess.Popen", side_effect=fake_popen):
        try:
            with managed_isolated_browser(port=9998):
                pass
        except RuntimeError:
            pass

    assert captured_cmd, "Chromium launch command must be captured"
    assert "--disable-blink-features=AutomationControlled" in captured_cmd, (
        "--disable-blink-features=AutomationControlled must be present in launch flags"
    )
    assert not any("--enable-automation" in arg for arg in captured_cmd), (
        "--enable-automation must NEVER be present in launch flags"
    )


def test_clean_user_agent_without_headless() -> None:
    """Verify that navigator.userAgent does not contain HeadlessChrome or Headless."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser("data:text/html,<h1>User Agent Test</h1>")
        try:
            ua = str(browser.evaluate("navigator.userAgent") or "")
            assert "HeadlessChrome" not in ua, f"User-Agent contains HeadlessChrome: {ua}"
            assert "Headless" not in ua, f"User-Agent contains Headless: {ua}"
            assert "Chrome/" in ua, f"User-Agent must look like real Chrome: {ua}"
        finally:
            browser.close()


def test_navigator_webdriver_evaluates_falsy() -> None:
    """Verify that navigator.webdriver evaluates to undefined or false in browser context."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser("data:text/html,<h1>Webdriver Test</h1>")
        try:
            val = browser.evaluate("navigator.webdriver")
            # In JS undefined evaluates to None in Python CDP evaluate
            assert val is None or val is False, f"navigator.webdriver must be undefined or false, got {val!r}"
            # Check descriptor check
            is_defined = browser.evaluate("Boolean(navigator.webdriver)")
            assert is_defined is False, "Boolean(navigator.webdriver) must be false"
        finally:
            browser.close()


def test_reflex_headed_launch_mode_omits_headless_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that REFLEX_HEADED=1 causes managed_isolated_browser to omit --headless=new."""
    captured_cmds: list[list[str]] = []

    def mock_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        captured_cmds.append(list(cmd))
        raise RuntimeError("Captured")

    with patch("subprocess.Popen", side_effect=mock_popen):
        # Case A: REFLEX_HEADED=1
        monkeypatch.setenv("REFLEX_HEADED", "1")
        try:
            with managed_isolated_browser(port=9997):
                pass
        except RuntimeError:
            pass

        assert captured_cmds, "Expected command capture"
        cmd_headed = captured_cmds[-1]
        assert not any("--headless" in arg for arg in cmd_headed), (
            "When REFLEX_HEADED=1, --headless flag must be omitted"
        )

        # Case B: Default (REFLEX_HEADED unset)
        monkeypatch.delenv("REFLEX_HEADED", raising=False)
        try:
            with managed_isolated_browser(port=9996):
                pass
        except RuntimeError:
            pass

        cmd_headless = captured_cmds[-1]
        assert any("--headless=new" in arg for arg in cmd_headless), (
            "By default, --headless=new flag must be present"
        )


def test_mouse_moved_event_dispatched_when_light_stealth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that when light_stealth is True, mouseMoved event is dispatched before click with accurate coords."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    monkeypatch.setenv("REFLEX_LIGHT_STEALTH", "1")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <button id="target_btn" style="position: absolute; left: 100px; top: 100px; width: 120px; height: 50px;">
    Stealth Target
  </button>
  <script>
    window.recorded_events = [];
    const btn = document.getElementById('target_btn');
    ['mousemove', 'mousedown', 'mouseup', 'click'].forEach(evtType => {
      btn.addEventListener(evtType, e => {
        window.recorded_events.push({
          type: e.type,
          x: Math.round(e.clientX),
          y: Math.round(e.clientY),
          t: performance.now()
        });
      });
    });
  </script>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html, light_stealth=True)
        try:
            page = browser.observe()
            act = next(a for a in page["actions"] if a.get("label") == "Stealth Target")
            browser.act(act, page=page)

            events = browser.evaluate("window.recorded_events")
            assert isinstance(events, list) and len(events) >= 4, f"Expected at least 4 mouse events, got {events}"

            # Verify event sequence
            event_types = [e["type"] for e in events]
            assert event_types[0] == "mousemove", f"First event must be mousemove, got {event_types}"
            assert "mousedown" in event_types
            assert "click" in event_types

            # Verify mousemove coordinates match button bounding box
            move_evt = events[0]
            assert 100 <= move_evt["x"] <= 220, f"Move X out of bounds: {move_evt['x']}"
            assert 100 <= move_evt["y"] <= 150, f"Move Y out of bounds: {move_evt['y']}"
            assert move_evt["t"] <= events[1]["t"], "mousemove timestamp must precede mousedown"
        finally:
            browser.close()


def test_mouse_moved_bypassed_when_light_stealth_disabled() -> None:
    """Verify that when light_stealth is False, top-level click bypasses mouseMoved event for speed."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <button id="fast_btn" style="position: absolute; left: 100px; top: 100px; width: 120px; height: 50px;">
    Fast Target
  </button>
  <script>
    window.recorded_events = [];
    const btn = document.getElementById('fast_btn');
    btn.addEventListener('mousemove', () => window.recorded_events.push('mousemove'));
    btn.addEventListener('click', () => window.recorded_events.push('click'));
  </script>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html, light_stealth=False)
        try:
            page = browser.observe()
            act = next(a for a in page["actions"] if a.get("label") == "Fast Target")
            browser.act(act, page=page)

            events = browser.evaluate("window.recorded_events")
            assert "click" in events
            assert "mousemove" not in events, "When light_stealth=False, mousemove should not be dispatched"
        finally:
            browser.close()


def test_randomized_delays_bounded_in_50_to_250ms_range() -> None:
    """Verify that light_stealth=True introduces randomized delays strictly within [50ms, 250ms]."""
    sleep_durations: list[float] = []

    orig_sleep = time.sleep

    def mock_sleep(s: float) -> None:
        sleep_durations.append(s)
        orig_sleep(0.001)

    with patch("time.sleep", side_effect=mock_sleep):
        cfg = ReflexConfig(light_stealth=True)
        # Execute multiple actions through mock/cdp to sample stealth delays
        # Stealth constants: MIN_DELAY = 0.050, MAX_DELAY = 0.250
        samples = [_stealth_delay(cfg) for _ in range(10)]
        for s in samples:
            assert 0.050 <= s <= 0.250, f"Stealth delay {s} out of [0.050, 0.250] range"

        # Verify randomization: not all samples are identical
        assert len(set(samples)) > 1, "Stealth delays must be randomized, not static"


def test_delays_bypassed_when_light_stealth_disabled() -> None:
    """Verify that light_stealth=False introduces zero delay for high-speed test execution."""
    cfg = ReflexConfig(light_stealth=False)
    delay = _stealth_delay(cfg)
    assert delay == 0.0, f"When light_stealth=False, delay must be exactly 0.0, got {delay}"


def test_light_stealth_config_and_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ReflexConfig.light_stealth defaults to False and is enabled via REFLEX_LIGHT_STEALTH=1."""
    # 1. Default config
    monkeypatch.delenv("REFLEX_LIGHT_STEALTH", raising=False)
    cfg_default = ReflexConfig()
    assert cfg_default.light_stealth is False, "light_stealth must default to False"

    # 2. Environment variable toggle
    monkeypatch.setenv("REFLEX_LIGHT_STEALTH", "1")
    cfg_env = ReflexConfig()
    assert cfg_env.light_stealth is True, "REFLEX_LIGHT_STEALTH=1 must enable light_stealth"

    # 3. Explicit override beats env var
    cfg_override = ReflexConfig(light_stealth=False)
    assert cfg_override.light_stealth is False, "Explicit parameter must override environment variable"
