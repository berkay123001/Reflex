"""Tests for Milestone 2 / Requirement R1d: Accessibility Tree (AXTree) Supplementation.

Verifies:
1. Discovery of custom interactive accessibility elements lacking semantic tags (canvas, custom ARIA widgets).
2. Strict deduplication against snapshot.js using backendDOMNodeId (no duplicate actions).
3. Filtering of invisible, disabled, zero-size, and aria-hidden nodes.
4. Configuration toggle: config.ax_supplement = False and REFLEX_AX_SUPPLEMENT=0 disable AXTree pipeline.
5. Action execution (click, fill) on AX-supplemented elements registered into window.__jevFast.
6. Interactive role matrix (combobox, spinbutton, menuitem, treeitem).
7. Multi-frame accessibility element discovery and coordinate translation.
8. ReflexConfig model precedence and environment variable handling.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import shutil
import threading
import time
from typing import Any

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser
from reflex.config import ReflexConfig


def is_chromium_available() -> bool:
    """Check if Chromium or Chrome binary is present in PATH."""
    return bool(
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )


def test_ax_custom_accessibility_element_discovery() -> None:
    """Verify that AXTree discovers interactive elements (canvas buttons, switches, sliders) lacking semantic tags."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>AX Custom Elements</title></head>
<body>
  <h1>Interactive Non-Semantic Widgets</h1>
  <!-- Canvas-based interactive element -->
  <canvas id="canvas_btn" role="button" tabindex="0" aria-label="Canvas Play Button"
          style="width: 120px; height: 60px;"></canvas>

  <!-- Custom switch widget -->
  <div id="custom_switch" role="switch" aria-checked="false" tabindex="0" aria-label="Mute Audio Toggle"
       style="width: 80px; height: 30px;">OFF</div>

  <!-- Custom slider widget -->
  <div id="custom_slider" role="slider" aria-valuenow="40" aria-valuemin="0" aria-valuemax="100"
       tabindex="0" aria-label="Volume Level Slider" style="width: 150px; height: 20px;"></div>

  <!-- Custom tab widget -->
  <div id="custom_tab" role="tab" aria-selected="true" tabindex="0" aria-label="Account Settings Tab"
       style="width: 100px; height: 35px;">Settings</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # 1. Canvas button discovery
            canvas_act = next((a for a in actions if a.get("label") == "Canvas Play Button"), None)
            assert canvas_act is not None, "Canvas button must be discovered via AXTree"
            assert canvas_act["role"] == "button"
            assert canvas_act["kind"] == "click"
            assert canvas_act.get("rect", {}).get("w", 0) > 0

            # 2. Switch discovery
            switch_act = next((a for a in actions if a.get("label") == "Mute Audio Toggle"), None)
            assert switch_act is not None, "Custom switch must be discovered via AXTree"
            assert switch_act["role"] == "switch"
            assert switch_act["kind"] == "click"

            # 3. Slider discovery
            slider_act = next((a for a in actions if a.get("label") == "Volume Level Slider"), None)
            assert slider_act is not None, "Custom slider must be discovered via AXTree"
            assert slider_act["role"] == "slider"

            # 4. Tab discovery
            tab_act = next((a for a in actions if a.get("label") == "Account Settings Tab"), None)
            assert tab_act is not None, "Custom tab must be discovered via AXTree"
            assert tab_act["role"] == "tab"

            # 5. ID sequence verification
            action_ids = [a["id"] for a in actions if a["id"].startswith("e")]
            assert action_ids == [f"e{i+1}" for i in range(len(action_ids))], "Action IDs must be contiguous"
        finally:
            browser.close()


def test_ax_deduplication_via_backend_dom_node_id() -> None:
    """Verify that semantic elements discovered by snapshot.js are not duplicated by AXTree."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <!-- Standard semantic elements discovered by snapshot.js -->
  <button id="std_btn">Standard Checkout</button>
  <a href="/test" id="std_link">Privacy Policy</a>
  <input type="text" id="std_inp" aria-label="First Name" value="Alice" />

  <!-- Custom element -->
  <div id="custom_act" role="button" tabindex="0" aria-label="Custom Action Button"
       style="width: 100px; height: 40px;">Custom Action</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # Count occurrences of each action label
            checkout_acts = [a for a in actions if a.get("label") == "Standard Checkout"]
            assert len(checkout_acts) == 1, "Semantic button must not be duplicated"

            policy_acts = [a for a in actions if a.get("label") == "Privacy Policy"]
            assert len(policy_acts) == 1, "Semantic link must not be duplicated"

            name_acts = [a for a in actions if a.get("label") == "First Name"]
            assert len(name_acts) == 1, "Semantic text input must not be duplicated"

            custom_acts = [a for a in actions if a.get("label") == "Custom Action Button"]
            assert len(custom_acts) == 1, "Custom action must be discovered exactly once"
        finally:
            browser.close()


def test_ax_filtering_invisible_disabled_zero_size_aria_hidden() -> None:
    """Verify that invisible, disabled, zero-size, or aria-hidden nodes are strictly excluded."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <!-- Visible valid button -->
  <button id="valid_btn">Permitted Button</button>

  <!-- Invisible: display none -->
  <div style="display: none;" role="button" aria-label="Hidden Display Button">Hidden Display</div>

  <!-- Invisible: visibility hidden -->
  <div style="visibility: hidden;" role="button" aria-label="Hidden Vis Button">Hidden Vis</div>

  <!-- Disabled elements -->
  <button disabled aria-label="Disabled Native Button">Disabled Native</button>
  <div role="button" aria-disabled="true" tabindex="0" aria-label="Aria Disabled Button">Aria Disabled</div>

  <!-- Zero-size element -->
  <div style="width: 0px; height: 0px; overflow: hidden;" role="button" aria-label="Zero Size Button">Zero</div>

  <!-- Aria-hidden container -->
  <div aria-hidden="true">
    <div role="button" tabindex="0" aria-label="Aria Hidden Nested Button">Aria Hidden Nested</div>
  </div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]
            labels = [a.get("label") for a in actions]

            assert "Permitted Button" in labels, "Permitted button must be present"
            assert "Hidden Display Button" not in labels, "display:none element must be excluded"
            assert "Hidden Vis Button" not in labels, "visibility:hidden element must be excluded"
            assert "Disabled Native Button" not in labels, "disabled button must be excluded"
            assert "Aria Disabled Button" not in labels, "aria-disabled element must be excluded"
            assert "Zero Size Button" not in labels, "Zero-size element must be excluded"
            assert "Aria Hidden Nested Button" not in labels, "aria-hidden element must be excluded"
        finally:
            browser.close()


def test_ax_supplement_config_toggle_and_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that config.ax_supplement = False or REFLEX_AX_SUPPLEMENT=0 disables AXTree supplementation."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <button id="btn1">Semantic Always Present</button>
  <canvas id="c1" role="button" tabindex="0" aria-label="Canvas AX Only"
          style="width: 100px; height: 40px;"></canvas>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        # Case A: Disabled via environment variable REFLEX_AX_SUPPLEMENT=0
        monkeypatch.setenv("REFLEX_AX_SUPPLEMENT", "0")
        browser_disabled = Browser(html)
        try:
            page_dis = browser_disabled.observe()
            labels_dis = [a.get("label") for a in page_dis["actions"]]
            assert "Semantic Always Present" in labels_dis
            assert "Canvas AX Only" not in labels_dis, "AX-only element must NOT be discovered when AX supplement disabled"
        finally:
            browser_disabled.close()

        # Case B: Enabled via environment variable REFLEX_AX_SUPPLEMENT=1
        monkeypatch.setenv("REFLEX_AX_SUPPLEMENT", "1")
        browser_enabled = Browser(html)
        try:
            page_en = browser_enabled.observe()
            labels_en = [a.get("label") for a in page_en["actions"]]
            assert "Semantic Always Present" in labels_en
            assert "Canvas AX Only" in labels_en, "AX-only element must be discovered when AX supplement enabled"
        finally:
            browser_enabled.close()


def test_ax_action_execution_click_and_fill() -> None:
    """Verify that act() successfully executes click and fill on AX-supplemented elements."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <div id="status">initial</div>
  <!-- Custom click target -->
  <div id="custom_clickable" role="button" tabindex="0"
       style="width: 120px; height: 40px; background: rgb(238, 238, 238);"
       onclick="document.getElementById('status').textContent='custom_clicked'"
       aria-label="Custom Click Target">Custom Click Target</div>

  <!-- Custom editable target -->
  <div id="custom_text" role="textbox" contenteditable="true" tabindex="0"
       style="width: 200px; height: 40px; border: 1px solid black;"
       aria-label="Custom Rich Textbox"></div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()

            # 1. Click execution on AX-supplemented element
            click_act = next(a for a in page["actions"] if a.get("label") == "Custom Click Target")
            res_click = browser.act(click_act, page=page)
            assert res_click["executed"] == click_act["id"]
            assert browser.evaluate("document.getElementById('status').textContent") == "custom_clicked"

            # 2. Fill execution on AX-supplemented element
            fill_act = next(a for a in page["actions"] if a.get("label") == "Custom Rich Textbox")
            res_fill = browser.act(fill_act, page=page, text="Verified AX Typing")
            assert res_fill["executed"] == fill_act["id"]
            assert "Verified AX Typing" in str(browser.evaluate("document.getElementById('custom_text').innerText"))
        finally:
            browser.close()


def test_ax_interactive_role_matrix() -> None:
    """Verify that AXTree recognizes combobox, spinbutton, menuitem, and treeitem roles."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <div role="combobox" aria-expanded="false" tabindex="0" aria-label="City Selector"
       style="width: 150px; height: 35px;">Select City</div>
  <div role="spinbutton" aria-valuenow="5" aria-valuemin="1" aria-valuemax="10" tabindex="0"
       aria-label="Quantity Picker" style="width: 80px; height: 35px;">5</div>
  <div role="menu">
    <div role="menuitem" tabindex="0" aria-label="Print Document" style="width: 120px; height: 30px;">Print</div>
  </div>
  <div role="tree">
    <div role="treeitem" tabindex="0" aria-label="Project Folder" style="width: 140px; height: 30px;">Project</div>
  </div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            combo = next((a for a in actions if a.get("label") == "City Selector"), None)
            assert combo is not None
            assert combo["role"] == "combobox"

            spin = next((a for a in actions if a.get("label") == "Quantity Picker"), None)
            assert spin is not None
            assert spin["role"] == "spinbutton"
            assert spin["kind"] == "fill"

            menu = next((a for a in actions if a.get("label") == "Print Document"), None)
            assert menu is not None
            assert menu["role"] == "menuitem"
            assert menu["kind"] == "click"

            tree = next((a for a in actions if a.get("label") == "Project Folder"), None)
            assert tree is not None
            assert tree["role"] == "treeitem"
            assert tree["kind"] == "click"
        finally:
            browser.close()


def test_ax_multi_frame_coordinate_translation() -> None:
    """Verify that elements inside visible same-origin iframes have valid coordinates and actions."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    class IframeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if "/frame" in self.path:
                self.wfile.write(b"""<!DOCTYPE html>
<html>
<body>
  <button id="iframe_btn" style="width: 120px; height: 40px;">Frame Button</button>
  <div id="iframe_switch" role="switch" aria-checked="false" tabindex="0" aria-label="Frame Switch"
       style="width: 90px; height: 30px;">Toggle</div>
</body>
</html>""")
            else:
                self.wfile.write(b"""<!DOCTYPE html>
<html>
<body>
  <h1>Parent Document</h1>
  <iframe src="/frame" style="width: 400px; height: 300px; margin-top: 50px; margin-left: 50px; border: none;"></iframe>
</body>
</html>""")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), IframeHandler)
    server_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with managed_isolated_browser(port=0) as cdp_url:
            browser = Browser(f"http://127.0.0.1:{server_port}/")
            try:
                page = browser.observe()
                actions = page["actions"]

                # Verify iframe button is discovered
                btn = next((a for a in actions if a.get("label") == "Frame Button"), None)
                assert btn is not None, "Button in iframe must be discovered"
                assert btn.get("rect", {}).get("y", 0) >= 50, "Translated coordinate must account for iframe offset"

                # Verify iframe switch is discovered
                sw = next((a for a in actions if a.get("label") == "Frame Switch"), None)
                assert sw is not None, "Switch in iframe must be discovered"
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_ax_config_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ReflexConfig.ax_supplement defaults to True and respects REFLEX_AX_SUPPLEMENT."""
    # 1. Default config
    monkeypatch.delenv("REFLEX_AX_SUPPLEMENT", raising=False)
    cfg_default = ReflexConfig()
    assert cfg_default.ax_supplement is True, "ax_supplement must default to True"

    # 2. Disabled via env var
    monkeypatch.setenv("REFLEX_AX_SUPPLEMENT", "0")
    cfg_env = ReflexConfig()
    assert cfg_env.ax_supplement is False, "REFLEX_AX_SUPPLEMENT=0 must disable ax_supplement"

    # 3. Explicit override beats env var
    cfg_override = ReflexConfig(ax_supplement=True)
    assert cfg_override.ax_supplement is True, "Explicit parameter must override environment variable"
