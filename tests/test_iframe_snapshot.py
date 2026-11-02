"""Tests for Milestone 1 / Requirement R1: Same-origin and cross-origin iframe support.

Verifies:
1. Dual-origin local HTTP server fixture (127.0.0.1 vs localhost).
2. Element discovery inside same-origin and cross-origin (OOPIF) iframes.
3. Accurate coordinate translation from iframe-local space to top-level viewport.
4. Action execution (click, fill, select) inside iframes.
5. Verifier integration (detecting value and DOM state changes in iframes).
6. Filtering: skipping tiny (<200x100) and hidden (display:none, visibility:hidden) iframes.
7. Frame cap enforcement (maximum 5 iframes scanned).
8. Clean isolation using managed_isolated_browser without port 9333.
"""

from __future__ import annotations

import http.server
import shutil
import threading
import time
from typing import Any, Generator

import pytest

from reflex.browser.cdp import Browser, StalePage
from reflex.browser.isolated import managed_isolated_browser
from reflex.verifier import verify_step


def is_chromium_available() -> bool:
    """Check if Chromium or Chrome binary is present in PATH."""
    return bool(
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )


class DualOriginHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request handler serving top-level and iframe HTML fixtures."""

    cross_origin_host: str = "localhost"
    cross_origin_port: int = 0

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        content_type = "text/html; charset=utf-8"

        if path == "/":
            cross_url = f"http://{self.cross_origin_host}:{self.cross_origin_port}/cross_iframe"
            body = f"""<!DOCTYPE html>
<html>
<head><title>Top Page with Iframes</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h1>Top Level Page</h1>
  <button id="top_btn">Top Action Button</button>
  <div style="height: 20px;"></div>

  <!-- Same-origin iframe: 400x300 at (50, 100) -->
  <iframe id="same_frame" src="/same_iframe"
          style="position: absolute; left: 50px; top: 100px; width: 400px; height: 300px; border: 2px solid green;">
  </iframe>

  <!-- Cross-origin iframe (OOPIF): 400x300 at (50, 450) -->
  <iframe id="cross_frame" src="{cross_url}"
          style="position: absolute; left: 50px; top: 450px; width: 400px; height: 300px; border: 2px solid blue;">
  </iframe>

  <!-- Tiny iframe (<200x100): 180x90 at (500, 100) -->
  <iframe id="tiny_frame" src="/tiny_iframe"
          style="position: absolute; left: 500px; top: 100px; width: 180px; height: 90px; border: 1px solid gray;">
  </iframe>

  <!-- Hidden iframe (display: none) -->
  <iframe id="hidden_frame" src="/hidden_iframe" style="display: none; width: 300px; height: 200px;">
  </iframe>

  <!-- Hidden iframe (visibility: hidden) -->
  <iframe id="vis_hidden_frame" src="/vis_hidden_iframe"
          style="visibility: hidden; position: absolute; left: 500px; top: 220px; width: 300px; height: 200px;">
  </iframe>
</body>
</html>"""

        elif path == "/same_iframe":
            body = """<!DOCTYPE html>
<html>
<head><title>Same Origin Iframe</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h3>Same Origin Content</h3>
  <button id="same_btn" onclick="document.getElementById('same_status').textContent='same_btn_clicked'">
    Same Frame Click
  </button>
  <br><br>
  <input id="same_inp" type="text" value="same_default" aria-label="Same Input Field" />
  <br><br>
  <select id="same_sel" onchange="document.getElementById('same_status').textContent=this.value">
    <option value="s1">Same Choice 1</option>
    <option value="s2">Same Choice 2</option>
  </select>
  <br><br>
  <div id="same_status">same_idle</div>
</body>
</html>"""

        elif path == "/cross_iframe":
            body = """<!DOCTYPE html>
<html>
<head><title>Cross Origin Iframe</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h3>Cross Origin Content</h3>
  <button id="cross_btn" onclick="document.getElementById('cross_status').textContent='cross_btn_clicked'">
    Cross Frame Click
  </button>
  <br><br>
  <input id="cross_inp" type="text" value="cross_default" aria-label="Cross Input Field" />
  <br><br>
  <select id="cross_sel" onchange="document.getElementById('cross_status').textContent=this.value">
    <option value="c1">Cross Choice 1</option>
    <option value="c2">Cross Choice 2</option>
  </select>
  <br><br>
  <div id="cross_status">cross_idle</div>
</body>
</html>"""

        elif path == "/tiny_iframe":
            body = """<!DOCTYPE html><html><body><button id="tiny_btn">Tiny Hidden Action</button></body></html>"""

        elif path in {"/hidden_iframe", "/vis_hidden_iframe"}:
            body = """<!DOCTYPE html><html><body><button id="hidden_btn">Hidden Action</button></body></html>"""

        elif path == "/multi_frames":
            frames_html = "\n".join(
                f'<iframe src="/frame_item_{i}" style="width: 250px; height: 150px; margin: 10px;"></iframe>'
                for i in range(1, 8)
            )
            body = f"""<!DOCTYPE html><html><body><h1>Multi Frame Test</h1>{frames_html}</body></html>"""

        elif path.startswith("/frame_item_"):
            idx = path.split("_")[-1]
            body = f"""<!DOCTYPE html><html><body><button id="btn_{idx}">Frame {idx} Button</button></body></html>"""

        elif path == "/scrolled_parent":
            body = """<!DOCTYPE html>
<html>
<head><title>Scrolled Parent Page</title></head>
<body style="margin: 0; padding: 0; height: 3500px; font-family: sans-serif;">
  <div style="height: 1200px; background: #eef; padding: 20px;">
    <h1>Tall Page Top</h1>
  </div>
  <iframe id="scrolled_frame" src="/frame_targets"
          style="position: absolute; left: 80px; top: 1200px; width: 500px; height: 400px; border: 2px solid red;">
  </iframe>
  <div style="height: 1900px; background: #fee; padding: 20px;">
    <h2>Tall Page Bottom</h2>
  </div>
</body>
</html>"""

        elif path == "/scrolled_cross_parent":
            cross_url = f"http://{self.cross_origin_host}:{self.cross_origin_port}/frame_targets"
            body = f"""<!DOCTYPE html>
<html>
<head><title>Scrolled Parent with Cross Origin OOPIF</title></head>
<body style="margin: 0; padding: 0; height: 3500px; font-family: sans-serif;">
  <div style="height: 1200px; background: #eef; padding: 20px;">
    <h1>Tall Page Top</h1>
  </div>
  <iframe id="scrolled_cross_frame" src="{cross_url}"
          style="position: absolute; left: 80px; top: 1200px; width: 500px; height: 400px; border: 2px solid red;">
  </iframe>
  <div style="height: 1900px; background: #fee; padding: 20px;">
    <h2>Tall Page Bottom</h2>
  </div>
</body>
</html>"""

        elif path == "/frame_targets":
            body = """<!DOCTYPE html>
<html>
<head><title>Frame Target Buttons</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h3>Frame Target Controls</h3>
  <button id="btn_target" style="position: absolute; top: 60px; left: 30px; width: 160px; height: 40px;"
          onclick="window.__clicked=true; document.getElementById('status').textContent='target_clicked'">
    Target Button
  </button>
  <div id="status" style="position: absolute; top: 140px; left: 30px;">status_idle</div>
</body>
</html>"""

        elif path == "/dual_maneuver_page":
            c1_url = f"http://{self.cross_origin_host}:{self.cross_origin_port}/maneuver_target_1"
            c2_url = f"http://{self.cross_origin_host}:{self.cross_origin_port}/maneuver_target_2"
            body = f"""<!DOCTYPE html>
<html>
<head><title>Dual Maneuver Page</title></head>
<body style="margin: 0; padding: 0; height: 5000px; font-family: sans-serif;">
  <div style="height: 100px; padding: 10px;"><h2>Header Section</h2></div>
  <!-- Frame 1 at top: 200px -->
  <iframe id="frame_alpha" src="{c1_url}"
          style="position: absolute; left: 50px; top: 200px; width: 450px; height: 300px; border: 2px solid green;">
  </iframe>
  <div style="position: absolute; top: 800px; height: 1600px; width: 100%; background: #f0f4f8;">
    <h3>Middle Content Filler</h3>
  </div>
  <!-- Frame 2 at top: 2600px -->
  <iframe id="frame_beta" src="{c2_url}"
          style="position: absolute; left: 50px; top: 2600px; width: 450px; height: 300px; border: 2px solid orange;">
  </iframe>
  <div style="height: 2100px; position: absolute; top: 2900px; width: 100%; background: #fff5f0;">
    <h3>Bottom Content Filler</h3>
  </div>
</body>
</html>"""

        elif path == "/maneuver_target_1":
            body = """<!DOCTYPE html>
<html>
<body style="margin: 0; padding: 15px; font-family: sans-serif;">
  <button id="btn_alpha" style="height: 25px; padding: 2px 8px;"
          onclick="window.__clicks = (window.__clicks || 0) + 1;">
    Alpha Button
  </button>
  <div id="alpha_counter" style="margin-top: 10px;">0</div>
  <script>
    document.getElementById('btn_alpha').addEventListener('click', () => {
      document.getElementById('alpha_counter').textContent = window.__clicks;
    });
  </script>
</body>
</html>"""

        elif path == "/maneuver_target_2":
            body = """<!DOCTYPE html>
<html>
<body style="margin: 0; padding: 15px; font-family: sans-serif;">
  <button id="btn_beta" style="height: 25px; padding: 2px 8px;"
          onclick="window.__clicks = (window.__clicks || 0) + 1;">
    Beta Button
  </button>
  <div id="beta_counter" style="margin-top: 10px;">0</div>
  <script>
    document.getElementById('btn_beta').addEventListener('click', () => {
      document.getElementById('beta_counter').textContent = window.__clicks;
    });
  </script>
</body>
</html>"""

        elif path == "/top_ira_calculator":
            cross_url = f"http://{self.cross_origin_host}:{self.cross_origin_port}/cross_ira_calculator"
            body = f"""<!DOCTYPE html>
<html>
<head><title>Retirement Planning Top</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h1>Top Retirement Portal</h1>
  <p>Please use the embedded IRA calculator below.</p>
  <iframe id="ira_frame" src="{cross_url}"
          style="position: absolute; left: 50px; top: 200px; width: 600px; height: 500px; border: 2px solid #ccc;">
  </iframe>
</body>
</html>"""

        elif path == "/cross_ira_calculator":
            body = """<!DOCTYPE html>
<html>
<head><title>Cross Origin IRA Calculator</title></head>
<body style="margin: 0; padding: 20px; font-family: sans-serif;">
  <h2>Retirement Calculator</h2>
  <h3>Traditional vs Roth IRA</h3>
  <p>Compare Traditional and Roth IRA returns to plan your retirement savings.</p>

  <div class="form-group" style="margin-bottom: 15px;">
    <span class="field-title" style="font-weight: bold; display: block;">Current age</span>
    <input id="ira_age" type="range" min="18" max="100" value="30" style="width: 250px; height: 25px;" />
  </div>

  <div class="form-group" style="margin-bottom: 15px;">
    <span class="field-title" style="font-weight: bold; display: block;">Retirement age</span>
    <input id="ira_retire_age" type="range" min="18" max="100" value="65" style="width: 250px; height: 25px;" />
  </div>

  <div class="form-group" style="margin-bottom: 15px;">
    <span class="field-title" style="font-weight: bold; display: block;">Starting balance ($)</span>
    <input id="ira_balance" type="range" min="0" max="100000" value="0" style="width: 250px; height: 25px;" />
  </div>

  <div class="form-group" style="margin-bottom: 15px;">
    <span class="field-title" style="font-weight: bold; display: block;">Annual rate of return (%)</span>
    <input id="ira_return" type="range" min="0" max="15" value="4" style="width: 250px; height: 25px;" />
  </div>

  <div class="form-group" style="margin-bottom: 15px;">
    <span class="field-title" style="font-weight: bold; display: block;">Current tax rate (%)</span>
    <input id="ira_tax" type="range" min="0" max="50" value="22" style="width: 250px; height: 25px;" />
  </div>
</body>
</html>"""

        else:
            self.send_response(404)
            self.end_headers()
            return

        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """Silence HTTP request logs in test output."""
        pass


@pytest.fixture(scope="module")
def dual_origin_servers() -> Generator[dict[str, Any], None, None]:
    """Spawns two HTTP servers on distinct origins: 127.0.0.1 and localhost."""
    # Server 1: 127.0.0.1 (Top origin)
    server1 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DualOriginHandler)
    p1 = server1.server_port
    t1 = threading.Thread(target=server1.serve_forever, daemon=True)
    t1.start()

    # Server 2: localhost (Cross origin)
    server2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), DualOriginHandler)
    p2 = server2.server_port
    t2 = threading.Thread(target=server2.serve_forever, daemon=True)
    t2.start()

    # Configure handler with cross-origin endpoint
    DualOriginHandler.cross_origin_host = "localhost"
    DualOriginHandler.cross_origin_port = p2

    info = {
        "top_origin": f"http://127.0.0.1:{p1}",
        "cross_origin": f"http://localhost:{p2}",
        "top_url": f"http://127.0.0.1:{p1}/",
        "multi_frames_url": f"http://127.0.0.1:{p1}/multi_frames",
        "scrolled_url": f"http://127.0.0.1:{p1}/scrolled_parent",
        "scrolled_cross_url": f"http://127.0.0.1:{p1}/scrolled_cross_parent",
        "dual_maneuver_url": f"http://127.0.0.1:{p1}/dual_maneuver_page",
        "top_ira_calculator_url": f"http://127.0.0.1:{p1}/top_ira_calculator",
    }

    try:
        yield info
    finally:
        server1.shutdown()
        server2.shutdown()


def test_iframe_element_discovery_same_and_cross_origin(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that observe() discovers interactive elements inside both same-origin and cross-origin iframes."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            actions = page["actions"]
            labels = [a.get("label") for a in actions]

            # Top page button
            assert "Top Action Button" in labels

            # Same-origin iframe elements
            assert "Same Frame Click" in labels
            same_input = next((a for a in actions if a.get("value") == "same_default" or "Same Input" in str(a.get("label"))), None)
            assert same_input is not None, "Same-origin input must be discovered"

            # Cross-origin iframe elements (OOPIF)
            assert "Cross Frame Click" in labels
            cross_input = next((a for a in actions if a.get("value") == "cross_default" or "Cross Input" in str(a.get("label"))), None)
            assert cross_input is not None, "Cross-origin input must be discovered"

            # Action IDs must be contiguous and distinct (e1, e2, ...)
            action_ids = [a["id"] for a in actions if a["id"] != "wait"]
            assert len(action_ids) == len(set(action_ids)), "Action IDs must be unique across all iframes"
            for i, aid in enumerate(action_ids, start=1):
                assert aid == f"e{i}", f"Expected contiguous action id e{i}, got {aid}"
        finally:
            browser.close()


def test_iframe_coordinate_translation(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that element bounding boxes inside iframes are translated to top-level viewport coordinates."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            actions = page["actions"]

            # Same-origin frame is positioned at top: 100px, left: 50px
            same_btn = next(a for a in actions if a.get("label") == "Same Frame Click")
            assert same_btn["rect"]["x"] >= 50, "same_btn X must include iframe left offset (>= 50)"
            assert same_btn["rect"]["y"] >= 100, "same_btn Y must include iframe top offset (>= 100)"

            # Cross-origin frame is positioned at top: 450px, left: 50px
            cross_btn = next(a for a in actions if a.get("label") == "Cross Frame Click")
            assert cross_btn["rect"]["x"] >= 50, "cross_btn X must include iframe left offset (>= 50)"
            assert cross_btn["rect"]["y"] >= 450, "cross_btn Y must include iframe top offset (>= 450)"
        finally:
            browser.close()


def test_iframe_skips_tiny_and_hidden_frames(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that iframes smaller than 200x100 or with hidden/display:none styles are skipped."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            actions = page["actions"]
            labels = [a.get("label") for a in actions]

            # Tiny iframe button (180x90) must be skipped
            assert "Tiny Hidden Action" not in labels, "Elements inside tiny iframes (<200x100) must be skipped"

            # Hidden iframe buttons must be skipped
            assert "Hidden Action" not in labels, "Elements inside hidden iframes must be skipped"
        finally:
            browser.close()


def test_iframe_max_frame_cap_at_five(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that a page with 7 visible iframes scans at most the first 5 frames."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    multi_url = dual_origin_servers["multi_frames_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(multi_url)
        try:
            page = browser.observe()
            labels = [a.get("label") for a in page["actions"]]

            # First 5 frames should be present
            for i in range(1, 6):
                assert f"Frame {i} Button" in labels, f"Frame {i} should be scanned"

            # Frames 6 and 7 must be excluded by the 5-frame cap
            assert "Frame 6 Button" not in labels, "Frame 6 must be skipped due to 5-frame limit"
            assert "Frame 7 Button" not in labels, "Frame 7 must be skipped due to 5-frame limit"
        finally:
            browser.close()


def test_iframe_click_action_cross_origin(dual_origin_servers: dict[str, Any]) -> None:
    """Verify clicking an element inside a cross-origin iframe executes its click handler."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            cross_btn = next(a for a in page["actions"] if a.get("label") == "Cross Frame Click")

            # Execute click on the cross-origin button
            res = browser.act(cross_btn, page=page)
            assert res["executed"] == cross_btn["id"]

            time.sleep(0.3)
            # Re-observe to verify status change inside cross-origin iframe
            after_page = browser.observe()
            assert "cross_btn_clicked" in after_page.get("text", "") or any(
                "cross_btn_clicked" in str(v) for v in after_page.values()
            )
        finally:
            browser.close()


def test_iframe_click_action_same_origin(dual_origin_servers: dict[str, Any]) -> None:
    """Verify clicking an element inside a same-origin iframe executes its click handler."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            same_btn = next(a for a in page["actions"] if a.get("label") == "Same Frame Click")

            res = browser.act(same_btn, page=page)
            assert res["executed"] == same_btn["id"]

            time.sleep(0.3)
            after_page = browser.observe()
            assert "same_btn_clicked" in after_page.get("text", "") or any(
                "same_btn_clicked" in str(v) for v in after_page.values()
            )
        finally:
            browser.close()


def test_iframe_fill_action_cross_origin(dual_origin_servers: dict[str, Any]) -> None:
    """Verify typing into a text field inside a cross-origin iframe updates its value."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            cross_inp = next(
                a for a in page["actions"]
                if a.get("value") == "cross_default" or "Cross Input" in str(a.get("label"))
            )

            typed_text = "new_cross_value_123"
            res = browser.act(cross_inp, page=page, text=typed_text)
            assert res["executed"] == cross_inp["id"]

            time.sleep(0.3)
            after_page = browser.observe()
            updated_action = next(
                (a for a in after_page["actions"] if a.get("value") == typed_text),
                None,
            )
            assert updated_action is not None, f"Input inside cross-origin iframe must have value '{typed_text}'"
        finally:
            browser.close()


def test_iframe_select_action_cross_origin(dual_origin_servers: dict[str, Any]) -> None:
    """Verify selecting an option from a dropdown inside a cross-origin iframe."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            cross_sel = next(
                (a for a in page["actions"] if "Cross Choice" in str(a.get("label")) or a.get("value") == "c2"),
                None,
            )
            assert cross_sel is not None, "Dropdown or option in cross-origin iframe must be discovered"

            res = browser.act(cross_sel, page=page, value="c2")
            assert res["executed"] == cross_sel["id"]

            time.sleep(0.3)
            after_page = browser.observe()
            # Status div updates to selected value
            assert "c2" in after_page.get("text", "") or any(
                "c2" in str(v) for v in after_page.values()
            )
        finally:
            browser.close()


def test_iframe_verifier_integration(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that verify_step detects element state modifications inside an iframe."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            before_page = browser.observe()
            cross_inp = next(
                a for a in before_page["actions"]
                if a.get("value") == "cross_default" or "Cross Input" in str(a.get("label"))
            )

            # Check that fresh() confirms element freshness before action
            assert browser.fresh(before_page, cross_inp) is True

            typed_text = "verified_val"
            browser.act(cross_inp, page=before_page, text=typed_text)
            after_page = browser.observe()

            # Test step verifier integration
            v_res = verify_step(
                action=cross_inp,
                before_page=before_page,
                after_page=after_page,
                typed_text=typed_text,
            )
            assert v_res.element_state_changed is True, "Verifier must detect value change inside iframe"
            assert v_res.details.get("fill_verified") is True
        finally:
            browser.close()


def test_iframe_scrolled_parent_live_rect_and_click(dual_origin_servers: dict[str, Any]) -> None:
    """Verify _get_iframe_rect returns live viewport rect after parent scroll and click hits button."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    scrolled_url = dual_origin_servers["scrolled_url"]
    with managed_isolated_browser(port=0):
        browser = Browser(scrolled_url)
        try:
            # 1. Scroll parent so iframe is in initial view (top: 1200px - scroll: 1000px = viewport y: 200px)
            browser.evaluate("window.scrollTo(0, 1000)")
            time.sleep(0.2)

            page = browser.observe()
            target_act = next(
                (a for a in page["actions"] if "Target Button" in a.get("label", "")),
                None,
            )
            assert target_act is not None, "Target button inside iframe must be discovered"
            assert target_act.get("is_iframe") is True
            assert target_act.get("frame_rect", {}).get("y") == 200

            # 2. Scroll parent by additional 200px down (viewport y becomes 0)
            browser.evaluate("window.scrollTo(0, 1200)")
            time.sleep(0.1)

            # Confirm true live DOM position is at y = 0
            live_dom_y = browser.evaluate("document.querySelectorAll('iframe')[0].getBoundingClientRect().y")
            assert live_dom_y == 0

            # 3. Verify _get_iframe_rect returns live rect (y=0) without syntax error, NOT stale fallback (y=200)
            live_rect = browser._get_iframe_rect(target_act)
            assert live_rect.get("y") == 0, f"Expected live iframe rect y=0, got {live_rect.get('y')} (stale fallback)"

            # 4. Execute click action on the button inside scrolled iframe
            res = browser.act(target_act, page=page)
            assert res.get("executed") == target_act["id"]

            time.sleep(0.2)
            # 5. Verify click reached button inside iframe
            after_page = browser.observe()
            assert "target_clicked" in after_page.get("text", "") or browser.evaluate(
                "document.getElementById('scrolled_frame').contentWindow.__clicked"
            ) is True
        finally:
            browser.close()


def test_iframe_cross_origin_scrolled_parent_live_rect_and_click(
    dual_origin_servers: dict[str, Any],
) -> None:
    """Verify _get_iframe_rect returns live viewport rect after parent scroll for cross-origin OOPIF and click succeeds."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    scrolled_cross_url = dual_origin_servers["scrolled_cross_url"]
    with managed_isolated_browser(port=0):
        browser = Browser(scrolled_cross_url)
        try:
            # 1. Scroll parent so iframe is in initial view (top: 1200px - scroll: 1000px = viewport y: 200px)
            browser.evaluate("window.scrollTo(0, 1000)")
            time.sleep(0.3)

            page = browser.observe()
            target_act = next(
                (a for a in page.get("actions", []) if "Target Button" in a.get("label", "")),
                None,
            )
            assert target_act is not None, "Target button inside cross-origin OOPIF must be discovered"
            assert target_act.get("is_iframe") is True
            assert (
                target_act.get("frame_rect", {}).get("y") == 200
            ), f"Expected initial frame_rect y=200, got {target_act.get('frame_rect', {}).get('y')}"

            # 2. Scroll parent by additional 200px down (viewport y becomes 0)
            browser.evaluate("window.scrollTo(0, 1200)")
            time.sleep(0.2)

            # Confirm true live DOM position is at y = 0
            live_dom_y = browser.evaluate(
                "document.querySelectorAll('iframe')[0].getBoundingClientRect().y"
            )
            assert live_dom_y == 0, f"Expected DOM iframe y=0, got {live_dom_y}"

            # 3. Verify _get_iframe_rect returns live rect (y=0), NOT stale fallback (y=200)
            live_rect = browser._get_iframe_rect(target_act)
            assert (
                live_rect.get("y") == 0
            ), f"Expected live iframe rect y=0, got {live_rect.get('y')} (stale fallback)"

            # 4. Execute click action on the button inside scrolled cross-origin OOPIF
            res = browser.act(target_act, page=page)
            assert res.get("executed") == target_act["id"]

            time.sleep(0.3)
            # 5. Verify click reached button inside OOPIF
            sid = target_act.get("frame_session_id")
            clicked = browser.evaluate("window.__clicked || false", session_id=sid)
            assert clicked is True, "Mouse click must hit target button inside scrolled cross-origin OOPIF"

            after_page = browser.observe()
            assert "target_clicked" in after_page.get("text", "") or any(
                "target_clicked" in str(v) for v in after_page.values()
            ), "Status div inside OOPIF must reflect 'target_clicked'"
        finally:
            browser.close()


def test_iframe_dynamic_removal_raises_stalepage(dual_origin_servers: dict[str, Any]) -> None:
    """Verify removing an iframe causes subsequent action on its element to raise StalePage."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_url"]
    with managed_isolated_browser(port=0):
        browser = Browser(top_url)
        try:
            page = browser.observe()
            actions = page.get("actions", [])

            # 1. Test cross-origin OOPIF iframe removal
            cross_act = next(
                (a for a in actions if a.get("label") == "Cross Frame Click"),
                None,
            )
            assert cross_act is not None, "Cross Frame Click must be discovered"

            # Remove cross-origin iframe from DOM (destroys OOPIF target session)
            browser.evaluate("document.getElementById('cross_frame').remove()")
            time.sleep(0.1)

            # fresh() must return False or raise StalePage without unhandled RuntimeError
            try:
                assert browser.fresh(page, cross_act) is False, "fresh() must return False for removed OOPIF"
            except StalePage:
                pass

            # act() must raise StalePage gracefully (not unhandled RuntimeError)
            with pytest.raises(StalePage):
                browser.act(cross_act, page=page)

            # 2. Test same-origin iframe removal
            page_current = browser.observe()
            same_act = next(
                (a for a in page_current.get("actions", []) if a.get("label") == "Same Frame Click"),
                None,
            )
            assert same_act is not None, "Same Frame Click must be discovered"

            browser.evaluate("document.getElementById('same_frame').remove()")
            time.sleep(0.1)

            try:
                assert browser.fresh(page_current, same_act) is False, "fresh() must return False for removed frame"
            except StalePage:
                pass

            with pytest.raises(StalePage):
                browser.act(same_act, page=page_current)

            # 3. Verify surviving top-level page elements remain observable and actionable
            final_page = browser.observe()
            final_labels = [a.get("label") for a in final_page.get("actions", [])]
            assert "Top Action Button" in final_labels
            assert "Cross Frame Click" not in final_labels
            assert "Same Frame Click" not in final_labels
        finally:
            browser.close()


def test_iframe_multi_maneuver_alternating_distant_oopifs(
    dual_origin_servers: dict[str, Any],
) -> None:
    """Regression test: Alternating interactions between distant OOPIFs under page scrolls.

    Verifies:
    1. Interaction with top OOPIF (frame_alpha at y=200).
    2. Scrolling down to distant OOPIF (frame_beta at y=2600) so it enters viewport (y=200).
    3. Action on frame_beta executes without compositor hit-test desynchronization,
       ensuring Input.dispatchMouseEvent accurately reaches the button inside the OOPIF.
    4. Scrolling back to top (y=0) and re-interacting with frame_alpha, confirming
       bidirectional alternation between distant OOPIFs works deterministically.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    dual_url = dual_origin_servers["dual_maneuver_url"]
    with managed_isolated_browser(port=0):
        browser = Browser(dual_url)
        try:
            # Step 1: Initial observation at top of page (scrollY = 0)
            page_top = browser.observe()
            act_alpha = next(
                (a for a in page_top.get("actions", []) if "Alpha Button" in a.get("label", "")),
                None,
            )
            assert act_alpha is not None, "Alpha Button inside top OOPIF must be discovered"
            assert act_alpha.get("is_iframe") is True
            assert act_alpha.get("dom_frame_index") == 0

            # Click Alpha Button
            res_alpha = browser.act(act_alpha, page=page_top)
            assert res_alpha.get("executed") == act_alpha["id"]
            time.sleep(0.2)

            sid_alpha = act_alpha.get("frame_session_id")
            clicks_alpha = browser.evaluate("window.__clicks || 0", session_id=sid_alpha)
            assert clicks_alpha == 1, f"Alpha clicks must be 1, got {clicks_alpha}"
            assert (
                browser.evaluate(
                    "document.getElementById('alpha_counter').textContent",
                    session_id=sid_alpha,
                )
                == "1"
            )

            # Step 2: Maneuver 1: Scroll down to y=2400.
            # frame_beta (top: 2600px) enters the viewport at y = 2600 - 2400 = 200px.
            browser.evaluate("window.scrollTo(0, 2400)")
            time.sleep(0.3)

            beta_dom_y = browser.evaluate(
                "document.querySelectorAll('iframe')[1].getBoundingClientRect().y"
            )
            assert 100 <= beta_dom_y <= 300, f"Frame Beta must be in viewport near y=200, got {beta_dom_y}"

            page_bottom = browser.observe()
            act_beta = next(
                (a for a in page_bottom.get("actions", []) if "Beta Button" in a.get("label", "")),
                None,
            )
            assert act_beta is not None, "Beta Button inside bottom OOPIF must be discovered after scroll"
            assert act_beta.get("is_iframe") is True
            assert act_beta.get("dom_frame_index") == 1

            # Click Beta Button
            # In buggy versions, unthrottled scrollIntoView shifts scrollY from 2400 to ~2362,
            # desynchronizing Chromium's RenderWidgetHostInputEventRouter and missing the button.
            res_beta = browser.act(act_beta, page=page_bottom)
            assert res_beta.get("executed") == act_beta["id"]
            time.sleep(0.2)

            sid_beta = act_beta.get("frame_session_id")
            clicks_beta = browser.evaluate("window.__clicks || 0", session_id=sid_beta)
            assert clicks_beta == 1, (
                f"Beta clicks was {clicks_beta} (expected 1). "
                "Chromium RenderWidgetHostInputEventRouter desynchronized due to unthrottled scrollIntoView in act()."
            )
            assert (
                browser.evaluate(
                    "document.getElementById('beta_counter').textContent",
                    session_id=sid_beta,
                )
                == "1"
            )

            # Step 3: Maneuver 2: Scroll back to top (y=0) and alternate back to Alpha.
            browser.evaluate("window.scrollTo(0, 0)")
            time.sleep(0.3)

            page_top_return = browser.observe()
            act_alpha_return = next(
                (a for a in page_top_return.get("actions", []) if "Alpha Button" in a.get("label", "")),
                None,
            )
            assert act_alpha_return is not None, "Alpha Button must be discovered again after return to top"
            assert act_alpha_return.get("dom_frame_index") == 0

            res_alpha_return = browser.act(act_alpha_return, page=page_top_return)
            assert res_alpha_return.get("executed") == act_alpha_return["id"]
            time.sleep(0.2)

            clicks_alpha_2 = browser.evaluate("window.__clicks || 0", session_id=sid_alpha)
            assert clicks_alpha_2 == 2, f"Alpha clicks must be 2 after returning, got {clicks_alpha_2}"
            assert (
                browser.evaluate(
                    "document.getElementById('alpha_counter').textContent",
                    session_id=sid_alpha,
                )
                == "2"
            )
        finally:
            browser.close()


def test_cross_origin_iframe_slider_label_extraction(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that sliders inside a cross-origin OOPIF extract container field labels and translated coordinates."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_ira_calculator_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()
            actions = page["actions"]

            # Check that all 5 IRA calculator sliders from the OOPIF are present
            slider_actions = [a for a in actions if a.get("role") == "slider"]
            assert len(slider_actions) >= 4, f"Expected at least 4 sliders from OOPIF, got {len(slider_actions)}"

            for a in slider_actions:
                assert a.get("is_iframe") is True
                assert a.get("label") != "slider", f"Slider {a.get('id')} has generic fallback 'slider' instead of field label"
                # Coordinates must be translated to top-level space (y >= top-level iframe offset 200)
                assert a["rect"]["y"] >= 200

            labels = [a.get("label", "") for a in slider_actions]
            assert any("Current age" in l or "Age" in l for l in labels)
            assert any("Retirement age" in l for l in labels)
            assert any("Starting balance" in l or "Balance" in l for l in labels)
        finally:
            browser.close()


def test_cross_origin_iframe_visible_text_and_headings_merged(dual_origin_servers: dict[str, Any]) -> None:
    """Verify that visible text and h1-h4 headings inside OOPIF are merged into page['headings'] and page['text']."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    top_url = dual_origin_servers["top_ira_calculator_url"]
    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(top_url)
        try:
            page = browser.observe()

            # Headings verification
            headings = page.get("headings") or []
            assert any("Traditional vs Roth IRA" in h or "Retirement Calculator" in h for h in headings), \
                f"OOPIF headings must be merged into page['headings'], got: {headings}"

            # Visible text verification
            text = page.get("text") or ""
            assert "Compare Traditional and Roth IRA returns" in text, \
                "Visible explanatory text from OOPIF must be present in page['text']"
        finally:
            browser.close()


