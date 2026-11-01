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

from reflex.browser.cdp import Browser
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
