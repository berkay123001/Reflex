"""Tests for fluid screencast via dedicated CDP WebSocket connection (Milestone G4 / Requirement R7)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import MagicMock

import websockets

import reflex.server as server_module
from reflex.browser.cdp import Browser
from reflex.browser.isolated import find_free_port, managed_isolated_browser
from reflex.browser.screencast import Screencast

logger = logging.getLogger(__name__)


def test_browser_stores_cdp_port() -> None:
    """Verify Browser stores self.cdp_port."""
    with managed_isolated_browser(port=0):
        browser = Browser("about:blank")
        try:
            assert hasattr(browser, "cdp_port"), "Browser must expose self.cdp_port"
            assert isinstance(browser.cdp_port, int)
            assert browser.cdp_port > 0
            assert browser.cdp_port != 9333
        finally:
            browser.close()


def test_mock_cdp_screencast_flow() -> None:
    """Test 1: Mock CDP server emits screencastFrame, checks screencastFrameAck, and verifies wait_frame."""
    received_commands: list[dict[str, Any]] = []
    ack_received = threading.Event()
    start_screencast_received = threading.Event()
    stop_ws_server = asyncio.Event()

    frame1_raw = b"\xff\xd8\xff\xe0mock_jpeg_frame_1"
    frame1_b64 = base64.b64encode(frame1_raw).decode("ascii")

    frame2_raw = b"\xff\xd8\xff\xe0mock_jpeg_frame_2"
    frame2_b64 = base64.b64encode(frame2_raw).decode("ascii")

    async def ws_handler(websocket: Any) -> None:
        async for message in websocket:
            data = json.loads(message)
            received_commands.append(data)
            method = data.get("method")
            if method == "Page.startScreencast":
                start_screencast_received.set()
                # Send first frame
                frame1_msg = {
                    "method": "Page.screencastFrame",
                    "params": {
                        "data": frame1_b64,
                        "sessionId": 42,
                        "metadata": {"timestamp": time.time()},
                    },
                }
                await websocket.send(json.dumps(frame1_msg))
            elif method == "Page.screencastFrameAck":
                ack_received.set()
                # If first ack received, send second frame
                if data.get("params", {}).get("sessionId") == 42:
                    frame2_msg = {
                        "method": "Page.screencastFrame",
                        "params": {
                            "data": frame2_b64,
                            "sessionId": 43,
                            "metadata": {"timestamp": time.time()},
                        },
                    }
                    await websocket.send(json.dumps(frame2_msg))

    ws_port = find_free_port()

    async def run_ws() -> None:
        async with websockets.serve(ws_handler, "127.0.0.1", ws_port):
            await stop_ws_server.wait()

    ws_loop = asyncio.new_event_loop()
    ws_thread = threading.Thread(
        target=lambda: ws_loop.run_until_complete(run_ws()),
        daemon=True,
    )
    ws_thread.start()

    # Create mock HTTP /json/list server
    class MockHttpHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/json/list":
                body = json.dumps([
                    {
                        "id": "mock_target_1",
                        "type": "page",
                        "webSocketDebuggerUrl": f"ws://127.0.0.1:{ws_port}/devtools/page/mock_target_1",
                    }
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *_args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockHttpHandler)
    http_port = httpd.server_address[1]
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    try:
        # Start Screencast pointing at mock CDP
        sc = Screencast(cdp_port=http_port, target_id="mock_target_1")
        sc.start()

        # Wait for Page.startScreencast
        assert start_screencast_received.wait(timeout=2.0), "Page.startScreencast must be received by mock server"

        # Verify first frame returned by wait_frame
        seq1, frame1 = sc.wait_frame(-1, timeout=2.0)
        assert seq1 >= 1
        assert frame1 == frame1_raw

        # Wait for Page.screencastFrameAck
        assert ack_received.wait(timeout=2.0), "Page.screencastFrameAck must be received by mock server"
        ack_cmds = [c for c in received_commands if c.get("method") == "Page.screencastFrameAck"]
        assert len(ack_cmds) >= 1
        assert ack_cmds[0]["params"]["sessionId"] == 42

        # Verify second frame arrived
        seq2, frame2 = sc.wait_frame(seq1, timeout=2.0)
        assert seq2 > seq1
        assert frame2 == frame2_raw

        sc.stop()
        assert not sc.is_running
    finally:
        httpd.shutdown()
        ws_loop.call_soon_threadsafe(stop_ws_server.set)
        ws_thread.join(timeout=1.0)


def test_screencast_connection_refused_and_server_fallback() -> None:
    """Verify fallback when CDP connection is refused or unavailable."""
    unused_port = find_free_port()
    sc = Screencast(cdp_port=unused_port, target_id="invalid_tab")
    sc.start()

    # wait_frame must not throw, but cleanly return (0, None) on timeout
    seq, frame = sc.wait_frame(-1, timeout=0.2)
    assert seq == 0
    assert frame is None
    sc.stop()

    # Verify server.py fallback logic when _SCREENCAST is None or fails
    mock_session = MagicMock()
    mock_browser = MagicMock()
    mock_browser.target = "tab1"
    mock_browser.capture_jpeg_bytes.return_value = b"fallback_jpeg_bytes"
    mock_session.browser = mock_browser

    server_module._SESSION = mock_session
    server_module._SCREENCAST = None

    # Simulate handler executing one stream iteration logic
    handler = server_module.ReflexHTTPHandler.__new__(server_module.ReflexHTTPHandler)
    mock_wfile = MagicMock()
    handler.wfile = mock_wfile
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    # Run one slice of _serve_mjpeg_stream with broken socket to terminate loop quickly
    mock_wfile.write.side_effect = BrokenPipeError("client disconnected")
    handler._serve_mjpeg_stream()

    # Browser.capture_jpeg_bytes should have been called as fallback
    mock_browser.capture_jpeg_bytes.assert_called()


def test_real_isolated_browser_screencast_fluidity_under_lock() -> None:
    """Test 2: Real isolated Chromium via managed_isolated_browser (NEVER port 9333).

    While Browser._cdp_lock is held for 2 seconds, verify >= 5 unique frames are
    captured, confirming fluidity vs old method which got 0 frames due to lock contention.
    """
    html_page = (
        "data:text/html,"
        "<!DOCTYPE html><html><body>"
        "<h1 id='c'>0</h1>"
        "<div id='b' style='width:120px;height:120px;background:red;'></div>"
        "<script>"
        "let c = 0;"
        "setInterval(() => {"
        "  c++;"
        "  document.getElementById('c').textContent = c;"
        "  document.getElementById('b').style.background = c % 2 === 0 ? 'blue' : 'green';"
        "}, 50);"
        "</script></body></html>"
    )

    with managed_isolated_browser(port=0) as cdp_url:
        port = int(cdp_url.split(":")[-1])
        assert port != 9333, "Must not use port 9333"

        browser = Browser(html_page)
        try:
            assert browser.cdp_port == port
            time.sleep(0.3)

            sc = Screencast(browser.cdp_port, browser.target)
            sc.start()

            init_seq, init_frame = sc.wait_frame(-1, timeout=2.0)
            assert init_seq >= 1, "Should receive initial screencast frame"
            assert init_frame is not None and len(init_frame) > 0

            lock_acquired = threading.Event()
            lock_release = threading.Event()

            def lock_holder() -> None:
                with browser._cdp_lock:
                    lock_acquired.set()
                    time.sleep(2.0)
                lock_release.set()

            holder_thread = threading.Thread(target=lock_holder, daemon=True)
            holder_thread.start()
            assert lock_acquired.wait(timeout=1.0)

            # Confirm old method gets 0 frames while lock is held (lock contention)
            old_result = browser.capture_jpeg_bytes()
            assert old_result is None, "Old capture_jpeg_bytes must fail to acquire lock and return None"

            # Collect frames via Screencast while lock is held
            t0 = time.time()
            collected_frames: list[bytes] = []
            cur_seq = init_seq
            while not lock_release.is_set():
                seq, frame = sc.wait_frame(cur_seq, timeout=0.1)
                if frame and seq > cur_seq:
                    cur_seq = seq
                    collected_frames.append(frame)

            duration = time.time() - t0
            holder_thread.join(timeout=1.0)

            unique_frames = len(set(collected_frames))
            fps = len(collected_frames) / duration if duration > 0 else 0

            logger.info(
                "Screencast lock test: %d frames (%d unique) in %.2fs (%.1f fps) while _cdp_lock was held.",
                len(collected_frames),
                unique_frames,
                duration,
                fps,
            )

            # Assert at least 5 different frames were captured while lock was held for 2s
            assert unique_frames >= 5, (
                f"Expected >= 5 unique frames while lock held, got {unique_frames} ({len(collected_frames)} total)"
            )
            assert fps >= 5.0, f"Expected fluid fps >= 5.0, got {fps:.1f} fps"

            sc.stop()
        finally:
            browser.close()
