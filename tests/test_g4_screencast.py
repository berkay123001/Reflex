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


def test_screencast_idle_resends_frame_every_second() -> None:
    """Requirement R3 (G5c): When screencast is idle (no new frames), resend the last frame every ~1s."""
    mock_browser = MagicMock()
    mock_browser.target = "mock_tab_idle"
    mock_browser.capture_jpeg_bytes.return_value = b"fallback_jpeg_frame"
    mock_session = MagicMock()
    mock_session.browser = mock_browser

    orig_session = server_module._SESSION
    orig_screencast = server_module._SCREENCAST
    server_module._SESSION = mock_session

    class FakeIdleScreencast:
        def __init__(self) -> None:
            self.is_running = True
            self.seq = 1
            self.latest_frame = b"static_screencast_frame"

        def wait_frame(self, after_seq: int = -1, timeout: float = 0.1) -> tuple[int, bytes | None]:
            if after_seq < 1:
                return 1, self.latest_frame
            time.sleep(min(timeout, 0.05))
            return 1, self.latest_frame

    server_module._SCREENCAST = FakeIdleScreencast()

    written_frames: list[tuple[float, bytes]] = []
    t_start = time.monotonic()

    handler = server_module.ReflexHTTPHandler.__new__(server_module.ReflexHTTPHandler)
    mock_wfile = MagicMock()
    handler.wfile = mock_wfile
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    def fake_write(data: bytes) -> None:
        if data.startswith(b"--reflexframe") or data == b"\r\n":
            return
        now_rel = time.monotonic() - t_start
        written_frames.append((now_rel, data))
        # After collecting 3 identical frames (~2s total stream), terminate
        if len(written_frames) >= 3:
            raise BrokenPipeError("Collected sufficient heartbeat frames")

    mock_wfile.write.side_effect = fake_write

    try:
        handler._serve_mjpeg_stream()
    finally:
        server_module._SESSION = orig_session
        server_module._SCREENCAST = orig_screencast

    # Verify we got 3 frames of identical data
    assert len(written_frames) == 3
    assert all(f == b"static_screencast_frame" for _, f in written_frames)

    # Verify ~1s intervals between consecutive frame transmissions
    interval_1 = written_frames[1][0] - written_frames[0][0]
    interval_2 = written_frames[2][0] - written_frames[1][0]

    assert 0.85 <= interval_1 <= 1.35, f"Interval 1 was {interval_1:.2f}s, expected ~1.0s"
    assert 0.85 <= interval_2 <= 1.35, f"Interval 2 was {interval_2:.2f}s, expected ~1.0s"

    # capture_jpeg_bytes was NOT called because < 3s elapsed
    mock_browser.capture_jpeg_bytes.assert_not_called()


def test_screencast_3s_idle_invokes_capture_jpeg_bytes() -> None:
    """Requirement R3 (G5c): If screencast produces no frames for 3s, invoke capture_jpeg_bytes."""
    mock_browser = MagicMock()
    mock_browser.target = "mock_tab_3s"
    mock_browser.capture_jpeg_bytes.return_value = b"fresh_fallback_snapshot_3s"
    mock_session = MagicMock()
    mock_session.browser = mock_browser

    orig_session = server_module._SESSION
    orig_screencast = server_module._SCREENCAST
    server_module._SESSION = mock_session

    class FakeIdleScreencast3s:
        def __init__(self) -> None:
            self.is_running = True
            self.seq = 1
            self.latest_frame = b"old_screencast_frame"

        def wait_frame(self, after_seq: int = -1, timeout: float = 0.1) -> tuple[int, bytes | None]:
            if after_seq < 1:
                return 1, self.latest_frame
            time.sleep(min(timeout, 0.05))
            return 1, self.latest_frame

    server_module._SCREENCAST = FakeIdleScreencast3s()

    written_frames: list[tuple[float, bytes]] = []
    t_start = time.monotonic()

    handler = server_module.ReflexHTTPHandler.__new__(server_module.ReflexHTTPHandler)
    mock_wfile = MagicMock()
    handler.wfile = mock_wfile
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()

    def fake_write(data: bytes) -> None:
        if data.startswith(b"--reflexframe") or data == b"\r\n":
            return
        now_rel = time.monotonic() - t_start
        written_frames.append((now_rel, data))
        # Once the fallback snapshot arrives (at >= 3.0s), terminate
        if data == b"fresh_fallback_snapshot_3s":
            raise BrokenPipeError("Fallback snapshot received; terminating stream test")

    mock_wfile.write.side_effect = fake_write

    t0 = time.monotonic()
    try:
        handler._serve_mjpeg_stream()
    finally:
        server_module._SESSION = orig_session
        server_module._SCREENCAST = orig_screencast

    elapsed = time.monotonic() - t0

    # 1. capture_jpeg_bytes must have been called on mock_browser
    assert mock_browser.capture_jpeg_bytes.call_count >= 1, (
        f"Expected capture_jpeg_bytes to be called, but call_count={mock_browser.capture_jpeg_bytes.call_count}"
    )

    # 2. Verify fallback was not triggered prematurely (must be at least 3.0s, allow 2.9s margin)
    assert elapsed >= 2.9, f"Fallback occurred too early: {elapsed:.2f}s"

    # 3. Last frame sent must be the fresh snapshot from capture_jpeg_bytes
    assert written_frames[-1][1] == b"fresh_fallback_snapshot_3s"

    # 4. Frames sent before 3.0s must be the initial/heartbeat frames
    pre_fallback_frames = [f for _, f in written_frames[:-1]]
    assert len(pre_fallback_frames) >= 3  # initial (t=0), heartbeat 1 (t~1s), heartbeat 2 (t~2s)
    assert all(f == b"old_screencast_frame" for f in pre_fallback_frames)

