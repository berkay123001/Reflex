"""CDP Screencast streaming module (`reflex/browser/screencast.py`).

Streams live page frames via a dedicated CDP WebSocket connection without
contending with `Browser._cdp_lock` or using the `browser_harness` daemon.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any

import websockets

logger = logging.getLogger(__name__)


class Screencast:
    """Dedicated CDP WebSocket screencast client.

    Maintains a dedicated background worker thread with its own asyncio loop,
    discovering the tab's WebSocket debugger URL from http://127.0.0.1:{cdp_port}/json/list,
    issuing `Page.startScreencast`, receiving `Page.screencastFrame` events, and sending
    immediate `Page.screencastFrameAck` responses without acquiring `Browser._cdp_lock`.
    """

    def __init__(self, cdp_port: int, target_id: str | None = None) -> None:
        self.cdp_port = cdp_port
        self.target_id = target_id
        self.latest_frame: bytes | None = None
        self.seq: int = 0
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._current_ws: Any = None
        self._req_id: int = 0
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None and self._thread.is_alive())

    def update_target(self, target_id: str | None) -> None:
        """Update target ID and trigger reconnect if target has changed."""
        with self._cond:
            if self.target_id == target_id:
                return
            self.target_id = target_id
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._cancel_active)
            except RuntimeError as exc:
                logger.debug("Loop runtime error during update_target: %s", exc)

    def start(self) -> None:
        """Start the background screencast thread."""
        with self._cond:
            if self._running:
                return
            self._running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"Screencast-{self.cdp_port}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop screencast worker thread cleanly."""
        with self._cond:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            self._cond.notify_all()

        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._cancel_active)
            except RuntimeError as exc:
                logger.debug("Loop runtime error during stop: %s", exc)

        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() != thread:
            thread.join(timeout=timeout)
        self._thread = None

    def wait_frame(self, after_seq: int = -1, timeout: float = 0.5) -> tuple[int, bytes | None]:
        """Wait for a frame where seq > after_seq, or return (current_seq, latest_frame) on timeout."""
        with self._cond:
            if self.latest_frame is not None and self.seq > after_seq:
                return self.seq, self.latest_frame
            if timeout > 0:
                self._cond.wait(timeout=timeout)
            return self.seq, self.latest_frame

    def _fetch_ws_url(self) -> str | None:
        """Query http://127.0.0.1:{cdp_port}/json/list to find the target's webSocketDebuggerUrl."""
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.cdp_port}/json/list",
                headers={"User-Agent": "Reflex-Screencast"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = resp.read().decode("utf-8")
                targets = json.loads(data)
            if not isinstance(targets, list):
                return None
            target_id = self.target_id
            if target_id:
                for item in targets:
                    if item.get("id") == target_id and item.get("webSocketDebuggerUrl"):
                        return str(item.get("webSocketDebuggerUrl"))
            for item in targets:
                if item.get("type") == "page" and item.get("webSocketDebuggerUrl"):
                    return str(item.get("webSocketDebuggerUrl"))
            for item in targets:
                if item.get("webSocketDebuggerUrl"):
                    return str(item.get("webSocketDebuggerUrl"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.debug("Failed fetching CDP target list: %s", exc)
            return None
        return None

    def _cancel_active(self) -> None:
        if self._current_ws is not None:
            try:
                asyncio.create_task(self._current_ws.close())
            except (RuntimeError, websockets.ConnectionClosed) as exc:
                logger.debug("Error requesting socket close: %s", exc)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._worker_main())
        except asyncio.CancelledError:
            logger.debug("Screencast main task cancelled")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Screencast worker loop terminated: %s", exc)
        finally:
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for t in tasks:
                t.cancel()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()
            self._loop = None

    async def _worker_main(self) -> None:
        backoff = 0.1
        while self._running and not self._stop_event.is_set():
            ws_url = await asyncio.to_thread(self._fetch_ws_url)
            if not ws_url:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 1.0)
                continue

            try:
                async with websockets.connect(
                    ws_url,
                    max_size=16 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self._current_ws = ws
                    backoff = 0.1

                    self._req_id += 1
                    start_cmd = {
                        "id": self._req_id,
                        "method": "Page.startScreencast",
                        "params": {
                            "format": "jpeg",
                            "quality": 60,
                            "maxWidth": 1120,
                            "maxHeight": 780,
                            "everyNthFrame": 1,
                        },
                    }
                    await ws.send(json.dumps(start_cmd))

                    while self._running and not self._stop_event.is_set():
                        try:
                            msg_raw = await ws.recv()
                        except (websockets.ConnectionClosed, asyncio.CancelledError):
                            break

                        try:
                            msg = json.loads(msg_raw)
                        except json.JSONDecodeError as exc:
                            logger.debug("Skipping unparseable CDP message: %s", exc)
                            continue

                        if msg.get("method") == "Page.screencastFrame":
                            params = msg.get("params", {})
                            data_b64 = params.get("data")
                            if data_b64:
                                try:
                                    frame_bytes = base64.b64decode(data_b64)
                                    with self._cond:
                                        self.latest_frame = frame_bytes
                                        self.seq += 1
                                        self._cond.notify_all()
                                except (ValueError, binascii.Error) as exc:
                                    logger.debug("Failed decoding base64 screencast frame: %s", exc)

                            session_id = params.get("sessionId")
                            self._req_id += 1
                            ack_cmd = {
                                "id": self._req_id,
                                "method": "Page.screencastFrameAck",
                                "params": {"sessionId": session_id},
                            }
                            try:
                                await ws.send(json.dumps(ack_cmd))
                            except (websockets.ConnectionClosed, OSError) as exc:
                                logger.debug("Failed sending screencast frame ack: %s", exc)
                                break
            except (websockets.ConnectionClosed, OSError, asyncio.CancelledError) as exc:
                logger.debug("Screencast connection closed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Screencast socket exception: %s", exc)
            finally:
                self._current_ws = None

            if self._running and not self._stop_event.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 1.0)
