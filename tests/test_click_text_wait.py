"""After a click, settle in the page, then poll text only if it grows."""

from __future__ import annotations

import json
import shutil
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from reflex.browser.cdp import (
    _PAGE_TEXT_LEN,
    CLICK_SETTLE_BUDGET_S,
    CLICK_TEXT_CAP_S,
    CLICK_TEXT_QUIET_S,
    READ_STATE,
    Browser,
)


def _page(text: str = "hello") -> dict[str, Any]:
    return {
        "url": "https://example.test/chat",
        "title": "Chat",
        "text": text,
        "actions": [],
        "scroll": {"y": 0, "height": 800},
    }


def _browser(lengths: list[int], sleeps: list[float], clock: dict[str, float]) -> tuple[Browser, list[str]]:
    expressions: list[str] = []
    remaining = list(lengths)

    def monotonic() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["t"] += seconds

    def evaluate(expression: str) -> Any:
        expressions.append(expression)
        if expression == _PAGE_TEXT_LEN:
            if remaining:
                return remaining.pop(0)
            return lengths[-1] if lengths else 0
        return _page()

    def call(method: str, **kwargs: Any) -> dict[str, Any]:
        expressions.append(str(kwargs.get("expression") or method))
        return {}

    browser = Browser.__new__(Browser)
    browser.after_input = {"kind": "click", "id": "send", "node": 4}
    browser.last_action_rect = None
    browser.last_jpeg_bytes = b"frame"
    browser.evaluate = evaluate  # type: ignore[method-assign]
    browser.call = call  # type: ignore[method-assign]
    browser._monotonic = monotonic  # type: ignore[attr-defined]
    browser._sleep = sleep  # type: ignore[attr-defined]
    return browser, expressions


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    clock = {"t": 50.0}

    def monotonic() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock.setdefault("sleeps", []).append(seconds)  # type: ignore[attr-defined]
        clock["t"] += seconds

    monkeypatch.setattr("reflex.browser.cdp.time.monotonic", monotonic)
    monkeypatch.setattr("reflex.browser.cdp.time.sleep", sleep)
    return clock


def _bind_sleep(fake_clock: dict[str, float], sleeps: list[float]) -> None:
    from reflex.browser import cdp

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock["t"] += seconds

    cdp.time.sleep = sleep  # type: ignore[method-assign]


def test_static_click_wait_median_is_at_most_200ms(fake_clock: dict[str, float]) -> None:
    samples: list[float] = []
    settle_calls = 0
    for _ in range(7):
        sleeps: list[float] = []
        _bind_sleep(fake_clock, sleeps)
        browser, _expressions = _browser([40], sleeps, fake_clock)

        def call(method: str, **kwargs: Any) -> dict[str, Any]:
            nonlocal settle_calls
            if method == "Runtime.evaluate" and kwargs.get("awaitPromise") is True:
                settle_calls += 1
            return {}

        browser.call = call  # type: ignore[method-assign]
        browser.observe(screenshot=False)
        samples.append(sum(sleeps))
    median = statistics.median(samples)
    assert median <= 0.200 + 1e-6, samples
    assert median <= CLICK_SETTLE_BUDGET_S + 1e-6
    assert all(sample < CLICK_TEXT_QUIET_S for sample in samples)
    assert settle_calls == 7


def test_growth_within_the_arm_window_waits_for_quiet(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    _bind_sleep(fake_clock, sleeps)
    browser, expressions = _browser([100, 100, 180, 260], sleeps, fake_clock)
    browser.observe(screenshot=False)
    waited = sum(sleeps)
    assert expressions.count(_PAGE_TEXT_LEN) >= 4
    assert waited >= CLICK_TEXT_QUIET_S
    assert waited < CLICK_TEXT_CAP_S


def test_growth_after_the_arm_window_does_not_poll(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    _bind_sleep(fake_clock, sleeps)
    reads = {"n": 0}

    def evaluate(expression: str) -> Any:
        if expression == _PAGE_TEXT_LEN:
            reads["n"] += 1
            return 10 if reads["n"] <= 8 else 500
        return _page("partial")

    browser, _expressions = _browser([10], sleeps, fake_clock)
    browser.evaluate = evaluate  # type: ignore[method-assign]
    page = browser.observe(screenshot=False)
    assert sum(sleeps) <= CLICK_SETTLE_BUDGET_S + 1e-6
    assert sum(sleeps) < CLICK_TEXT_QUIET_S
    assert page["text"] == "partial"


def test_streaming_text_is_fully_visible(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    _bind_sleep(fake_clock, sleeps)
    full = "x" * 80
    state = {"n": 0}

    def evaluate(expression: str) -> Any:
        if expression == _PAGE_TEXT_LEN:
            state["n"] = min(len(full), state["n"] + 16)
            return state["n"]
        return _page(full[: state["n"]])

    browser, _expressions = _browser([0], sleeps, fake_clock)
    browser.evaluate = evaluate  # type: ignore[method-assign]
    page = browser.observe(screenshot=False)
    assert page["text"] == full
    assert sum(sleeps) >= CLICK_TEXT_QUIET_S
    assert sum(sleeps) <= CLICK_TEXT_CAP_S


def test_growing_text_stops_at_ten_seconds(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    produced = {"n": 0}
    _bind_sleep(fake_clock, sleeps)

    def evaluate(expression: str) -> Any:
        if expression == _PAGE_TEXT_LEN:
            produced["n"] += 10
            return produced["n"]
        return _page("stream")

    browser = Browser.__new__(Browser)
    browser.after_input = {"kind": "click", "id": "send", "node": 4}
    browser.last_action_rect = None
    browser.last_jpeg_bytes = b"frame"
    browser.evaluate = evaluate  # type: ignore[method-assign]
    browser.call = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    browser.observe(screenshot=False)
    assert sum(sleeps) == pytest.approx(CLICK_TEXT_CAP_S)
    assert sum(sleeps) <= CLICK_TEXT_CAP_S


def test_fill_does_not_poll_page_text(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    _bind_sleep(fake_clock, sleeps)
    browser, expressions = _browser([10, 20, 30], sleeps, fake_clock)
    browser.after_input = {"kind": "fill", "node": 3, "text": "ada"}
    browser.observe(screenshot=False)
    assert expressions.count(_PAGE_TEXT_LEN) == 0
    assert sleeps == []


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port == 9333:
        return _free_port()
    return port


class _LivePage:
    """Minimal CDP session so the settle wait can run against a real document."""

    def __init__(self, websocket_url: str) -> None:
        from websockets.sync.client import connect

        self._ws = connect(websocket_url, open_timeout=5, max_size=8_000_000)
        self._next = 0

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self._next += 1
        message_id = self._next
        self._ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            message = json.loads(self._ws.recv(timeout=15))
            if message.get("id") != message_id:
                continue
            if message.get("error"):
                raise RuntimeError(str(message["error"]))
            result = message.get("result") or {}
            return result if isinstance(result, dict) else {"value": result}

    def evaluate(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            raise RuntimeError(str(result["exceptionDetails"])[:300])
        return result.get("result", {}).get("value")

    def close(self) -> None:
        self._ws.close()


def _browser_on(page: _LivePage) -> Browser:
    browser = Browser.__new__(Browser)
    browser.call = page.call  # type: ignore[method-assign]
    browser.evaluate = page.evaluate  # type: ignore[method-assign]
    browser.last_action_rect = None
    browser.last_jpeg_bytes = b"frame"
    browser._try_pierce_cloudflare = lambda _info: False  # type: ignore[method-assign]
    return browser


def test_static_fixture_click_settle_median(tmp_path: Path) -> None:
    binary = shutil.which("chromium") or shutil.which("google-chrome")
    if not binary:
        pytest.skip("chromium is required to time a static click")
    page_path = tmp_path / "static.html"
    page_path.write_text(
        "<!doctype html><meta charset=utf-8><title>static</title>"
        "<button style='font:16px sans-serif;min-height:32px'>Go</button><p>ready and still</p>",
        encoding="utf-8",
    )
    port = _free_port()
    profile = tmp_path / "profile"
    proc = subprocess.Popen(
        [
            binary,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--disable-gpu",
            "--remote-allow-origins=*",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    live: _LivePage | None = None
    try:
        deadline = time.time() + 8
        websocket_url = ""
        while time.time() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                    targets = json.load(response)
                found = next(item for item in targets if item.get("type") == "page" and item.get("webSocketDebuggerUrl"))
                websocket_url = str(found["webSocketDebuggerUrl"])
                break
            except (OSError, TimeoutError, ValueError, StopIteration, KeyError):
                time.sleep(0.05)
        assert websocket_url
        live = _LivePage(websocket_url)
        live.call("Runtime.enable")
        live.call("Page.enable")
        live.call("Page.navigate", url=page_path.resolve().as_uri())
        time.sleep(0.25)
        browser = _browser_on(live)
        real_evaluate = live.evaluate

        def evaluate(expression: str) -> Any:
            if expression == READ_STATE:
                return {
                    "url": page_path.resolve().as_uri(),
                    "title": "static",
                    "text": "ready and still",
                    "actions": [],
                    "scroll": {"y": 0, "height": 800},
                }
            return real_evaluate(expression)

        browser.evaluate = evaluate  # type: ignore[method-assign]
        samples: list[float] = []
        for _ in range(7):
            browser.after_input = {"kind": "click", "node": 1}
            started = time.perf_counter()
            browser.observe(screenshot=False)
            samples.append(time.perf_counter() - started)
        assert statistics.median(samples) <= 0.200, samples
    finally:
        if live is not None:
            live.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
