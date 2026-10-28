"""After a click, observation polls until page text has been quiet or the cap hits."""

from __future__ import annotations

from typing import Any

import pytest

from reflex.browser.cdp import CLICK_TEXT_CAP_S, CLICK_TEXT_QUIET_S, Browser, _PAGE_TEXT_LEN


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
        return {
            "url": "https://example.test/chat",
            "title": "Chat",
            "text": "hello",
            "actions": [],
            "scroll": {"y": 0, "height": 800},
        }

    browser = Browser.__new__(Browser)
    browser.after_input = {"kind": "click", "id": "send", "node": 4}
    browser.last_action_rect = None
    browser.last_jpeg_bytes = b"frame"
    browser.evaluate = evaluate  # type: ignore[method-assign]
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


def test_stable_text_waits_out_the_quiet_window(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock["t"] += seconds

    # Rebind the already patched sleep so this test records its own list.
    import reflex.browser.cdp as cdp

    cdp.time.sleep = sleep  # type: ignore[method-assign]
    browser, expressions = _browser([80, 80], sleeps, fake_clock)
    page = browser.observe(screenshot=False)
    waited = sum(sleeps)
    assert page["url"] == "https://example.test/chat"
    assert sleeps
    assert waited >= CLICK_TEXT_QUIET_S - 0.02
    assert waited < 2
    assert waited < CLICK_TEXT_CAP_S
    assert expressions.count(_PAGE_TEXT_LEN) > 2
    assert not any("MutationObserver" in expr for expr in expressions)


def test_equal_reads_still_wait_for_later_growth(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock["t"] += seconds

    import reflex.browser.cdp as cdp

    cdp.time.sleep = sleep  # type: ignore[method-assign]
    browser, expressions = _browser([100, 100, 180, 260], sleeps, fake_clock)
    browser.observe(screenshot=False)
    waited = sum(sleeps)
    reads = expressions.count(_PAGE_TEXT_LEN)
    assert reads >= 4
    assert sleeps
    assert waited >= CLICK_TEXT_QUIET_S
    assert waited < CLICK_TEXT_CAP_S
    assert not any("MutationObserver" in expr for expr in expressions)


def test_growing_text_waits_for_quiet_window(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock["t"] += seconds

    import reflex.browser.cdp as cdp

    cdp.time.sleep = sleep  # type: ignore[method-assign]
    browser, expressions = _browser([10, 40, 70, 70], sleeps, fake_clock)
    browser.observe(screenshot=False)
    waited = sum(sleeps)
    assert waited >= CLICK_TEXT_QUIET_S
    assert waited < CLICK_TEXT_CAP_S
    assert expressions.count(_PAGE_TEXT_LEN) > 2


def test_growing_text_stops_at_ten_seconds(fake_clock: dict[str, float]) -> None:
    sleeps: list[float] = []
    produced = {"n": 0}

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        fake_clock["t"] += seconds

    import reflex.browser.cdp as cdp

    cdp.time.sleep = sleep  # type: ignore[method-assign]

    def evaluate(expression: str) -> Any:
        if expression == _PAGE_TEXT_LEN:
            produced["n"] += 10
            return produced["n"]
        return {"url": "https://example.test/chat", "title": "Chat", "text": "stream", "actions": [], "scroll": {}}

    browser = Browser.__new__(Browser)
    browser.after_input = {"kind": "click", "id": "send", "node": 4}
    browser.last_action_rect = None
    browser.last_jpeg_bytes = b"frame"
    browser.evaluate = evaluate  # type: ignore[method-assign]
    browser.observe(screenshot=False)
    assert sum(sleeps) == pytest.approx(CLICK_TEXT_CAP_S)
    assert sum(sleeps) <= CLICK_TEXT_CAP_S
