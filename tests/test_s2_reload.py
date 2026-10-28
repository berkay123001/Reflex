"""System 2 can reload the current page and the next decision sees the new observation."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from reflex.agent import ReflexSession
from reflex.browser.cdp import Browser
from reflex.config import ReflexConfig
from reflex.supervisor import ESCALATION_SYSTEM_PROMPT, SupervisorDecision, UserMessageInterpretation


def test_s2_schema_lists_reload_without_dropping_existing_operations() -> None:
    assert '"RELOAD"' in ESCALATION_SYSTEM_PROMPT or "RELOAD" in ESCALATION_SYSTEM_PROMPT
    for operation in ("CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "GO_BACK", "GOTO_URL", "RELOAD"):
        assert operation in ESCALATION_SYSTEM_PROMPT


def test_reload_reloads_then_observes_before_the_next_decision() -> None:
    initial = {
        "url": "https://example.test/before",
        "title": "Before",
        "text": "old page",
        "fingerprint": "before",
        "actions": [{"id": "btn", "node": 1, "kind": "click", "role": "button", "label": "Go"}],
    }
    refreshed = {
        "url": "https://example.test/after",
        "title": "After",
        "text": "new page",
        "fingerprint": "after",
        "actions": [{"id": "btn", "node": 2, "kind": "click", "role": "button", "label": "Go"}],
    }
    pages = [initial, refreshed]
    order: list[str] = []

    browser = Browser.__new__(Browser)
    browser.after_input = None
    browser.last_action_rect = None
    browser.last_jpeg_bytes = None

    def call(method: str, **_params: Any) -> dict[str, Any]:
        order.append(method)
        return {}

    def observe(screenshot: bool = True) -> dict[str, Any]:
        order.append("observe")
        return pages.pop(0) if pages else refreshed

    browser.call = call  # type: ignore[method-assign]
    browser.evaluate = lambda _expr: 1  # type: ignore[method-assign]
    browser.observe = observe  # type: ignore[method-assign]
    browser.fresh = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    browser._wait_for_navigation_and_settle = lambda **_kwargs: None  # type: ignore[method-assign]
    browser.close = lambda: None  # type: ignore[method-assign]

    seen_urls: list[str] = []

    def choose(page: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        seen_urls.append(str(page.get("url")))
        if len(seen_urls) == 1:
            return {
                "operation": "CLICK",
                "target": "1",
                "choice": "btn",
                "confidence": 0.1,
                "target_confidence": 0.1,
                "latency_ms": 2,
                "model": "jev-test",
                "usage": {},
            }
        return {
            "operation": "DONE",
            "target": None,
            "choice": "DONE",
            "confidence": 0.99,
            "target_confidence": 1.0,
            "latency_ms": 2,
            "model": "jev-test",
            "usage": {},
        }

    sup = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="RELOAD",
        reasoning="reload and start again",
        model="supervisor-test",
        latency_ms=3,
    )
    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", side_effect=choose),
        patch("reflex.agent.resolve_escalation", return_value=sup),
    ):
        session = ReflexSession(
            url="https://example.test/before",
            goal="Restart the page",
            config=ReflexConfig(verify_done_with_supervisor=False),
        )
        session.step()
        session.step()

    assert order.index("Page.reload") < order.index("observe", order.index("Page.reload"))
    assert "Input.dispatchMouseEvent" not in order
    assert seen_urls == ["https://example.test/before", "https://example.test/after"]
    assert session.history[0]["kind"] == "reload"
    assert session.status == "done"
