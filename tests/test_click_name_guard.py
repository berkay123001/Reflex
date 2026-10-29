"""A click is refused when the live accessible name no longer matches the recorded raw name."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reflex.agent import ReflexSession
from reflex.browser.cdp import MARKER, Browser, StalePage
from reflex.config import ReflexConfig
from reflex.supervisor import UserMessageInterpretation
from tests.test_form_submit_gate import _NODE_RUNNER, _free_port

_SNAPSHOT_JS = Path(__file__).resolve().parent.parent / "reflex" / "browser" / "snapshot.js"
_RAW_NAME_HTML = """<!DOCTYPE html>
<html><body>
<article><h2>Quantum article</h2><a href="#a">Read more</a></article>
<article><h2>Other article</h2><a href="#b">Read more</a></article>
<div role="group"><h3>Weekly question</h3>
<label>Save (draft)<input type="radio" name="keep"></label></div>
</body></html>
"""


def _guard(node: int, name: str) -> list[Any]:
    return [node, "button", name, None, None, None, None, False, None, None, None, None, None, ""]


def _page(
    label: str,
    live_name: str,
    *,
    raw: str | None = None,
    node: int = 7,
    kind: str = "click",
) -> dict[str, Any]:
    guard = _guard(node, live_name)
    action: dict[str, Any] = {"id": "send", "node": node, "kind": kind, "role": "button", "label": label}
    if raw is not None:
        action["accessible_name"] = raw
    return {
        "url": "https://example.test/chat",
        "title": "Chat",
        "text": "hello",
        "fingerprint": "fp-a",
        "marker": "marker-a",
        "page_key": ["origin-a", "https://example.test/chat"],
        "guards": {str(node): guard, node: guard},
        "actions": [action],
    }


_HIT = {
    "x": 10,
    "y": 12,
    "rx": 2,
    "ry": 4,
    "rw": 40,
    "rh": 18,
    "jsClicked": False,
    "prevOrigin": 1,
}


def _scripted_evaluate(
    live_name: str,
    page: dict[str, Any],
    *,
    kind: str,
    node: int = 7,
    expect_geometry: bool,
) -> Any:
    """Answer evaluates in act() order. The marker read stays exact; later reads follow the call."""
    state = {"phase": 0}

    def evaluate(expression: str) -> Any:
        if expression == MARKER:
            return page.get("marker")
        state["phase"] += 1
        phase = state["phase"]
        if phase == 1:
            return [page["page_key"], _guard(node, live_name)]
        if kind == "click" and phase == 2:
            return live_name
        geometry_phase = 3 if kind == "click" else 2
        if expect_geometry and phase == geometry_phase:
            return dict(_HIT)
        if kind == "click":
            return live_name
        return None

    return evaluate


def _browser(
    live_name: str,
    page: dict[str, Any],
    *,
    expect_geometry: bool = True,
) -> tuple[Browser, list[tuple[str, dict[str, Any]]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    browser = Browser.__new__(Browser)
    browser.after_input = None
    browser.last_action_rect = None
    kind = str(page["actions"][0].get("kind") or "click")
    browser.evaluate = _scripted_evaluate(  # type: ignore[method-assign]
        live_name, page, kind=kind, node=int(page["actions"][0]["node"]), expect_geometry=expect_geometry
    )

    def call(method: str, **params: Any) -> dict[str, Any]:
        calls.append((method, params))
        return {}

    browser.call = call  # type: ignore[method-assign]
    return browser, calls


def _clicked(calls: list[tuple[str, dict[str, Any]]]) -> bool:
    return any(
        method == "Input.dispatchMouseEvent" and params.get("type") == "mousePressed" for method, params in calls
    )


def test_changed_accessible_name_does_not_click() -> None:
    page = _page("Send message", "Stop reply", raw="Send message")
    browser, calls = _browser("Stop reply", page, expect_geometry=False)
    action = page["actions"][0]
    with pytest.raises(StalePage, match="Accessible name changed"):
        browser.act(action, page)
    assert not _clicked(calls)


@pytest.mark.parametrize(
    ("label", "raw", "live_name"),
    [
        ("Send message", "Send message", "Stop reply"),
        ("Save (draft) (Card title)", "Save (draft)", "Save"),
        ("Read more (Quantum (2026) notes)", "Read more (Quantum (2026) notes)", "Read more (Quantum"),
        ("Guide · Read the longer article · extra", "Guide · Read the longer article · extra", "extra"),
        ("Weekly question → Save (draft)", "Save (draft)", "Save"),
    ],
)
def test_renamed_raw_name_does_not_click(label: str, raw: str, live_name: str) -> None:
    page = _page(label, live_name, raw=raw)
    browser, calls = _browser(live_name, page, expect_geometry=False)
    with pytest.raises(StalePage, match="Accessible name changed"):
        browser.act(page["actions"][0], page)
    assert not _clicked(calls)


@pytest.mark.parametrize(
    ("label", "raw", "live_name"),
    [
        ("Send message", "Send message", "Send message"),
        ("Weekly question → Choice one", "Choice one", "Choice one"),
        (
            "Guide · Read the longer article title here",
            "Read the longer article title here",
            "Read the longer article title here",
        ),
        ("Open City", "City", "City"),
        ("Resend code 12s", "Resend code 12s", "Resend code 15s"),
        ("Send message", "Send message", ""),
        ("Read more (Quantum article)", "Read more", "Read more"),
        ("Save (draft) (Card title)", "Save (draft)", "Save (draft)"),
        ("Read more (Quantum (2026) notes)", "Read more", "Read more"),
        ("Weekly question → Save (draft)", "Save (draft)", "Save (draft)"),
        ("Open Save (draft)", "Save (draft)", "Save (draft)"),
        ("Heading · Read more (2026)", "Read more (2026)", "Read more (2026)"),
        ("Weekly question → Yes (Card title)", "Yes", "Yes"),
        ("Guide · Read more (Quantum (2026) notes)", "Read more", "Read more"),
    ],
)
def test_matching_raw_name_still_clicks(label: str, raw: str, live_name: str) -> None:
    page = _page(label, live_name, raw=raw)
    browser, calls = _browser(live_name, page)
    result = browser.act(page["actions"][0], page)
    assert result["executed"] == "send"
    assert _clicked(calls)


def test_missing_raw_name_uses_guard_not_the_decorated_label() -> None:
    page = _page("Weekly question → Save (draft)", "Save (draft)")
    browser, calls = _browser("Save (draft)", page)
    result = browser.act(page["actions"][0], page)
    assert result["executed"] == "send"
    assert _clicked(calls)


def test_fill_does_not_compare_accessible_name() -> None:
    page = _page("City", "Other value", raw="City", kind="fill")
    browser, calls = _browser("Other value", page)
    result = browser.act(page["actions"][0], page, text="kept")
    assert result["executed"] == "send"
    assert ("Input.insertText", {"text": "kept"}) in calls


def test_session_observes_again_and_does_not_click_renamed_control() -> None:
    node = 7
    live_name = "Stop reply"
    original = _page("Send message", live_name, raw="Send message")
    renamed = {
        **original,
        "fingerprint": "fp-b",
        "marker": "marker-b",
        "actions": [{"id": "send", "node": node, "kind": "click", "role": "button", "label": live_name}],
    }
    pages = [original, renamed]
    calls: list[str] = []

    browser = Browser.__new__(Browser)
    browser.after_input = None
    browser.last_action_rect = None
    browser.last_jpeg_bytes = None

    def observe(screenshot: bool = True) -> dict[str, Any]:
        del screenshot
        return pages.pop(0)

    def call(method: str, **_params: Any) -> dict[str, Any]:
        calls.append(method)
        return {}

    browser.observe = observe  # type: ignore[method-assign]
    browser.evaluate = _scripted_evaluate(  # type: ignore[method-assign]
        live_name, original, kind="click", node=node, expect_geometry=False
    )
    browser.call = call  # type: ignore[method-assign]
    browser.close = lambda: None  # type: ignore[method-assign]

    def choose(page: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        del page
        return {
            "operation": "CLICK",
            "target": "1",
            "choice": "send",
            "confidence": 0.95,
            "target_confidence": 0.9,
            "latency_ms": 2,
            "model": "jev-test",
            "usage": {},
        }

    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", side_effect=choose),
    ):
        session = ReflexSession(
            url="https://example.test/chat",
            goal="Send the note",
            config=ReflexConfig(require_confirmation_for_destructive=False, verify_done_with_supervisor=False),
        )
        session.step()

    assert not pages
    assert session.page["actions"][0]["label"] == live_name
    assert session.history == []
    assert session.status == "ready"
    assert "Input.dispatchMouseEvent" not in calls


def test_session_does_not_reclick_when_refreshed_name_still_differs() -> None:
    node = 7
    label = "Read more (Quantum article)"
    live_name = "Stop reply"
    original = _page(label, live_name, raw="Read more")
    refreshed = {**original, "fingerprint": "fp-b", "marker": "marker-b"}
    pages = [original, refreshed]
    calls: list[str] = []
    act_calls = {"n": 0}

    browser = Browser.__new__(Browser)
    browser.after_input = None
    browser.last_action_rect = None
    browser.last_jpeg_bytes = None

    def observe(screenshot: bool = True) -> dict[str, Any]:
        del screenshot
        return pages.pop(0)

    def call(method: str, **_params: Any) -> dict[str, Any]:
        calls.append(method)
        return {}

    browser.observe = observe  # type: ignore[method-assign]
    browser.evaluate = _scripted_evaluate(  # type: ignore[method-assign]
        live_name, original, kind="click", node=node, expect_geometry=False
    )
    browser.call = call  # type: ignore[method-assign]
    browser.close = lambda: None  # type: ignore[method-assign]
    real_act = browser.act

    def counting_act(*args: Any, **kwargs: Any) -> dict[str, Any]:
        act_calls["n"] += 1
        return real_act(*args, **kwargs)

    browser.act = counting_act  # type: ignore[method-assign]

    def choose(page: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        del page
        return {
            "operation": "CLICK",
            "target": "1",
            "choice": "send",
            "confidence": 0.95,
            "target_confidence": 0.9,
            "latency_ms": 2,
            "model": "jev-test",
            "usage": {},
        }

    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", side_effect=choose),
    ):
        session = ReflexSession(
            url="https://example.test/chat",
            goal="Open the article",
            config=ReflexConfig(require_confirmation_for_destructive=False, verify_done_with_supervisor=False),
        )
        session.step()

    assert not pages
    assert session.page["actions"][0]["label"] == label
    assert session.history == []
    assert session.status == "ready"
    assert act_calls["n"] == 1
    assert session.pending_escalation_reason is None
    assert "Input.dispatchMouseEvent" not in calls


def test_snapshot_keeps_raw_accessible_name(tmp_path: Path) -> None:
    binary = shutil.which("chromium") or shutil.which("google-chrome")
    node = shutil.which("node")
    if not binary or not node:
        pytest.skip("chromium and node are required to execute snapshot.js")
    html_path = tmp_path / "names.html"
    html_path.write_text(_RAW_NAME_HTML, encoding="utf-8")
    profile = tmp_path / "profile"
    port = _free_port()
    proc = subprocess.Popen(
        [
            binary,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--remote-allow-origins=*",
            "--window-size=1400,8000",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        result = subprocess.run(
            [node, "-e", _NODE_RUNNER, str(port), html_path.resolve().as_uri(), str(_SNAPSHOT_JS)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert result.returncode == 0, result.stderr
    page = json.loads(result.stdout)
    actions = page["actions"]
    reads = [a for a in actions if a.get("accessible_name") == "Read more"]
    assert len(reads) >= 2
    assert all(a["label"] != "Read more" for a in reads)
    assert any(a["label"].startswith("Read more (") for a in reads)
    saves = [a for a in actions if a.get("accessible_name") == "Save (draft)"]
    assert len(saves) == 1
    assert saves[0]["label"].startswith("Weekly question → ")
    assert saves[0]["label"].endswith("Save (draft)")
