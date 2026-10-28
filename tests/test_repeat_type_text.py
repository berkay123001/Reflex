"""A repeated TYPE_TEXT on text that is still present skips the text helper and re-asks S1."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx

from reflex.agent import ReflexSession
from reflex.browser.cdp import Browser
from reflex.config import ReflexConfig
from reflex.supervisor import FieldTextOutcome, UserMessageInterpretation
from reflex.system1 import choose_action

_TYPED = "north bridge"


def _page(email_value: str) -> dict[str, Any]:
    return {
        "url": "https://example.test/form",
        "title": "Form",
        "text": "fields",
        "fingerprint": "fp",
        "actions": [
            {"id": "email", "node": 3, "kind": "fill", "role": "textbox", "label": "Email", "value": email_value},
            {"id": "note", "node": 4, "kind": "fill", "role": "textbox", "label": "Note", "value": ""},
            {"id": "go", "node": 5, "kind": "click", "role": "button", "label": "Continue"},
        ],
    }


def _decision(operation: str, target_key: str | None = None, choice: str | None = None) -> dict[str, Any]:
    probabilities = {"CLICK": 0.05, "TYPE_TEXT": 0.05, "DONE": 0.05, "BLOCKED": 0.05, operation: 0.8}
    answers: dict[str, Any] = {"operation": {"choice": operation, "probabilities": probabilities}}
    if target_key and choice:
        others = {"1": 0.1, "2": 0.1, "3": 0.1}
        others[choice] = 0.7
        answers[target_key] = {"choice": choice, "probabilities": others}
    return {"model": "jev-test", "answers": answers, "usage": {}}


def _type_labels(body: dict[str, Any]) -> list[str]:
    question = (body.get("questions") or {}).get("type_text_target") or {}
    criteria = question.get("criteria") or {}
    return [str(item.get("element") or "") for item in criteria.values()]


def _run(
    page: dict[str, Any],
    history: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> tuple[ReflexSession, list[dict[str, Any]], list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []
    helper_calls: list[dict[str, Any]] = []
    browser = Browser.__new__(Browser)
    browser.after_input = None
    browser.last_action_rect = None
    browser.last_jpeg_bytes = None
    browser.observe = lambda screenshot=True: page  # type: ignore[method-assign]
    browser.fresh = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    browser.act = lambda *_args, **_kwargs: {"executed": "ok"}  # type: ignore[method-assign]
    browser.close = lambda: None  # type: ignore[method-assign]

    def post(*_args: Any, **kwargs: Any) -> httpx.Response:
        bodies.append(kwargs.get("json") or {})
        payload = answers[len(bodies) - 1] if len(bodies) <= len(answers) else answers[-1]
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://127.0.0.1/s1"))

    def helper(**kwargs: Any) -> FieldTextOutcome:
        helper_calls.append(kwargs)
        return FieldTextOutcome(text="other text", model="helper-test")

    with (
        patch("reflex.agent.Browser", return_value=browser),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.system1._HTTP_CLIENT.post", post),
        patch("reflex.agent.generate_field_text", helper),
    ):
        session = ReflexSession(
            url="https://example.test/form",
            goal="Fill the empty note",
            config=ReflexConfig(decision_backend="local_turnike", verify_done_with_supervisor=False),
        )
        session.history = list(history)
        session.step()
    return session, bodies, helper_calls


def test_repeat_type_text_skips_helper_and_masks_that_field() -> None:
    history = [{"kind": "fill", "operation": "TYPE_TEXT", "node": 3, "action": "Email", "text": _TYPED}]
    answers = [
        _decision("TYPE_TEXT", "type_text_target", "1"),
        _decision("CLICK", "click_target", "3"),
    ]
    session, bodies, helper_calls = _run(_page(_TYPED), history, answers)
    assert helper_calls == []
    assert session.text_calls == []
    assert len(bodies) == 2
    assert any("Email" in label for label in _type_labels(bodies[0]))
    assert any("Note" in label for label in _type_labels(bodies[1]))
    assert not any("Email" in label for label in _type_labels(bodies[1]))
    assert "TYPE_TEXT" in bodies[1]["questions"]["operation"]["criteria"]


def test_masking_the_only_fill_removes_type_text_from_the_request() -> None:
    page = {
        "url": "https://example.test/form",
        "title": "Form",
        "text": "one field",
        "actions": [{"id": "email", "node": 3, "kind": "fill", "role": "textbox", "label": "Email", "value": _TYPED}],
    }
    bodies: list[dict[str, Any]] = []

    def post(*_args: Any, **kwargs: Any) -> httpx.Response:
        bodies.append(kwargs.get("json") or {})
        payload = {
            "model": "jev-test",
            "answers": {"operation": {"choice": "DONE", "probabilities": {"DONE": 0.9, "BLOCKED": 0.1}}},
            "usage": {},
        }
        return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://127.0.0.1/s1"))

    with patch("reflex.system1._HTTP_CLIENT.post", post):
        choose_action(page, "Finish", [], ReflexConfig(decision_backend="local_turnike"), mask_fill_nodes={3})
    criteria = bodies[0]["questions"]["operation"]["criteria"]
    assert "TYPE_TEXT" not in criteria
    assert "type_text_target" not in bodies[0]["questions"]


def test_changed_value_is_not_masked_and_helper_runs() -> None:
    history = [{"kind": "fill", "operation": "TYPE_TEXT", "node": 3, "action": "Email", "text": _TYPED}]
    session, bodies, helper_calls = _run(
        _page("south ridge"),
        history,
        [_decision("TYPE_TEXT", "type_text_target", "1")],
    )
    assert len(bodies) == 1
    assert any("Email" in label for label in _type_labels(bodies[0]))
    assert len(helper_calls) == 1
    assert len(session.text_calls) == 1


def test_older_type_text_is_not_masked() -> None:
    history = [
        {"kind": "fill", "operation": "TYPE_TEXT", "node": 3, "action": "Email", "text": _TYPED},
        {"kind": "click", "operation": "CLICK", "node": 5, "action": "Continue", "text": None},
    ]
    session, bodies, helper_calls = _run(_page(_TYPED), history, [_decision("TYPE_TEXT", "type_text_target", "1")])
    assert len(bodies) == 1
    assert any("Email" in label for label in _type_labels(bodies[0]))
    assert len(helper_calls) == 1
    assert len(session.text_calls) == 1
