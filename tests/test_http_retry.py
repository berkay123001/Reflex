"""One retry for Meta and OpenRouter HTTP 429 and 5xx responses."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from reflex.config import ReflexConfig
from reflex.supervisor import _chat_completion
from reflex.system1 import choose_action

_PAGE = {
    "url": "https://example.com",
    "title": "Example",
    "text": "Hello",
    "actions": [
        {"id": "btn", "node": 1, "kind": "click", "role": "button", "label": "Go"},
    ],
}
_CHAT_OK = {
    "model": "supervisor-test",
    "choices": [{"message": {"content": '{"action_type": "ASK_USER"}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
_DECISION_OK = {
    "model": "jev-test",
    "answers": {
        "operation": {"choice": "CLICK", "probabilities": {"CLICK": 0.8, "DONE": 0.1, "BLOCKED": 0.1}},
        "click_target": {"choice": "1", "probabilities": {"1": 0.9}},
    },
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def _response(status: int, payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload if payload is not None else {"error": status},
        request=httpx.Request("POST", "https://example.test/retry"),
    )


def _scripted_post(statuses: list[int], ok_payload: dict[str, Any]) -> tuple[list[int], Any]:
    calls: list[int] = []

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        index = len(calls)
        code = statuses[index] if index < len(statuses) else statuses[-1]
        calls.append(code)
        if code == 200:
            return _response(200, ok_payload)
        return _response(code)

    return calls, post


@pytest.mark.parametrize(
    ("caller", "status"),
    [
        ("meta", 429),
        ("meta", 500),
        ("meta", 503),
        ("openrouter", 429),
        ("openrouter", 500),
        ("openrouter", 503),
        ("system1", 429),
        ("system1", 500),
        ("system1", 503),
    ],
)
def test_retryable_status_is_retried_once_then_returned(caller: str, status: int) -> None:
    calls, post = _scripted_post([status, 200], _CHAT_OK if caller != "system1" else _DECISION_OK)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)

    if caller == "system1":
        config = ReflexConfig(decision_backend="openrouter_jev")
        with (
            patch("reflex.system1.get_openrouter_api_key", return_value="test-key"),
            patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
            patch("reflex.system1.time.sleep", side_effect=sleep),
        ):
            result = choose_action(_PAGE, "Open the page", [], config)
        assert result["operation"] == "CLICK"
        assert result["retry_reason"] == f"http_{status}"
    else:
        model = "muse-spark-1.3-contributor" if caller == "meta" else "qwen/qwen3.8-flash"
        module_sleep = "reflex.supervisor.time.sleep"
        with (
            patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
            patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
            patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=post),
            patch(module_sleep, side_effect=sleep),
        ):
            parsed, _latency, _cost, _model = _chat_completion(
                model=model,
                system_prompt="sys",
                user_prompt="usr",
                allow_model_fallback=False,
            )
        assert parsed["action_type"] == "ASK_USER"
        assert parsed.retry_reason == f"http_{status}"
    assert calls == [status, 200]
    assert len(slept) == 1
    assert 1 <= slept[0] <= 2


@pytest.mark.parametrize("caller", ["meta", "openrouter", "system1"])
def test_second_5xx_is_not_retried_again(caller: str) -> None:
    calls, post = _scripted_post([500, 500, 500], _CHAT_OK if caller != "system1" else _DECISION_OK)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)

    with pytest.raises(RuntimeError):
        if caller == "system1":
            config = ReflexConfig(decision_backend="openrouter_jev")
            with (
                patch("reflex.system1.get_openrouter_api_key", return_value="test-key"),
                patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
                patch("reflex.system1.time.sleep", side_effect=sleep),
            ):
                choose_action(_PAGE, "Open the page", [], config)
        else:
            model = "muse-spark-1.3-contributor" if caller == "meta" else "qwen/qwen3.8-flash"
            with (
                patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
                patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
                patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=post),
                patch("reflex.supervisor.time.sleep", side_effect=sleep),
            ):
                _chat_completion(model=model, system_prompt="sys", user_prompt="usr", allow_model_fallback=False)
    assert calls == [500, 500]
    assert len(slept) == 1
    assert 1 <= slept[0] <= 2


@pytest.mark.parametrize("caller", ["openrouter", "system1"])
def test_http_400_is_not_retried(caller: str) -> None:
    calls, post = _scripted_post([400, 200], _CHAT_OK if caller != "system1" else _DECISION_OK)
    slept: list[float] = []

    with pytest.raises(RuntimeError):
        if caller == "system1":
            config = ReflexConfig(decision_backend="openrouter_jev")
            with (
                patch("reflex.system1.get_openrouter_api_key", return_value="test-key"),
                patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
                patch("reflex.system1.time.sleep", side_effect=lambda _seconds: slept.append(_seconds)),
            ):
                choose_action(_PAGE, "Open the page", [], config)
        else:
            model = "muse-spark-1.3-contributor" if caller == "meta" else "qwen/qwen3.8-flash"
            with (
                patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
                patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
                patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=post),
                patch("reflex.supervisor.time.sleep", side_effect=lambda _seconds: slept.append(_seconds)),
            ):
                _chat_completion(model=model, system_prompt="sys", user_prompt="usr", allow_model_fallback=False)
    assert calls == [400]
    assert slept == []


@pytest.mark.parametrize("exc_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_openrouter_transport_error_is_retried_once(exc_type: type[Exception]) -> None:
    calls: list[str] = []
    slept: list[float] = []

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        calls.append("post")
        if len(calls) == 1:
            raise exc_type("down")
        return _response(200, _DECISION_OK)

    config = ReflexConfig(decision_backend="openrouter_jev")
    with (
        patch("reflex.system1.get_openrouter_api_key", return_value="test-key"),
        patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
        patch("reflex.system1.time.sleep", side_effect=slept.append),
    ):
        result = choose_action(_PAGE, "Open the page", [], config)
    assert result["operation"] == "CLICK"
    assert result["retry_reason"] == "transport"
    assert calls == ["post", "post"]
    assert len(slept) == 1
    assert 1 <= slept[0] <= 2


def test_openrouter_transport_error_is_not_retried_twice() -> None:
    calls: list[str] = []
    slept: list[float] = []

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        calls.append("post")
        raise httpx.ConnectError("still down")

    config = ReflexConfig(decision_backend="openrouter_jev")
    with (
        patch("reflex.system1.get_openrouter_api_key", return_value="test-key"),
        patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
        patch("reflex.system1.time.sleep", side_effect=slept.append),
        pytest.raises(RuntimeError, match="request failed"),
    ):
        choose_action(_PAGE, "Open the page", [], config)
    assert calls == ["post", "post"]
    assert len(slept) == 1


def test_meta_400_retries_once_without_reasoning_fields() -> None:
    payloads: list[dict[str, Any]] = []

    def post(*_args: Any, **kwargs: Any) -> httpx.Response:
        payload = dict(kwargs["json"])
        payloads.append(payload)
        if len(payloads) == 1:
            return _response(400)
        return _response(200, _CHAT_OK)

    with (
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=post),
        patch("reflex.supervisor.time.sleep", side_effect=AssertionError("meta 400 fallback must not sleep")),
    ):
        parsed, _latency, _cost, _model = _chat_completion(
            model="muse-spark-1.3-contributor",
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=False,
        )
    assert parsed["action_type"] == "ASK_USER"
    assert payloads[0]["reasoning_effort"] == "low"
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in payloads[1]
    assert "response_format" not in payloads[1]
    assert len(payloads) == 2


def test_meta_400_plain_body_is_not_retried_again() -> None:
    payloads: list[dict[str, Any]] = []

    def post(*_args: Any, **kwargs: Any) -> httpx.Response:
        payloads.append(dict(kwargs["json"]))
        return _response(400)

    with (
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=post),
        patch("reflex.supervisor.time.sleep", side_effect=AssertionError("meta 400 fallback must not sleep")),
        pytest.raises(RuntimeError),
    ):
        _chat_completion(
            model="muse-spark-1.3-contributor",
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=False,
        )
    assert len(payloads) == 2
    assert "reasoning_effort" not in payloads[1]
    assert "response_format" not in payloads[1]


def test_local_turnike_transport_error_is_not_retried() -> None:
    calls: list[str] = []

    def post(*_args: Any, **_kwargs: Any) -> httpx.Response:
        calls.append("post")
        raise httpx.ConnectError("connection refused")

    config = ReflexConfig(decision_backend="local_turnike")
    with (
        patch("reflex.system1._HTTP_CLIENT.post", side_effect=post),
        patch("reflex.system1.time.sleep", side_effect=AssertionError("local_turnike must not sleep")),
        pytest.raises(RuntimeError, match="transport failed"),
    ):
        choose_action(_PAGE, "Open the page", [], config)
    assert calls == ["post"]
