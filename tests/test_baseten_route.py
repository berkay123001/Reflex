"""Meta open-weights models on Baseten (`meta-models/...`) are called on Baseten, not on the Meta API."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from reflex.config import BASETEN_CHAT_URL, META_API_CHAT_URL, calculate_usage_cost
from reflex.supervisor import _chat_completion

OPEN_MODEL = "meta-models/muse-spark-open-1026"
_CHAT_OK = {
    "model": OPEN_MODEL,
    "choices": [{"message": {"content": '{"action_type": "ASK_USER"}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
}


def _capture_post(calls: list[dict[str, Any]]) -> Any:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(200, json=_CHAT_OK, request=httpx.Request("POST", url))

    return post


def test_open_weights_model_goes_to_baseten_with_api_key_header_and_meta_payload() -> None:
    calls: list[dict[str, Any]] = []
    with (
        patch("reflex.supervisor.get_baseten_api_key", return_value="bt-key"),
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=_capture_post(calls)),
    ):
        result = _chat_completion(OPEN_MODEL, "sys", "user", reasoning_effort="minimal")

    assert len(calls) == 1
    assert calls[0]["url"] == BASETEN_CHAT_URL
    assert calls[0]["headers"]["Authorization"] == "Api-Key bt-key"
    payload = calls[0]["json"]
    assert payload["model"] == OPEN_MODEL
    assert payload["reasoning_effort"] == "minimal"
    assert payload["response_format"] == {"type": "json_object"}
    parsed, _latency, cost_usd, _model = result
    assert parsed == {"action_type": "ASK_USER"}
    expected_cost, _ = calculate_usage_cost(OPEN_MODEL, _CHAT_OK["usage"])
    assert cost_usd == pytest.approx(expected_cost)
    assert expected_cost == pytest.approx((1000 * 1.25 + 200 * 4.25) / 1_000_000)


def test_open_weights_model_without_baseten_key_fails_clearly_and_never_uses_meta() -> None:
    calls: list[dict[str, Any]] = []
    with (
        patch("reflex.supervisor.get_baseten_api_key", return_value=""),
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=_capture_post(calls)),
        pytest.raises(RuntimeError, match="BASETEN_API_KEY"),
    ):
        _chat_completion(OPEN_MODEL, "sys", "user")
    assert calls == []


def test_meta_api_models_still_use_meta_endpoint() -> None:
    calls: list[dict[str, Any]] = []
    with (
        patch("reflex.supervisor.get_baseten_api_key", return_value="bt-key"),
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=_capture_post(calls)),
    ):
        _chat_completion("muse-spark-1.3-contributor", "sys", "user")
    assert calls[0]["url"] == META_API_CHAT_URL
    assert calls[0]["headers"]["Authorization"] == "Bearer meta-key"
