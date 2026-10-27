"""Unit tests for Task G1: Remove silent backend and model fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from reflex.config import OPENROUTER_DECISIONS_URL, ReflexConfig
from reflex.context import DualTierContext
from reflex.supervisor import (
    _chat_completion,
    generate_field_text,
    interpret_user_message,
    resolve_escalation,
)
from reflex.system1 import choose_action


def test_config_allow_model_fallback_default():
    """Ensure ReflexConfig defaults allow_model_fallback to False."""
    cfg = ReflexConfig()
    assert cfg.allow_model_fallback is False

    cfg_custom = ReflexConfig(allow_model_fallback=True)
    assert cfg_custom.allow_model_fallback is True


def test_system1_local_turnike_connect_error_raises_runtime_error_and_does_not_mutate_backend():
    """When local_turnike encounters TransportError (e.g. ConnectError),

    System 1 must raise RuntimeError immediately, MUST NOT mutate config.decision_backend,
    and MUST NOT call OpenRouter.
    """
    config = ReflexConfig(decision_backend="local_turnike")
    page = {
        "url": "https://example.com",
        "title": "Example",
        "text": "Hello world",
        "actions": [
            {"id": "btn_1", "node": 10, "kind": "click", "role": "button", "label": "Submit"},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 500},
        ],
    }

    called_urls: list[str] = []

    def fake_post(url, *args, **kwargs):
        called_urls.append(url)
        raise httpx.ConnectError("Connection refused to 127.0.0.1:8794")

    with (
        patch("reflex.system1._HTTP_CLIENT.post", side_effect=fake_post),
        pytest.raises(RuntimeError) as exc_info,
    ):
        choose_action(page=page, compiled_goal="Submit form", history=[], config=config)

    # 1. An explicit RuntimeError is raised
    assert "System 1 (local_turnike) transport failed" in str(exc_info.value)

    # 2. config.decision_backend is NEVER mutated
    assert config.decision_backend == "local_turnike"

    # 3. OpenRouter was never contacted
    assert OPENROUTER_DECISIONS_URL not in called_urls
    assert called_urls == [config.local_turnike_url]


def test_system1_local_turnike_timeout_transport_error():
    """Verify other TransportError instances (like ReadTimeout) also fail explicitly without mutating backend."""
    config = ReflexConfig(decision_backend="local_turnike")
    page = {
        "url": "https://example.com",
        "title": "Example",
        "text": "Content",
        "actions": [
            {"id": "btn_1", "node": 10, "kind": "click", "role": "button", "label": "Submit"},
        ],
    }

    called_urls: list[str] = []

    def fake_post(url, *args, **kwargs):
        called_urls.append(url)
        raise httpx.ReadTimeout("Read timed out on socket")

    with (
        patch("reflex.system1._HTTP_CLIENT.post", side_effect=fake_post),
        pytest.raises(RuntimeError) as exc_info,
    ):
        choose_action(page=page, compiled_goal="Submit", history=[], config=config)

    assert "System 1 (local_turnike) transport failed" in str(exc_info.value)
    assert config.decision_backend == "local_turnike"
    assert OPENROUTER_DECISIONS_URL not in called_urls


def test_supervisor_chat_completion_disallows_fallback_by_default():
    """Supervisor _chat_completion must NOT try fallback models when allow_model_fallback is False."""
    requested_model = "custom/my-requested-model"
    attempted_models: list[str] = []

    def fake_post(url, *args, **kwargs):
        json_data = kwargs.get("json", {})
        attempted_models.append(json_data.get("model"))
        raise httpx.ConnectError("Connection to OpenRouter failed")

    with (
        patch("reflex.supervisor.get_openrouter_api_key", return_value="fake-or-key"),
        patch("reflex.supervisor.get_meta_api_key", return_value=""),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
        pytest.raises(RuntimeError) as exc_info,
    ):
        _chat_completion(
            model=requested_model,
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=False,
        )

    assert f"Supervisor ({requested_model}) call failed" in str(exc_info.value)
    # Only the requested model was tried (retried for attempts, but never fallback models)
    assert len(attempted_models) > 0
    assert set(attempted_models) == {requested_model}
    assert "qwen/qwen3.8-flash" not in attempted_models
    assert "anthropic/claude-haiku-4.5" not in attempted_models


def test_supervisor_chat_completion_allows_fallback_when_enabled():
    """When allow_model_fallback is True, fallback models are attempted upon failure."""
    primary_model = "primary/failing-model"
    attempted_models: list[str] = []

    def fake_post(url, *args, **kwargs):
        json_data = kwargs.get("json", {})
        attempted_models.append(json_data.get("model"))
        if json_data.get("model") == "qwen/qwen3.8-flash":
            resp_mock = MagicMock()
            resp_mock.status_code = 200
            resp_mock.json.return_value = {
                "choices": [{"message": {"content": '{"result": "recovered"}'}}],
                "usage": {"cost": 0.0005},
                "model": "qwen/qwen3.8-flash",
            }
            return resp_mock
        raise httpx.ConnectError("Primary failed")

    with (
        patch("reflex.supervisor.get_openrouter_api_key", return_value="fake-or-key"),
        patch("reflex.supervisor.get_meta_api_key", return_value=""),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
    ):
        parsed, _latency_ms, _cost, model = _chat_completion(
            model=primary_model,
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=True,
        )

    assert parsed == {"result": "recovered"}
    assert model == "qwen/qwen3.8-flash"
    assert primary_model in attempted_models
    assert "qwen/qwen3.8-flash" in attempted_models


def test_supervisor_meta_model_disallows_fallback_by_default():
    """When Meta API is used and allow_model_fallback is False, secondary candidates and OpenRouter are not called."""
    meta_model = "muse-spark-1.3-contributor"
    attempted_models: list[str] = []
    attempted_urls: list[str] = []

    def fake_post(url, *args, **kwargs):
        attempted_urls.append(url)
        json_data = kwargs.get("json", {})
        attempted_models.append(json_data.get("model"))
        raise httpx.ConnectError("Meta API unreachable")

    with (
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-token-123"),
        patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
        pytest.raises(RuntimeError) as exc_info,
    ):
        _chat_completion(
            model=meta_model,
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=False,
        )

    assert f"Supervisor ({meta_model}) call failed" in str(exc_info.value)
    # Only the exact meta_model was tried; secondary "muse-wonder-1.0" was NOT tried
    assert attempted_models == [meta_model]
    # OpenRouter was NOT called
    assert OPENROUTER_DECISIONS_URL not in attempted_urls


def test_supervisor_meta_model_allows_fallback_when_enabled():
    """When Meta API is used and allow_model_fallback is True, secondary candidates and OpenRouter fallback are tried."""
    meta_model = "muse-spark-1.3-contributor"
    attempted_models: list[str] = []

    def fake_post(url, *args, **kwargs):
        json_data = kwargs.get("json", {})
        m = json_data.get("model")
        attempted_models.append(m)
        if m == "qwen/qwen3.8-flash":
            resp_mock = MagicMock()
            resp_mock.status_code = 200
            resp_mock.json.return_value = {
                "choices": [{"message": {"content": '{"action": "ok"}'}}],
                "usage": {"cost": 0.0},
                "model": "qwen/qwen3.8-flash",
            }
            return resp_mock
        raise httpx.ConnectError(f"{m} failed")

    with (
        patch("reflex.supervisor.get_meta_api_key", return_value="meta-token-123"),
        patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
    ):
        parsed, _, _, model = _chat_completion(
            model=meta_model,
            system_prompt="sys",
            user_prompt="usr",
            allow_model_fallback=True,
        )

    assert parsed == {"action": "ok"}
    assert model == "qwen/qwen3.8-flash"
    assert "muse-spark-1.3-contributor" in attempted_models
    assert "muse-wonder-1.0" in attempted_models
    assert "qwen/qwen3.8-flash" in attempted_models


def test_supervisor_functions_respect_config_and_allow_model_fallback():
    """Verify generate_field_text, resolve_escalation, and interpret_user_message respect allow_model_fallback."""
    ctx = DualTierContext(primary_goal="Goal", active_subgoal="Subgoal")
    page = {"url": "https://example.com", "title": "T", "text": "Body"}
    cfg = ReflexConfig(allow_model_fallback=False)

    attempted_models: list[str] = []

    def fake_post(url, *args, **kwargs):
        json_data = kwargs.get("json", {})
        attempted_models.append(json_data.get("model"))
        raise httpx.ConnectError("Service offline")

    with (
        patch("reflex.supervisor.get_openrouter_api_key", return_value="or-key"),
        patch("reflex.supervisor.get_meta_api_key", return_value=""),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
    ):
        # 1. generate_field_text
        with pytest.raises(RuntimeError):
            generate_field_text(
                ctx=ctx,
                action={"kind": "fill", "label": "Email"},
                page=page,
                history=[],
                supervisor_model="custom/test-model",
                config=cfg,
            )
        assert set(attempted_models) == {"custom/test-model"}
        attempted_models.clear()

        # 2. resolve_escalation
        with pytest.raises(RuntimeError):
            resolve_escalation(
                ctx=ctx,
                page=page,
                elements=[],
                history=[],
                escalation_reason="Blocked",
                trigger_type="test",
                s1_decision=None,
                supervisor_model="custom/test-model",
                config=cfg,
            )
        assert set(attempted_models) == {"custom/test-model"}
        attempted_models.clear()

        # 3. interpret_user_message
        with pytest.raises(RuntimeError):
            interpret_user_message(
                user_message="Hello",
                ctx=ctx,
                page=page,
                history=[],
                pending_fact_key=None,
                supervisor_model="custom/test-model",
                config=cfg,
            )
        assert set(attempted_models) == {"custom/test-model"}
