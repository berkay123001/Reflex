"""Unit tests for Task G1: Remove silent backend and model fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from reflex.agent import ReflexSession
from reflex.config import OPENROUTER_DECISIONS_URL, ReflexConfig
from reflex.context import DualTierContext
from reflex.supervisor import (
    FieldTextOutcome,
    SupervisorDecision,
    UserMessageInterpretation,
    _chat_completion,
    generate_field_text,
    interpret_user_message,
    resolve_escalation,
)
from reflex.system1 import choose_action


@pytest.fixture(autouse=True)
def _mock_session_init_supervisor():
    with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
        yield


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


def test_supervisor_meta_model_missing_key_raises_runtime_error():
    """When a Meta model is requested but META_API_KEY is missing,

    Supervisor must raise RuntimeError immediately without contacting OpenRouter.
    """
    called_urls: list[str] = []

    def fake_post(url, *args, **kwargs):
        called_urls.append(url)
        return MagicMock()

    with (
        patch("reflex.supervisor.get_meta_api_key", return_value=""),
        patch("reflex.supervisor.get_openrouter_api_key", return_value="fake-or-key"),
        patch("reflex.supervisor._HTTP_CLIENT.post", side_effect=fake_post),
    ):
        # With allow_model_fallback=False
        with pytest.raises(RuntimeError) as exc_info:
            _chat_completion(
                model="muse-spark-1.3-contributor",
                system_prompt="sys",
                user_prompt="usr",
                allow_model_fallback=False,
            )
        assert "META_API_KEY not configured for model muse-spark-1.3-contributor" in str(exc_info.value)

        # And with allow_model_fallback=True
        with pytest.raises(RuntimeError) as exc_info2:
            _chat_completion(
                model="meta/llama-3-70b",
                system_prompt="sys",
                user_prompt="usr",
                allow_model_fallback=True,
            )
        assert "META_API_KEY not configured for model meta/llama-3-70b" in str(exc_info2.value)

    assert len(called_urls) == 0


def test_supervisor_functions_respect_allow_model_fallback():
    """Verify generate_field_text, resolve_escalation, and interpret_user_message respect allow_model_fallback."""
    ctx = DualTierContext(primary_goal="Goal", active_subgoal="Subgoal")
    page = {"url": "https://example.com", "title": "T", "text": "Body"}

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
                allow_model_fallback=False,
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
                allow_model_fallback=False,
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
                allow_model_fallback=False,
            )
        assert set(attempted_models) == {"custom/test-model"}


def _make_fake_browser(page: dict | None = None) -> MagicMock:
    if page is None:
        page = {
            "url": "https://example.com",
            "title": "Example",
            "text": "Page content",
            "fingerprint": "fp_1",
            "actions": [
                {"id": "btn_1", "node": 10, "kind": "click", "role": "button", "label": "Submit"},
                {"id": "inp_1", "node": 11, "kind": "fill", "role": "textbox", "label": "Email", "value": ""},
            ],
        }
    fake_b = MagicMock()
    fake_b.observe.return_value = page
    fake_b.fresh.return_value = True
    fake_b.act.return_value = {
        "url": page["url"],
        "title": page["title"],
        "text": page["text"],
        "actions": page["actions"],
        "fingerprint": "fp_2",
    }
    return fake_b


def test_session_step_runtime_error_sets_status_error_and_does_not_escalate():
    """End-to-end: On Turnike connection/transport failure (RuntimeError),

    session.step() sets status == 'error', populates last_error, and NEVER calls resolve_escalation.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=False)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    with (
        patch("reflex.agent.choose_action", side_effect=RuntimeError("Turnike server unreachable: ConnectError")),
        patch("reflex.agent.resolve_escalation") as mock_resolve_esc,
    ):
        snap = session.step()

    assert snap["status"] == "error"
    assert session.status == "error"
    assert "Turnike server unreachable: ConnectError" in snap["last_error"]
    assert snap["metrics"]["s1_errors"] == 0
    mock_resolve_esc.assert_not_called()


def test_session_step_httpx_error_sets_status_error_and_does_not_escalate():
    """End-to-end: On httpx.HTTPError from choose_action,

    session.step() sets status == 'error', populates last_error, and NEVER calls resolve_escalation.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=False)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    with (
        patch("reflex.agent.choose_action", side_effect=httpx.ConnectError("Connection refused to 127.0.0.1:8794")),
        patch("reflex.agent.resolve_escalation") as mock_resolve_esc,
    ):
        snap = session.step()

    assert snap["status"] == "error"
    assert session.status == "error"
    assert "Connection refused to 127.0.0.1:8794" in snap["last_error"]
    assert snap["metrics"]["s1_errors"] == 0
    mock_resolve_esc.assert_not_called()


def test_session_step_value_error_escalates_to_s2_and_increments_s1_errors():
    """End-to-end: On ValueError from choose_action (invalid model output),

    session.step() escalates to resolve_escalation and increments metrics['s1_errors'].
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=False)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    fake_sup = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Recovering from malformed S1 output",
        model="qwen/qwen3.8-omni-flash",
    )

    with (
        patch("reflex.agent.choose_action", side_effect=ValueError("Invalid output")),
        patch("reflex.agent.resolve_escalation", return_value=fake_sup) as mock_resolve_esc,
    ):
        snap = session.step()

    mock_resolve_esc.assert_called_once()
    assert mock_resolve_esc.call_args.kwargs["trigger_type"] == "s1_error"
    assert "Invalid output" in mock_resolve_esc.call_args.kwargs["escalation_reason"]
    assert snap["metrics"]["s1_errors"] == 1
    assert session.s1_errors == 1


@pytest.mark.parametrize("fallback_enabled", [True, False])
def test_session_allow_model_fallback_wires_to_resolve_escalation(fallback_enabled: bool):
    """End-to-end: config.allow_model_fallback on ReflexSession is forwarded to resolve_escalation."""
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=fallback_enabled)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    fake_sup = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="1",
        reasoning="Recovery",
        model="qwen/qwen3.8-omni-flash",
    )

    with (
        patch("reflex.agent.choose_action", side_effect=ValueError("Invalid output")),
        patch("reflex.agent.resolve_escalation", return_value=fake_sup) as mock_resolve_esc,
    ):
        session.step()

    mock_resolve_esc.assert_called_once()
    assert mock_resolve_esc.call_args.kwargs["allow_model_fallback"] is fallback_enabled


@pytest.mark.parametrize("fallback_enabled", [True, False])
def test_session_allow_model_fallback_wires_to_generate_field_text(fallback_enabled: bool):
    """End-to-end: config.allow_model_fallback on ReflexSession is forwarded to generate_field_text."""
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=fallback_enabled)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    s1_decision = {
        "operation": "TYPE_TEXT",
        "target": "2",
        "choice": "inp_1",
        "confidence": 0.95,
        "target_confidence": 0.95,
        "target_probabilities": {"2": 0.95},
        "operation_probabilities": {"TYPE_TEXT": 0.95},
        "latency_ms": 50,
        "model": "turnike-4b",
        "usage": {},
    }

    fake_outcome = FieldTextOutcome(
        text="test@example.com",
        ask_user=None,
        fact_key="email",
        model="qwen/qwen3.8-omni-flash",
        latency_ms=50,
        cost_usd=0.0001,
    )

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.generate_field_text", return_value=fake_outcome) as mock_gen_text,
    ):
        session.step()

    mock_gen_text.assert_called_once()
    assert mock_gen_text.call_args.kwargs["allow_model_fallback"] is fallback_enabled


@pytest.mark.parametrize("fallback_enabled", [True, False])
def test_session_allow_model_fallback_wires_to_interpret_user_message(fallback_enabled: bool):
    """End-to-end: config.allow_model_fallback on ReflexSession is forwarded to interpret_user_message."""
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig(allow_model_fallback=fallback_enabled)

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    fake_interp = UserMessageInterpretation(
        intent="CHAT_ONLY",
        direct_reply="Bu bir test cevabıdır.",
        model="qwen/qwen3.8-omni-flash",
        latency_ms=40,
        cost_usd=0.0001,
    )

    with patch("reflex.agent.interpret_user_message", return_value=fake_interp) as mock_interp:
        snap = session.reply_user("Ne durumdayız?")

    mock_interp.assert_called_once()
    assert mock_interp.call_args.kwargs["allow_model_fallback"] is fallback_enabled
    assert snap["status"] == "waiting_user"


def test_session_step_resolve_escalation_error_sets_status_error():
    """When resolve_escalation raises an error during step(),

    step() must catch it cleanly, set status='error', populate last_error, and return snapshot.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig()

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    with (
        patch("reflex.agent.choose_action", side_effect=ValueError("Invalid S1 model JSON")),
        patch("reflex.agent.resolve_escalation", side_effect=RuntimeError("Supervisor failed: 500")),
    ):
        snap = session.step()

    assert snap["status"] == "error"
    assert session.status == "error"
    assert "Supervisor failed: 500" in snap["last_error"]
    assert session.last_error == "Supervisor failed: 500"


def test_session_step_generate_field_text_error_sets_status_error():
    """When generate_field_text raises an error during fill execution,

    step() must catch it cleanly, set status='error', populate last_error, and return snapshot.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig()

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    s1_decision = {
        "operation": "TYPE_TEXT",
        "target": "2",
        "choice": "inp_1",
        "confidence": 0.95,
        "target_confidence": 0.95,
        "target_probabilities": {"2": 0.95},
        "operation_probabilities": {"TYPE_TEXT": 0.95},
        "latency_ms": 50,
        "model": "turnike-4b",
        "usage": {},
    }

    with (
        patch("reflex.agent.choose_action", return_value=s1_decision),
        patch("reflex.agent.generate_field_text", side_effect=RuntimeError("Field text service unavailable: 503")),
    ):
        snap = session.step()

    assert snap["status"] == "error"
    assert session.status == "error"
    assert "Field text service unavailable: 503" in snap["last_error"]
    assert session.last_error == "Field text service unavailable: 503"


def test_session_reply_user_interpret_error_sets_status_error():
    """When interpret_user_message raises an error during reply_user(),

    reply_user() must catch it cleanly, set status='error', populate last_error, and return snapshot.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig()

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    with patch("reflex.agent.interpret_user_message", side_effect=RuntimeError("Interpret service failed: 429")):
        snap = session.reply_user("Help me with this page")

    assert snap["status"] == "error"
    assert session.status == "error"
    assert "Interpret service failed: 429" in snap["last_error"]
    assert session.last_error == "Interpret service failed: 429"


@pytest.mark.parametrize("exc_to_raise", [KeyError("missing_field"), TypeError("unsupported operand")])
def test_session_step_unexpected_s1_exception_sets_status_error_without_escalation(exc_to_raise: Exception):
    """When choose_action raises an unexpected non-ValueError exception (e.g. KeyError or TypeError),

    step() must set status='error', populate last_error, and NOT escalate to S2.
    """
    fake_browser = _make_fake_browser()
    cfg = ReflexConfig()

    with patch("reflex.agent.Browser", return_value=fake_browser):
        session = ReflexSession(url="https://example.com", goal="Test goal", config=cfg)

    with (
        patch("reflex.agent.choose_action", side_effect=exc_to_raise),
        patch("reflex.agent.resolve_escalation") as mock_resolve_esc,
    ):
        snap = session.step()

    assert snap["status"] == "error"
    assert session.status == "error"
    assert str(exc_to_raise) in snap["last_error"]
    assert snap["metrics"]["s1_errors"] == 0
    mock_resolve_esc.assert_not_called()

