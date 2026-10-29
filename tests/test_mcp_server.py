"""Tests for Reflex MCP server (`reflex/mcp_server.py`) and isolated browser lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.elicitation import AcceptedElicitation, CancelledElicitation, DeclinedElicitation

from reflex.browser.isolated import FORBIDDEN_SHARED_PORT, is_port_open
from reflex.mcp_server import (
    ActionConfirmElicitationSchema,
    UserReplyElicitationSchema,
    reflex_confirm,
    reflex_reply,
    reflex_start,
    reflex_stop,
    reflex_wait,
)
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation
from scripts.eval import managed_web_server


class _FakeReflexSession:
    """Thread-recording fake `ReflexSession` for testing MCP tool flows and thread affinity."""

    def __init__(
        self,
        url: str,
        goal: str,
        config: Any = None,
        *,
        scenario: str = "ask_then_done",
    ) -> None:
        self.url = url
        self.goal = goal
        self.config = config
        self.scenario = scenario
        self.status = "ready"
        self.history: list[dict[str, Any]] = []
        self.last_asked_question: str | None = None
        self.pending_confirmation: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.final_answer: str | None = None
        self.closed = False

        self.init_thread_id = threading.get_ident()
        self.call_thread_ids: list[tuple[str, int]] = [("init", self.init_thread_id)]
        self.replies: list[str] = []
        self.confirmations: list[tuple[bool, str]] = []

    def step(self) -> dict[str, Any]:
        self.call_thread_ids.append(("step", threading.get_ident()))
        step_no = len(self.history) + 1

        if self.scenario == "ask_then_done":
            if step_no == 1:
                self.history.append(
                    {
                        "step": 1,
                        "actor": "S1",
                        "operation": "CLICK",
                        "action": "Search Flights",
                        "kind": "click",
                        "target": "1",
                        "url": "https://example.com/search",
                    }
                )
                self.status = "waiting_user"
                self.last_asked_question = "Hangi şehre uçmak istiyorsunuz?"
            else:
                for idx in range(2, 8):
                    self.history.append(
                        {
                            "step": idx,
                            "actor": "S1",
                            "operation": "CLICK",
                            "action": f"Step Action {idx}",
                            "kind": "click",
                            "target": str(idx),
                            "url": "https://flights.example.com/results",
                        }
                    )
                self.status = "done"
                self.final_answer = f"{self.replies[-1]} uçuşu başarıyla bulundu."

        elif self.scenario == "confirm_flow":
            if step_no == 1:
                self.history.append(
                    {
                        "step": 1,
                        "actor": "S1",
                        "operation": "CLICK",
                        "action": "Open Settings",
                        "kind": "click",
                        "target": "1",
                        "url": "https://admin.example.com/settings",
                    }
                )
                self.status = "waiting_confirmation"
                self.pending_confirmation = {
                    "summary": "Delete workspace permanently",
                    "reason": "Irreversible destructive action",
                    "action": {"label": "Delete Workspace"},
                }
            else:
                self.history.append(
                    {
                        "step": step_no,
                        "actor": "S1",
                        "operation": "CLICK",
                        "action": "Confirmed Action",
                        "kind": "click",
                        "target": "2",
                        "url": "https://admin.example.com/done",
                    }
                )
                self.status = "done"
                self.final_answer = "İşlem tamamlandı."

        return self.snapshot()

    def reply_user(self, message: str) -> dict[str, Any]:
        self.call_thread_ids.append(("reply_user", threading.get_ident()))
        self.replies.append(message)
        self.last_asked_question = None
        self.status = "ready"
        return self.snapshot()

    def confirm_action(self, approved: bool, alternative_instruction: str = "") -> dict[str, Any]:
        self.call_thread_ids.append(("confirm_action", threading.get_ident()))
        self.confirmations.append((approved, alternative_instruction))
        self.pending_confirmation = None
        if approved:
            self.status = "ready"
        elif alternative_instruction.strip():
            self.status = "ready"
        else:
            self.status = "blocked"
            self.last_error = "User declined executing Delete workspace permanently."
        return self.snapshot()

    def close(self) -> None:
        self.call_thread_ids.append(("close", threading.get_ident()))
        self.closed = True

    def snapshot(self) -> dict[str, Any]:
        chat_log = []
        if self.final_answer:
            chat_log.append({"role": "assistant", "content": self.final_answer})
        return {
            "status": self.status,
            "history": list(self.history),
            "asked_question": self.last_asked_question,
            "pending_confirmation": (
                {
                    "summary": self.pending_confirmation["summary"],
                    "reason": self.pending_confirmation["reason"],
                    "action_label": self.pending_confirmation["action"].get("label"),
                }
                if self.pending_confirmation
                else None
            ),
            "last_error": self.last_error,
            "context": {"chat_log": chat_log},
            "metrics": {
                "steps": len(self.history),
                "total_cost_usd": 0.0025,
            },
        }


@pytest.mark.anyio
async def test_mcp_flow_start_wait_waiting_user_reply_done() -> None:
    """M4: Full tool flow start -> wait -> waiting_user -> reply -> wait -> done -> stop with fake ReflexSession."""
    created_sessions: list[_FakeReflexSession] = []

    def fake_factory(url: str, goal: str, config: Any = None) -> _FakeReflexSession:
        sess = _FakeReflexSession(url=url, goal=goal, config=config, scenario="ask_then_done")
        created_sessions.append(sess)
        return sess

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=fake_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9501"
        mock_iso.return_value.__exit__.return_value = None

        start_res = await reflex_start(goal="Uçuş ara", url="https://example.com", max_steps=20, show_browser=False)
        session_id = start_res["session_id"]
        assert session_id
        assert start_res["status"] in {"running", "waiting_user"}

        # 1. Wait until waiting_user (no elicitation capability on client)
        wait_res1 = await reflex_wait(session_id=session_id, timeout_s=10)
        assert wait_res1["status"] == "waiting_user"
        assert wait_res1["question"] == "Hangi şehre uçmak istiyorsunuz?"
        assert wait_res1["steps"] == 1
        assert wait_res1["visited_hosts"] == ["example.com"]

        # 2. Reply to user question
        reply_res = await reflex_reply(session_id=session_id, text="İstanbul")
        assert reply_res["session_id"] == session_id

        # 3. Wait until done
        wait_res2 = await reflex_wait(session_id=session_id, timeout_s=10)
        assert wait_res2["status"] == "done"
        assert wait_res2["final_answer"] == "İstanbul uçuşu başarıyla bulundu."
        assert wait_res2["steps"] == 7
        # Last 5 actions only
        assert len(wait_res2["last_actions"]) == 5
        assert [a["step"] for a in wait_res2["last_actions"]] == [3, 4, 5, 6, 7]
        assert wait_res2["visited_hosts"] == ["example.com", "flights.example.com"]
        assert wait_res2["cost_usd"] == pytest.approx(0.0025)

        # 4. Stop session
        stop_res = await reflex_stop(session_id=session_id)
        assert stop_res == {"session_id": session_id, "status": "stopped"}

    assert len(created_sessions) == 1
    fake_sess = created_sessions[0]
    assert fake_sess.closed is True
    assert fake_sess.replies == ["İstanbul"]
    # Verify concurrency invariant: every method on ReflexSession ran on the worker thread!
    worker_tid = fake_sess.init_thread_id
    assert worker_tid != threading.get_ident()
    for method_name, tid in fake_sess.call_thread_ids:
        assert tid == worker_tid, f"{method_name} ran on {tid} instead of worker thread {worker_tid}"


@pytest.mark.anyio
async def test_mcp_flow_confirm_rejection_and_stop() -> None:
    """M4: Confirm rejection transitions session to blocked and never executes the destructive action."""
    created_sessions: list[_FakeReflexSession] = []

    def fake_factory(url: str, goal: str, config: Any = None) -> _FakeReflexSession:
        sess = _FakeReflexSession(url=url, goal=goal, config=config, scenario="confirm_flow")
        created_sessions.append(sess)
        return sess

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=fake_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9502"
        mock_iso.return_value.__exit__.return_value = None

        start_res = await reflex_start(goal="Delete workspace", url="https://admin.example.com", show_browser=False)
        session_id = start_res["session_id"]

        wait_res = await reflex_wait(session_id=session_id, timeout_s=10)
        assert wait_res["status"] == "waiting_confirmation"
        assert wait_res["pending_action"] == {
            "summary": "Delete workspace permanently",
            "reason": "Irreversible destructive action",
            "action_label": "Delete Workspace",
        }

        confirm_res = await reflex_confirm(session_id=session_id, approved=False)
        assert confirm_res["status"] == "blocked"

        wait_after = await reflex_wait(session_id=session_id, timeout_s=5)
        assert wait_after["status"] == "blocked"

        stop_res = await reflex_stop(session_id=session_id)
        assert stop_res["status"] == "stopped"

    fake_sess = created_sessions[0]
    assert fake_sess.confirmations == [(False, "")]
    assert fake_sess.closed is True


@pytest.mark.anyio
async def test_mcp_elicitation_waiting_user_and_declined_confirmation() -> None:
    """M3: When client supports ctx.elicit, waiting_user and waiting_confirmation use ctx.elicit;
    declined or cancelled confirmation elicitation ALWAYS resolves to approved=False.
    """
    created_sessions: list[_FakeReflexSession] = []

    def ask_factory(url: str, goal: str, config: Any = None) -> _FakeReflexSession:
        sess = _FakeReflexSession(url=url, goal=goal, config=config, scenario="ask_then_done")
        created_sessions.append(sess)
        return sess

    def confirm_factory(url: str, goal: str, config: Any = None) -> _FakeReflexSession:
        sess = _FakeReflexSession(url=url, goal=goal, config=config, scenario="confirm_flow")
        created_sessions.append(sess)
        return sess

    # 1. waiting_user with accepted elicitation completes in a single reflex_wait call
    fake_ctx_accept = MagicMock()
    fake_ctx_accept.supports_elicitation = True

    async def mock_elicit_user(message: str, schema: Any) -> Any:
        assert "Hangi şehre" in message
        assert schema is UserReplyElicitationSchema
        return AcceptedElicitation(data=UserReplyElicitationSchema(text="Ankara"))

    fake_ctx_accept.elicit.side_effect = mock_elicit_user

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=ask_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9503"
        mock_iso.return_value.__exit__.return_value = None

        start_res = await reflex_start(goal="Uçuş ara", url="https://example.com", show_browser=False)
        sid = start_res["session_id"]
        wait_res = await reflex_wait(session_id=sid, timeout_s=10, ctx=fake_ctx_accept)
        assert wait_res["status"] == "done"
        assert wait_res["final_answer"] == "Ankara uçuşu başarıyla bulundu."
        await reflex_stop(session_id=sid)

    # 2. waiting_confirmation with declined elicitation -> approved=False -> blocked
    fake_ctx_decline = MagicMock()
    fake_ctx_decline.supports_elicitation = True

    async def mock_elicit_decline(message: str, schema: Any) -> Any:
        assert "Delete workspace permanently" in message
        assert schema is ActionConfirmElicitationSchema
        return DeclinedElicitation()

    fake_ctx_decline.elicit.side_effect = mock_elicit_decline

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=confirm_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9504"
        mock_iso.return_value.__exit__.return_value = None

        start_res = await reflex_start(goal="Delete workspace", url="https://admin.example.com", show_browser=False)
        sid = start_res["session_id"]
        wait_res = await reflex_wait(session_id=sid, timeout_s=10, ctx=fake_ctx_decline)
        assert wait_res["status"] == "blocked"
        await reflex_stop(session_id=sid)

    assert created_sessions[-1].confirmations == [(False, "")]

    # 3. waiting_confirmation with cancelled elicitation -> approved=False -> blocked
    fake_ctx_cancel = MagicMock()
    fake_ctx_cancel.supports_elicitation = True

    async def mock_elicit_cancel(message: str, schema: Any) -> Any:
        return CancelledElicitation()

    fake_ctx_cancel.elicit.side_effect = mock_elicit_cancel

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=confirm_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9505"
        mock_iso.return_value.__exit__.return_value = None

        start_res = await reflex_start(goal="Delete workspace", url="https://admin.example.com", show_browser=False)
        sid = start_res["session_id"]
        wait_res = await reflex_wait(session_id=sid, timeout_s=10, ctx=fake_ctx_cancel)
        assert wait_res["status"] == "blocked"
        await reflex_stop(session_id=sid)

    assert created_sessions[-1].confirmations == [(False, "")]


@pytest.mark.anyio
async def test_mcp_real_isolated_browser_smoke_on_local_fixture() -> None:
    """M4: Real isolated Chromium smoke test on local fixture page (never using port 9333)."""
    call_count = 0
    observed_ports: list[int] = []

    def fake_choose_action(page: dict[str, Any], goal: str, history: list[dict[str, Any]], config: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        actions = page.get("actions") or []
        if call_count == 1 and actions:
            first_click = next((a for a in actions if a.get("kind") == "click"), actions[0])
            return {
                "operation": "CLICK",
                "target": "1",
                "choice": first_click["id"],
                "confidence": 0.99,
                "target_confidence": 0.99,
                "target_margin": 0.95,
                "latency_ms": 25,
                "model": "smoke-s1",
                "usage": {"cost": 0.0001},
            }
        return {
            "operation": "DONE",
            "target": None,
            "choice": None,
            "confidence": 0.99,
            "target_confidence": 0.99,
            "target_margin": 0.95,
            "latency_ms": 20,
            "model": "smoke-s1",
            "usage": {"cost": 0.0001},
        }

    with (
        managed_web_server() as base_url,
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", side_effect=fake_choose_action),
        patch(
            "reflex.agent.resolve_escalation",
            return_value=SupervisorDecision(
                action_type="COMPLETE_TASK",
                summary_for_user="Smoke test başarıyla tamamlandı.",
                reasoning="Task completed on fixture.",
                model="smoke-s2",
                latency_ms=10,
            ),
        ),
    ):
        start_res = await reflex_start(
            goal="Sayfadaki ilk bağlantıya tıkla ve tamamla",
            url=f"{base_url}/qa/blog",
            max_steps=5,
            show_browser=False,
        )
        sid = start_res["session_id"]
        try:
            from reflex.mcp_server import _get_worker

            worker = _get_worker(sid)
            observed_ports.append(worker.port)
            assert worker.port != FORBIDDEN_SHARED_PORT

            wait_res = await reflex_wait(session_id=sid, timeout_s=25)
            assert wait_res["status"] == "done"
            assert wait_res["steps"] >= 1
            assert len(wait_res["last_actions"]) >= 1
            assert "127.0.0.1" in wait_res["visited_hosts"]
            assert wait_res["final_answer"] is not None
        finally:
            stop_res = await reflex_stop(session_id=sid)
            assert stop_res["status"] == "stopped"

    assert observed_ports and all(p != 9333 for p in observed_ports)


@pytest.mark.anyio
async def test_f1_single_active_session_rejects_second_start_until_stopped() -> None:
    """F1: reflex_start rejects a second session while an unstopped session exists (including done/blocked/error),
    and accepts a new start after reflex_stop is called.
    """
    def fake_factory(url: str, goal: str, config: Any = None) -> _FakeReflexSession:
        return _FakeReflexSession(url=url, goal=goal, config=config, scenario="ask_then_done")

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=fake_factory),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9510"
        mock_iso.return_value.__exit__.return_value = None

        start1 = await reflex_start(goal="İlk görev", url="https://example.com", show_browser=False)
        sid1 = start1["session_id"]

        # 1. While waiting_user, second start is rejected
        wait1 = await reflex_wait(session_id=sid1, timeout_s=5)
        assert wait1["status"] == "waiting_user"
        with pytest.raises(RuntimeError, match=f"Aktif oturum {sid1} var; önce reflex_stop çağırın"):
            await reflex_start(goal="İkinci görev", url="https://example.com", show_browser=False)

        # 2. Even after reaching 'done' (browser still open), second start is rejected
        await reflex_reply(session_id=sid1, text="İzmir")
        wait2 = await reflex_wait(session_id=sid1, timeout_s=5)
        assert wait2["status"] == "done"
        with pytest.raises(RuntimeError, match=f"Aktif oturum {sid1} var; önce reflex_stop çağırın"):
            await reflex_start(goal="İkinci görev", url="https://example.com", show_browser=False)

        # 3. After reflex_stop, a new reflex_start succeeds
        stop1 = await reflex_stop(session_id=sid1)
        assert stop1["status"] == "stopped"

        start2 = await reflex_start(goal="Yeni görev", url="https://example.com", show_browser=False)
        sid2 = start2["session_id"]
        assert sid2 != sid1
        await reflex_stop(session_id=sid2)


@pytest.mark.anyio
async def test_f2_stdio_disconnect_without_stop_cleans_up_chromium_and_profile() -> None:
    """F2: Real stdio_client + patched LLMs; closing connection without calling reflex_stop
    terminates the isolated Chromium process and deletes /tmp/reflex_eval_profile_* within 5s.
    """
    server_script = """
import json
import subprocess
from unittest.mock import patch
import reflex.agent
import reflex.browser.isolated as iso
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation
import reflex.mcp_server as srv

spawn_info = {}
orig_popen = subprocess.Popen

def tracking_popen(cmd, *args, **kwargs):
    for arg in cmd:
        if isinstance(arg, str) and arg.startswith("--remote-debugging-port="):
            spawn_info["port"] = int(arg.split("=", 1)[1])
        elif isinstance(arg, str) and arg.startswith("--user-data-dir="):
            spawn_info["profile_dir"] = arg.split("=", 1)[1]
    return orig_popen(cmd, *args, **kwargs)

def fake_choose_action(page, goal, history, config, **kwargs):
    return {
        "operation": "DONE",
        "target": None,
        "choice": None,
        "confidence": 0.99,
        "target_confidence": 0.99,
        "target_margin": 0.95,
        "latency_ms": 10,
        "model": "f2-mock-s1",
        "usage": {"cost": 0.0},
    }

def fake_resolve_escalation(**kwargs):
    return SupervisorDecision(
        action_type="COMPLETE_TASK",
        summary_for_user=json.dumps(spawn_info),
        reasoning="Done",
        model="f2-mock-s2",
        latency_ms=10,
    )

with (
    patch.object(iso.subprocess, "Popen", side_effect=tracking_popen),
    patch.object(reflex.agent, "interpret_user_message", return_value=UserMessageInterpretation()),
    patch.object(reflex.agent, "choose_action", side_effect=fake_choose_action),
    patch.object(reflex.agent, "resolve_escalation", side_effect=fake_resolve_escalation),
):
    srv.main()
"""

    port: int | None = None
    profile_dir: Path | None = None

    with managed_web_server() as base_url:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", server_script],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                r_start = await session.call_tool(
                    "reflex_start",
                    {
                        "goal": "F2 yetim Chromium temizlik testi",
                        "url": f"{base_url}/qa/blog",
                        "max_steps": 3,
                        "show_browser": False,
                    },
                )
                start_data = json.loads(r_start.content[0].text)
                sid = start_data["session_id"]

                r_wait = await session.call_tool(
                    "reflex_wait",
                    {"session_id": sid, "timeout_s": 25},
                )
                wait_data = json.loads(r_wait.content[0].text)
                assert wait_data["status"] == "done"
                info = json.loads(wait_data["final_answer"])
                port = int(info["port"])
                profile_dir = Path(info["profile_dir"])

                # Before disconnecting (without calling reflex_stop), both Chromium and profile exist
                assert is_port_open("127.0.0.1", port) is True
                assert profile_dir.exists() is True
                # Intentionally do NOT call reflex_stop!

    assert port is not None and profile_dir is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not is_port_open("127.0.0.1", port, timeout=0.15) and not profile_dir.exists():
            break
        time.sleep(0.1)

    assert is_port_open("127.0.0.1", port, timeout=0.2) is False
    assert profile_dir.exists() is False


def test_f3_all_mcp_tools_are_async() -> None:
    """F3: All 5 MCP tools must be async def so they do not block the anyio event loop."""
    import inspect

    for fn in (
        reflex_start,
        reflex_wait,
        reflex_reply,
        reflex_confirm,
        reflex_stop,
    ):
        assert inspect.iscoroutinefunction(fn), f"{fn.__name__} must be async def"


def test_f4_tool_descriptions_include_client_instructions() -> None:
    """F4: Tool descriptions must instruct client models on flow and explicit user confirmation."""
    confirm_doc = reflex_confirm.__doc__ or ""
    assert "approved=true YALNIZCA insan kullanıcı bu eylemi bu konuşmada" in confirm_doc
    assert "Kendi kararınla asla onaylama" in confirm_doc
    assert "pending_action" in confirm_doc

    start_doc = reflex_start.__doc__ or ""
    wait_doc = reflex_wait.__doc__ or ""
    for doc in (start_doc, wait_doc):
        assert "running" in doc
        assert "waiting_user" in doc
        assert "waiting_confirmation" in doc
        assert "reflex_wait" in doc
        assert "reflex_reply" in doc


def test_f5_main_sets_reflex_env_path_when_cwd_has_no_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """F5: When REFLEX_ENV_PATH is unset and cwd/.env is missing, main() sets REFLEX_ENV_PATH to repo root .env."""
    import os
    import reflex.mcp_server as srv
    from reflex.config import get_env_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REFLEX_ENV_PATH", raising=False)
    assert not (tmp_path / ".env").exists()

    with patch.object(srv.mcp, "run", return_value=None):
        srv.main()

    expected = Path(srv.__file__).resolve().parents[1] / ".env"
    assert os.environ.get("REFLEX_ENV_PATH") == str(expected)
    assert get_env_path() == expected


def test_f6_extract_final_answer_ignores_s2_reasoning_and_elicitation_stub_cleanup() -> None:
    """F6: _extract_final_answer must not fall back to S2 internal reasoning, and _client_supports_elicitation requires explicit flag or real request_context."""
    from reflex.mcp_server import _client_supports_elicitation, _extract_final_answer

    snap_only_reasoning = {
        "context": {"chat_log": []},
        "escalations": [{"reasoning": "Internal S2 chain-of-thought only", "summary_for_user": ""}],
    }
    assert _extract_final_answer(object(), snap_only_reasoning) is None

    snap_with_summary = {
        "context": {"chat_log": []},
        "escalations": [{"reasoning": "Internal S2 thought", "summary_for_user": "Kullanıcıya özet yanıt"}],
    }
    assert _extract_final_answer(object(), snap_with_summary) == "Kullanıcıya özet yanıt"

    class _BareStubWithElicitOnly:
        async def elicit(self, **kwargs: Any) -> Any:
            return None

        @property
        def request_context(self) -> Any:
            raise ValueError("Context is not available outside of a request")

    assert _client_supports_elicitation(_BareStubWithElicitOnly()) is False







