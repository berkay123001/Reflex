# ruff: noqa: BLE001
"""Reflex MCP Server (`FastMCP`, stdio transport).

Exposes autonomous browser agent sessions (`ReflexSession`) as MCP tools:
- `reflex_start(goal, url="https://www.google.com", max_steps=40, show_browser=True)`
- `reflex_wait(session_id, timeout_s=50)`
- `reflex_reply(session_id, text)`
- `reflex_confirm(session_id, approved, alternative_instruction="")`
- `reflex_stop(session_id)`

Concurrency invariant:
Each session runs inside its own dedicated background thread and its own isolated
Chromium instance (never port 9333). Only the session's background thread touches
the `ReflexSession` instance; `reflex_reply`, `reflex_confirm`, and `reflex_stop`
enqueue commands to that thread.

Elicitation (M3):
When the MCP client advertises elicitation support (e.g., Claude Code >= 2.1.76),
`reflex_wait` directly elicits user input for `waiting_user` and action approval
for `waiting_confirmation` via `ctx.elicit(...)`. Approval is never granted
automatically on the server; declined, cancelled, or unanswered confirmation
elicitations always resolve to `approved=False`.
"""

from __future__ import annotations

from concurrent.futures import Future
import inspect
import logging
import os
from pathlib import Path
import queue
import signal
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import anyio
from mcp import types as mcp_types
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from reflex.agent import ReflexSession
from reflex.browser.isolated import (
    find_free_port,
    managed_isolated_browser,
)
from reflex.config import ReflexConfig

logger = logging.getLogger("reflex.mcp_server")

MAX_WAIT_TIMEOUT_S = 50.0


class UserReplyElicitationSchema(BaseModel):
    """Schema for `waiting_user` elicitation via `ctx.elicit`."""

    text: str = Field(
        default="",
        description="Your answer or instruction for the browser agent",
    )


class ActionConfirmElicitationSchema(BaseModel):
    """Schema for `waiting_confirmation` elicitation via `ctx.elicit`."""

    approved: bool = Field(
        default=False,
        description="Set true to approve executing this action, or false to reject it",
    )
    alternative_instruction: str = Field(
        default="",
        description="Optional alternative instruction if rejecting the action",
    )


def _client_supports_elicitation(ctx: Context[Any, Any, Any] | Any | None) -> bool:
    """Return True if `ctx` represents an MCP client session that supports elicitation."""
    if ctx is None:
        return False
    elicit_fn = getattr(ctx, "elicit", None)
    if not callable(elicit_fn):
        return False

    explicit_flag = getattr(ctx, "supports_elicitation", None)
    if isinstance(explicit_flag, bool):
        return explicit_flag

    try:
        req_ctx = ctx.request_context
    except Exception:
        # If `ctx` is a lightweight test stub providing `.elicit` without `request_context`,
        # allow elicitation unless explicitly disabled.
        return "_request_context" not in getattr(ctx, "__dict__", {})

    if req_ctx is None:
        return False

    session = getattr(req_ctx, "session", None)
    if session is None:
        return False

    check_cap = getattr(session, "check_client_capability", None)
    if callable(check_cap):
        try:
            return bool(
                check_cap(
                    mcp_types.ClientCapabilities(
                        elicitation=mcp_types.ElicitationCapability(),
                    )
                )
            )
        except Exception:
            pass

    client_params = getattr(session, "_client_params", None)
    if client_params is not None:
        caps = getattr(client_params, "capabilities", None)
        return getattr(caps, "elicitation", None) is not None

    return False


def _extract_final_answer(session: Any, snap: dict[str, Any]) -> str | None:
    """Extract the final answer when a session finishes with `status == 'done'`."""
    direct = snap.get("final_answer") or getattr(session, "final_answer", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    ctx_dict = snap.get("context") or {}
    if isinstance(ctx_dict, dict):
        chat_log = ctx_dict.get("chat_log") or []
        if isinstance(chat_log, list):
            for msg in reversed(chat_log):
                if not isinstance(msg, dict) or msg.get("role") not in {"assistant", "agent"}:
                    continue
                content = str(msg.get("content") or msg.get("text") or "").strip()
                if content:
                    return content
        chat_msgs = ctx_dict.get("chat_messages") or []
        if isinstance(chat_msgs, list):
            for msg in reversed(chat_msgs):
                if isinstance(msg, dict) and msg.get("role") in {"assistant", "agent"}:
                    content = str(msg.get("text") or msg.get("content") or "").strip()
                    if content:
                        return content

    for esc in reversed(snap.get("escalations") or []):
        if isinstance(esc, dict):
            for key in ("summary_for_user", "final_answer", "reasoning"):
                val = esc.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def _extract_question(session: Any, snap: dict[str, Any]) -> str | None:
    """Extract the pending question when a session is in `status == 'waiting_user'`."""
    for candidate in (
        snap.get("asked_question"),
        snap.get("question"),
        getattr(session, "last_asked_question", None),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    pending_q = getattr(session, "pending_question", None)
    if isinstance(pending_q, dict):
        q_str = pending_q.get("question")
        if isinstance(q_str, str) and q_str.strip():
            return q_str.strip()
    elif isinstance(pending_q, str) and pending_q.strip():
        return pending_q.strip()

    ctx_dict = snap.get("context") or {}
    if isinstance(ctx_dict, dict):
        for list_key in ("chat_log", "chat_messages"):
            msgs = ctx_dict.get(list_key)
            if isinstance(msgs, list):
                for msg in reversed(msgs):
                    if isinstance(msg, dict) and msg.get("role") in {"assistant", "agent"}:
                        txt = msg.get("content") or msg.get("text") or msg.get("message")
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()

    for tc in reversed(snap.get("text_calls") or []):
        if isinstance(tc, dict) and isinstance(tc.get("ask_user"), str) and tc["ask_user"].strip():
            return tc["ask_user"].strip()

    for esc in reversed(snap.get("escalations") or []):
        if isinstance(esc, dict):
            for key in ("question_for_user", "question"):
                val = esc.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def _extract_pending_action(session: Any, snap: dict[str, Any]) -> dict[str, Any] | str | None:
    """Extract the pending irreversible action details when in `status == 'waiting_confirmation'`."""
    raw = (
        snap.get("pending_confirmation")
        or snap.get("pending_action")
        or getattr(session, "pending_confirmation", None)
        or getattr(session, "pending_action", None)
    )
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"summary": raw, "reason": "", "action_label": raw}
    if isinstance(raw, dict):
        act_obj = raw.get("action")
        action_label = raw.get("action_label") or (act_obj.get("label") if isinstance(act_obj, dict) else None)
        out: dict[str, Any] = {
            "summary": str(raw.get("summary") or action_label or ""),
            "reason": str(raw.get("reason") or ""),
            "action_label": action_label,
        }
        for k, v in raw.items():
            if k not in out and k not in {"action", "usage", "signature"}:
                out[k] = v
        return out
    return str(raw)


def _format_last_actions(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the last 5 action records formatted concisely."""
    tail = [h for h in history if isinstance(h, dict)][-5:]
    formatted: list[dict[str, Any]] = []
    for idx, h in enumerate(tail, start=max(1, len(history) - len(tail) + 1)):
        item: dict[str, Any] = {
            "step": h.get("step", idx),
            "operation": h.get("operation"),
            "action": h.get("action") or h.get("label") or h.get("target") or "",
        }
        for key in ("actor", "kind", "target", "text", "url", "effective_progress"):
            if h.get(key) is not None:
                item[key] = h.get(key)
        formatted.append(item)
    return formatted


def _visited_hosts(history: list[dict[str, Any]], snap: dict[str, Any] | None = None) -> list[str]:
    """Distinct visited hosts in order (consecutive duplicates collapsed)."""
    hosts: list[str] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        host = urlsplit(str(h.get("url") or "")).hostname or ""
        if host and (not hosts or hosts[-1] != host):
            hosts.append(host)
    if not hosts and isinstance(snap, dict):
        ctx = snap.get("context") or {}
        if isinstance(ctx, dict):
            for u in ctx.get("visited_urls") or []:
                host = urlsplit(str(u or "")).hostname or ""
                if host and (not hosts or hosts[-1] != host):
                    hosts.append(host)
        page_url = str((snap.get("page") or {}).get("url") or "")
        host = urlsplit(page_url).hostname or ""
        if host and (not hosts or hosts[-1] != host):
            hosts.append(host)
    return hosts


def _extract_cost_usd(session: Any, snap: dict[str, Any]) -> float | str:
    metrics = snap.get("metrics") or {}
    if isinstance(metrics, dict) and "total_cost_usd" in metrics and metrics["total_cost_usd"] is not None:
        val = metrics["total_cost_usd"]
        if isinstance(val, (int, float)):
            return float(val)
        return str(val)
    direct = getattr(session, "cost_usd", None)
    if isinstance(direct, (int, float)):
        return float(direct)
    return 0.0


class _SessionWorker:
    """Owns a single `ReflexSession` and its isolated browser on a dedicated background thread."""

    def __init__(
        self,
        *,
        session_id: str,
        goal: str,
        url: str,
        max_steps: int,
        show_browser: bool,
        port: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.goal = goal
        self.url = url
        self.max_steps = max(1, int(max_steps))
        self.show_browser = bool(show_browser)
        self.port = port if (port is not None and int(port) != 0) else find_free_port()

        self._cond = threading.Condition()
        self._cmd_queue: queue.Queue[tuple[str, dict[str, Any], Future[dict[str, Any]]]] = queue.Queue()
        self._init_event = threading.Event()
        self._stop_event = threading.Event()

        self.status: str = "starting"
        self._initial_status: str = "running"
        self._payload: dict[str, Any] = {
            "session_id": self.session_id,
            "status": "running",
            "steps": 0,
            "last_actions": [],
            "question": None,
            "pending_action": None,
            "final_answer": None,
            "visited_hosts": [],
            "cost_usd": 0.0,
        }
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"reflex-mcp-{self.session_id}",
            daemon=True,
        )

    def start(self, init_timeout_s: float = 15.0) -> dict[str, Any]:
        self._thread.start()
        self._init_event.wait(timeout=init_timeout_s)
        with self._cond:
            status_out = (
                self._initial_status
                if self.status == "starting"
                else ("running" if self.status in {"starting", "ready"} else self.status)
            )
            return {"session_id": self.session_id, "status": status_out}

    def _sync_from_session_locked(self, session: Any, *, actively_running: bool) -> None:
        try:
            snap = session.snapshot() if hasattr(session, "snapshot") else {}
        except Exception:
            snap = {}
        if not isinstance(snap, dict):
            snap = {}

        raw_status = str(getattr(session, "status", None) or snap.get("status") or "ready")
        if raw_status == "ready" and actively_running:
            effective_status = "running"
        elif raw_status == "ready":
            effective_status = "running"
        else:
            effective_status = raw_status

        history = snap.get("history")
        if not isinstance(history, list):
            history = list(getattr(session, "history", None) or [])

        metrics = snap.get("metrics") or {}
        steps = int(metrics.get("steps", len(history))) if isinstance(metrics, dict) else len(history)

        question = _extract_question(session, snap) if effective_status == "waiting_user" else None
        pending_action = _extract_pending_action(session, snap) if effective_status == "waiting_confirmation" else None
        final_answer = _extract_final_answer(session, snap) if effective_status == "done" else None
        visited = _visited_hosts(history, snap)
        cost_usd = _extract_cost_usd(session, snap)
        last_error = snap.get("last_error") or getattr(session, "last_error", None)

        self.status = effective_status
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "status": effective_status,
            "steps": steps,
            "last_actions": _format_last_actions(history),
            "question": question,
            "pending_action": pending_action,
            "final_answer": final_answer,
            "visited_hosts": visited,
            "cost_usd": cost_usd,
        }
        if last_error:
            payload["last_error"] = str(last_error)
        self._payload = payload
        self._cond.notify_all()

    def _set_error_locked(self, message: str) -> None:
        self.status = "error"
        self._payload = {
            **self._payload,
            "status": "error",
            "last_error": str(message),
        }
        self._cond.notify_all()

    def _drain_commands(self, session: Any) -> bool:
        """Process all queued commands on the worker thread. Return True if stop was requested."""
        while True:
            try:
                cmd, kwargs, fut = self._cmd_queue.get_nowait()
            except queue.Empty:
                return False

            if self._execute_command(session, cmd, kwargs, fut):
                return True

    def _execute_command(
        self,
        session: Any,
        cmd: str,
        kwargs: dict[str, Any],
        fut: Future[dict[str, Any]],
    ) -> bool:
        """Execute a single queued command on the worker thread. Return True if stopping."""
        if cmd == "stop":
            self._stop_event.set()
            if not fut.done():
                fut.set_result({"session_id": self.session_id, "status": "stopped"})
            return True

        try:
            if cmd == "reply":
                session.reply_user(str(kwargs.get("text") or ""))
            elif cmd == "confirm":
                try:
                    sig = inspect.signature(session.confirm_action)
                    accepts_alt = len(sig.parameters) >= 2 or any(
                        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                        for p in sig.parameters.values()
                    )
                except Exception:
                    accepts_alt = True
                if accepts_alt:
                    session.confirm_action(
                        bool(kwargs.get("approved", False)),
                        str(kwargs.get("alternative_instruction") or ""),
                    )
                else:
                    session.confirm_action(bool(kwargs.get("approved", False)))
            with self._cond:
                is_ready = str(getattr(session, "status", "ready")) == "ready"
                self._sync_from_session_locked(session, actively_running=is_ready)
                res = dict(self._payload)
            if not fut.done():
                fut.set_result(res)
        except Exception as exc:
            with self._cond:
                self._set_error_locked(str(exc))
            if not fut.done():
                fut.set_exception(exc)
        return False

    def _run_loop(self) -> None:
        session: Any = None
        try:
            with managed_isolated_browser(port=self.port, headless=not self.show_browser):
                cfg = ReflexConfig()
                cfg.max_steps = max(cfg.max_steps, self.max_steps)

                try:
                    session = ReflexSession(url=self.url, goal=self.goal, config=cfg)
                except Exception as exc:
                    with self._cond:
                        self._initial_status = "error"
                        self._set_error_locked(str(exc))
                        self._init_event.set()
                    return

                with self._cond:
                    is_ready = str(getattr(session, "status", "ready")) == "ready"
                    self._sync_from_session_locked(session, actively_running=is_ready)
                    self._initial_status = self.status
                    self._init_event.set()

                while not self._stop_event.is_set():
                    if self._drain_commands(session):
                        break

                    raw_status = str(getattr(session, "status", "ready"))
                    if raw_status == "ready":
                        hist = getattr(session, "history", None) or []
                        if len(hist) >= self.max_steps:
                            session.status = "blocked"
                            session.last_error = f"Max steps ({self.max_steps}) reached"
                            with self._cond:
                                self._sync_from_session_locked(session, actively_running=False)
                            continue

                        try:
                            session.step()
                            with self._cond:
                                next_status = str(getattr(session, "status", "ready"))
                                next_hist = getattr(session, "history", None) or []
                                if next_status == "ready" and len(next_hist) >= self.max_steps:
                                    session.status = "blocked"
                                    session.last_error = f"Max steps ({self.max_steps}) reached"
                                    self._sync_from_session_locked(session, actively_running=False)
                                else:
                                    self._sync_from_session_locked(
                                        session,
                                        actively_running=(next_status == "ready"),
                                    )
                        except Exception as exc:
                            session.status = "error"
                            session.last_error = str(exc)
                            with self._cond:
                                self._sync_from_session_locked(session, actively_running=False)
                    else:
                        # Paused (waiting_user / waiting_confirmation) or terminal (done / blocked / error):
                        # Only the session thread drives session methods; wait for queued commands.
                        try:
                            cmd, kwargs, fut = self._cmd_queue.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if self._execute_command(session, cmd, kwargs, fut):
                            break
        except Exception as exc:
            with self._cond:
                self._initial_status = "error"
                self._set_error_locked(str(exc))
                self._init_event.set()
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            with self._cond:
                if self._stop_event.is_set():
                    self.status = "stopped"
                    self._payload = {**self._payload, "status": "stopped"}
                self._cond.notify_all()
            while True:
                try:
                    _cmd, _kwargs, fut = self._cmd_queue.get_nowait()
                    if not fut.done():
                        fut.set_result({"session_id": self.session_id, "status": "stopped"})
                except queue.Empty:
                    break

    def wait_for_state_change(self, timeout_s: float = MAX_WAIT_TIMEOUT_S) -> dict[str, Any]:
        """Wait until session leaves `running`/`starting` or `timeout_s` (clamped <= 50s) elapses."""
        clamped = min(max(float(timeout_s), 0.0), MAX_WAIT_TIMEOUT_S)
        deadline = time.monotonic() + clamped
        with self._cond:
            while self.status in {"starting", "running", "ready"} and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            return dict(self._payload)

    def enqueue_reply(self, text: str, timeout_s: float = 30.0) -> dict[str, Any]:
        if not text or not str(text).strip():
            raise ValueError("Reply text must be non-empty")
        fut: Future[dict[str, Any]] = Future()
        with self._cond:
            if self.status == "stopped" or self._stop_event.is_set():
                raise RuntimeError(f"Session {self.session_id} is already stopped")
            self.status = "running"
            self._payload = {**self._payload, "status": "running", "question": None}
            self._cmd_queue.put(("reply", {"text": str(text).strip()}, fut))
            self._cond.notify_all()
        return fut.result(timeout=timeout_s)

    def enqueue_confirm(
        self,
        approved: bool,
        alternative_instruction: str = "",
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        fut: Future[dict[str, Any]] = Future()
        with self._cond:
            if self.status == "stopped" or self._stop_event.is_set():
                raise RuntimeError(f"Session {self.session_id} is already stopped")
            self.status = "running"
            self._payload = {**self._payload, "status": "running", "pending_action": None}
            self._cmd_queue.put(
                (
                    "confirm",
                    {
                        "approved": bool(approved),
                        "alternative_instruction": str(alternative_instruction or ""),
                    },
                    fut,
                )
            )
            self._cond.notify_all()
        return fut.result(timeout=timeout_s)

    def stop(self, timeout_s: float = 10.0) -> dict[str, Any]:
        fut: Future[dict[str, Any]] = Future()
        with self._cond:
            self._stop_event.set()
            self._cmd_queue.put(("stop", {}, fut))
            self._cond.notify_all()
        try:
            fut.result(timeout=timeout_s)
        except Exception:
            pass
        self._thread.join(timeout=timeout_s)
        with self._cond:
            self.status = "stopped"
            self._payload = {**self._payload, "status": "stopped"}
            return {"session_id": self.session_id, "status": "stopped"}


_SESSIONS_LOCK = threading.Lock()
_SESSIONS: dict[str, _SessionWorker] = {}


def _get_worker(session_id: str) -> _SessionWorker:
    with _SESSIONS_LOCK:
        worker = _SESSIONS.get(session_id)
    if worker is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    return worker


async def _handle_elicitation_if_supported(
    worker: _SessionWorker,
    payload: dict[str, Any],
    ctx: Context[Any, Any, Any] | Any | None,
    remaining_timeout_s: float,
) -> dict[str, Any]:
    """If client supports elicitation, handle `waiting_user` / `waiting_confirmation` via `ctx.elicit`."""
    while _client_supports_elicitation(ctx) and payload.get("status") in {"waiting_user", "waiting_confirmation"}:
        status = payload.get("status")
        if status == "waiting_user":
            question = str(payload.get("question") or "Additional information is needed to continue:")
            try:
                elicit_res = await ctx.elicit(message=question, schema=UserReplyElicitationSchema)
            except Exception as exc:
                logger.debug("Elicitation for waiting_user failed or unsupported: %s", exc)
                return payload

            action = getattr(elicit_res, "action", None)
            data = getattr(elicit_res, "data", None)
            reply_text = ""
            if action == "accept" and data is not None:
                if isinstance(data, dict):
                    reply_text = str(data.get("text") or data.get("reply") or "").strip()
                elif isinstance(data, str):
                    reply_text = data.strip()
                else:
                    reply_text = str(getattr(data, "text", None) or getattr(data, "reply", None) or "").strip()

            if not reply_text:
                # Declined, cancelled, or empty reply -> return waiting_user state without auto-answering
                return payload

            await anyio.to_thread.run_sync(worker.enqueue_reply, reply_text)
            wait_budget = max(0.5, min(remaining_timeout_s, MAX_WAIT_TIMEOUT_S))
            payload = await anyio.to_thread.run_sync(worker.wait_for_state_change, wait_budget)

        elif status == "waiting_confirmation":
            pending = payload.get("pending_action") or {}
            if isinstance(pending, dict):
                summary = str(pending.get("summary") or pending.get("action_label") or "irreversible action")
                reason = str(pending.get("reason") or "")
                prompt_msg = f"Confirmation required for {summary}" + (f" ({reason})" if reason else "")
            else:
                prompt_msg = f"Confirmation required: {pending}"

            approved = False
            alt_instruction = ""
            try:
                elicit_res = await ctx.elicit(message=prompt_msg, schema=ActionConfirmElicitationSchema)
                action = getattr(elicit_res, "action", None)
                data = getattr(elicit_res, "data", None)
                # Approval is NEVER automatic: only explicit accept + approved=True is True.
                # Declined, cancelled, or unanswered elicitation -> approved=False.
                if action == "accept" and data is not None:
                    if isinstance(data, dict):
                        approved = bool(data.get("approved", False))
                        alt_instruction = str(data.get("alternative_instruction") or "")
                    else:
                        approved = bool(getattr(data, "approved", False))
                        alt_instruction = str(getattr(data, "alternative_instruction", "") or "")
            except Exception as exc:
                logger.debug("Elicitation for waiting_confirmation failed; defaulting to approved=False: %s", exc)
                approved = False
                alt_instruction = ""

            await anyio.to_thread.run_sync(worker.enqueue_confirm, approved, alt_instruction)
            wait_budget = max(0.5, min(remaining_timeout_s, MAX_WAIT_TIMEOUT_S))
            payload = await anyio.to_thread.run_sync(worker.wait_for_state_change, wait_budget)

    return payload


mcp = FastMCP("reflex")


@mcp.tool()
async def reflex_start(
    goal: str,
    url: str = "https://www.google.com",
    max_steps: int = 40,
    show_browser: bool = True,
) -> dict[str, Any]:
    """Start an autonomous Reflex browser session in its own isolated browser instance.

    Constraint (tek aktif oturum): Süreç başına yalnızca tek bir aktif oturum açılabilir
    (`done`, `blocked` veya `error` durumunda olup tarayıcısı hâlâ açık olan oturumlar dahil).
    Yeni bir oturum başlatmadan önce mevcut oturum için `reflex_stop` çağırın.

    Akış (Flow):
    1. `reflex_start` çağrısından dönen `session_id` ile `reflex_wait(session_id)` çağırın.
    2. `status == "running"` dönerse (süre doldu ama görev devam ediyor) tekrar `reflex_wait` çağırın.
    3. `status == "waiting_user"` dönerse `question` alanını insan kullanıcıya sorun (veya bu konuşmada
       kullanıcının zaten verdiği bilinen bilgiyle) `reflex_reply` çağırıp ardından tekrar `reflex_wait` çağırın.
    4. `status == "waiting_confirmation"` dönerse `pending_action` özetini insan kullanıcıya gösterip
       açık onayını isteyin; yanıtına göre `reflex_confirm` çağırıp ardından `reflex_wait` çağırın.
    5. `status` `done`, `blocked` veya `error` olduğunda işiniz bitince `reflex_stop(session_id)` çağırın.
    """
    if not goal or not goal.strip():
        raise ValueError("Supply a non-empty goal")
    with _SESSIONS_LOCK:
        for existing_id, existing_worker in list(_SESSIONS.items()):
            if existing_worker.status == "stopped":
                _SESSIONS.pop(existing_id, None)
                continue
            raise RuntimeError(f"Aktif oturum {existing_id} var; önce reflex_stop çağırın")
        session_id = uuid.uuid4().hex[:12]
        worker = _SessionWorker(
            session_id=session_id,
            goal=goal.strip(),
            url=url.strip() or "https://www.google.com",
            max_steps=max_steps,
            show_browser=show_browser,
        )
        _SESSIONS[session_id] = worker
    return await anyio.to_thread.run_sync(worker.start)


@mcp.tool()
async def reflex_wait(
    session_id: str,
    timeout_s: float = 50.0,
    ctx: Context[Any, Any, Any] | None = None,
) -> dict[str, Any]:
    """Wait until the session status changes (paused/done/blocked/error) or timeout_s (<=50s) expires.

    Akış (Flow):
    - `status == "running"`: Görev hâlâ çalışıyor; tekrar `reflex_wait` çağır.
    - `status == "waiting_user"`: `question` sorusunu kullanıcıya sor veya bilinen bilgiyle
      `reflex_reply(session_id, text)` çağır, ardından tekrar `reflex_wait` çağır.
    - `status == "waiting_confirmation"`: `pending_action` özetini kullanıcıya göster ve
      kullanıcıya sor; kendi kararınla asla onaylama. Yanıt aldıktan sonra `reflex_confirm` çağır.
    - `status` in (`"done"`, `"blocked"`, `"error"`): Sonucu (`final_answer` / `error`) kullanıcıya ilet
      ve tarayıcıyı kapatmak için `reflex_stop(session_id)` çağır.

    İstemci MCP elicitation (`ctx.elicit`) destekliyorsa `waiting_user` ve
    `waiting_confirmation` durumları `reflex_wait` içinde doğrudan kullanıcıya sorulur.
    """
    worker = _get_worker(session_id)
    clamped = min(max(float(timeout_s), 0.0), MAX_WAIT_TIMEOUT_S)
    t0 = time.monotonic()
    payload = await anyio.to_thread.run_sync(worker.wait_for_state_change, clamped)
    elapsed = time.monotonic() - t0
    remaining = max(0.5, clamped - elapsed)
    return await _handle_elicitation_if_supported(worker, payload, ctx, remaining)


@mcp.tool()
async def reflex_reply(
    session_id: str,
    text: str,
) -> dict[str, Any]:
    """Queue a user reply (`session.reply_user`) when the session is in `waiting_user` status.

    Yanıtı kuyruğa aldıktan sonra oturumun devam etmesini beklemek için `reflex_wait(session_id)` çağırın.
    """
    worker = _get_worker(session_id)
    return await anyio.to_thread.run_sync(worker.enqueue_reply, text)


@mcp.tool()
async def reflex_confirm(
    session_id: str,
    approved: bool,
    alternative_instruction: str = "",
) -> dict[str, Any]:
    """Queue an action confirmation (`session.confirm_action`) when the session is in `waiting_confirmation` status.

    KRİTİK KURAL: approved=true YALNIZCA insan kullanıcı bu eylemi bu konuşmada
    açıkça onayladıysa. Kendi kararınla asla onaylama.
    Onay istemeden önce `pending_action` özetini (`summary`, `action_label`, `reason`, `url`)
    mutlaka kullanıcıya göster ve açık yanıtını bekle. Kullanıcı reddederse veya alternatif
    bir talimat verirse `approved=false` (ve varsa `alternative_instruction`) gönder.
    Ardından `reflex_wait(session_id)` çağır.
    """
    worker = _get_worker(session_id)
    return await anyio.to_thread.run_sync(
        worker.enqueue_confirm,
        approved,
        alternative_instruction,
    )


@mcp.tool()
async def reflex_stop(
    session_id: str,
) -> dict[str, Any]:
    """Stop the session and clean up its isolated browser instance.

    Görev tamamlandığında (`done`), tıkandığında (`blocked`/`error`) veya yeni bir
    `reflex_start` çağrısı yapmadan önce mutlaka çağrılmalıdır.
    """
    with _SESSIONS_LOCK:
        worker = _SESSIONS.pop(session_id, None)
    if worker is None:
        raise KeyError(f"Unknown session_id: {session_id}")
    return await anyio.to_thread.run_sync(worker.stop)


def _stop_all_sessions(timeout_s: float = 4.0) -> None:
    """Stop all active session workers and clean up their isolated browsers."""
    with _SESSIONS_LOCK:
        workers = list(_SESSIONS.values())
        _SESSIONS.clear()
    for worker in workers:
        try:
            worker.stop(timeout_s=timeout_s)
        except Exception as exc:
            logger.debug("Error stopping session %s during shutdown: %s", worker.session_id, exc)


def main() -> None:
    """Entry point for `reflex-mcp` stdio server."""
    if not os.environ.get("REFLEX_ENV_PATH") and not (Path.cwd() / ".env").exists():
        os.environ["REFLEX_ENV_PATH"] = str(Path(__file__).resolve().parents[1] / ".env")

    def _raise_system_exit(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _raise_system_exit)
    signal.signal(signal.SIGINT, _raise_system_exit)
    try:
        mcp.run(transport="stdio")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _stop_all_sessions(timeout_s=4.0)
        finally:
            signal.signal(signal.SIGTERM, prev_sigterm)
            signal.signal(signal.SIGINT, prev_sigint)


if __name__ == "__main__":
    main()
