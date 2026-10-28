# ruff: noqa: BLE001, S110
"""Reflex benchmark evaluation runner.

Reads task definition files (JSON or JSONL), executes the ReflexSession.step()
loop per task, handles paused states (user replies, confirmations), evaluates JS
check expressions, records error classifications and per-step model telemetry,
and outputs JSONL results plus aggregate summary metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlsplit

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig

logger = logging.getLogger("reflex.eval")


@contextlib.contextmanager
def managed_web_server(
    host: str = "127.0.0.1",
    port: int = 0,
    web_dir: Path | str | None = None,
) -> Generator[str, None, None]:
    """Provide a local HTTP server serving reflex/web fixtures with clean URL rewrites."""
    target_dir = Path(web_dir) if web_dir else (Path(__file__).resolve().parent.parent / "reflex" / "web")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {target_dir}")

    static_map = {
        "/fixture": "/fixture.html",
        "/qa/survey": "/qa_survey.html",
        "/qa/ranking": "/qa_ranking.html",
        "/qa/blog": "/qa_blog.html",
        "/qa/shop": "/qa_shop.html",
        "/qa/tournament": "/qa_tournament.html",
        "/qa/crm": "/qa_crm.html",
    }

    class _FixtureHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(target_dir), **kwargs)

        def log_message(self, *args: Any) -> None:
            pass  # suppress request logging in stdout

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            clean_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
            if clean_path in static_map:
                mapped = static_map[clean_path]
                self.path = mapped + (f"?{parsed.query}" if parsed.query else "")
            super().do_GET()

    server = http.server.ThreadingHTTPServer((host, port), _FixtureHandler)
    assigned_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{assigned_port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def load_tasks(tasks_path: str | Path) -> list[dict[str, Any]]:
    """Load task definitions from a JSON or JSONL file.

    Supported formats:
    - JSON array: `[{"id": "...", "url": "...", "goal": "..."}, ...]`
    - JSON object with tasks key: `{"tasks": [...]}`
    - Single JSON object: `{"id": "...", "url": "...", "goal": "..."}`
    - JSONL: One task JSON per line.
    """
    path = Path(tasks_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    tasks: list[dict[str, Any]] = []

    # First attempt parsing as standard JSON
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            tasks = [t for t in data if isinstance(t, dict)]
        elif isinstance(data, dict):
            if "tasks" in data and isinstance(data["tasks"], list):
                tasks = [t for t in data["tasks"] if isinstance(t, dict)]
            else:
                tasks = [data]
        else:
            raise TypeError(f"Unexpected JSON data type in {path}: {type(data)}")
    except json.JSONDecodeError:
        # Fall back to JSONL parsing (line by line)
        for line_num, line in enumerate(raw_text.splitlines(), start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            try:
                task_obj = json.loads(line_str)
                if isinstance(task_obj, dict):
                    tasks.append(task_obj)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {exc}") from exc

    # Validate and normalize task entries
    normalized: list[dict[str, Any]] = []
    for idx, t in enumerate(tasks, start=1):
        task_id = str(t.get("id") or f"task_{idx}")
        url = t.get("url")
        goal = t.get("goal")
        if not url or not goal:
            raise ValueError(f"Task {task_id} missing required 'url' or 'goal' field: {t}")

        max_steps = t.get("max_steps")
        user_replies = t.get("user_replies")
        if user_replies is None:
            user_replies_list = []
        elif isinstance(user_replies, list):
            user_replies_list = [str(r) for r in user_replies]
        else:
            user_replies_list = [str(user_replies)]

        check_kind = str(t.get("check_kind", "correctness"))
        normalized.append(
            {
                "id": task_id,
                "url": str(url),
                "goal": str(goal),
                "max_steps": int(max_steps) if max_steps is not None else None,
                "user_replies": user_replies_list,
                "auto_confirm": bool(t.get("auto_confirm", False)),
                "check": str(t["check"]) if t.get("check") else None,
                "check_kind": check_kind,
            }
        )

    return normalized


def _session_known_cost_usd(session: ReflexSession) -> float:
    snap = session.snapshot()
    metrics = snap.get("metrics", {})
    known = metrics.get("known_cost_usd")
    if isinstance(known, (int, float)):
        return float(known)
    total = metrics.get("total_cost_usd", 0.0)
    if isinstance(total, (int, float)):
        return float(total)
    return 0.0


def _session_has_unknown_cost(session: ReflexSession) -> bool:
    snap = session.snapshot()
    metrics = snap.get("metrics", {})
    if metrics.get("total_cost_usd") == "bilinmiyor":
        return True
    for rec_list in (
        snap.get("decisions", []),
        snap.get("escalations", []),
        snap.get("text_calls", []),
        snap.get("interpret_calls", []),
    ):
        for item in rec_list:
            if isinstance(item, dict) and (
                item.get("cost_usd") == "bilinmiyor"
                or (isinstance(item.get("usage"), dict) and item["usage"].get("cost") == "bilinmiyor")
            ):
                return True
    return False


def _extract_asked_question(session: ReflexSession | None, snap: dict[str, Any]) -> str | None:
    """Extract the question text asked by the agent when terminating with status == 'asked_user'."""
    if isinstance(snap.get("asked_question"), str) and snap["asked_question"].strip():
        return snap["asked_question"].strip()
    if session is not None:
        last_q = getattr(session, "last_asked_question", None)
        if isinstance(last_q, str) and last_q.strip():
            return last_q.strip()
        pending_q = getattr(session, "pending_question", None)
        if isinstance(pending_q, dict) and isinstance(pending_q.get("question"), str) and pending_q["question"].strip():
            return pending_q["question"].strip()
        if isinstance(pending_q, str) and pending_q.strip():
            return pending_q.strip()
    ctx_dict = snap.get("context") or {}
    chat_msgs = ctx_dict.get("chat_messages") if isinstance(ctx_dict, dict) else None
    if isinstance(chat_msgs, list):
        for msg in reversed(chat_msgs):
            if isinstance(msg, dict) and msg.get("role") == "agent":
                txt = msg.get("text") or msg.get("content") or msg.get("message")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    for tc in reversed(snap.get("text_calls") or []):
        if isinstance(tc, dict) and isinstance(tc.get("ask_user"), str) and tc["ask_user"].strip():
            return tc["ask_user"].strip()
    for esc in reversed(snap.get("escalations") or []):
        if isinstance(esc, dict):
            for k in ("question_for_user", "question"):
                if isinstance(esc.get(k), str) and esc[k].strip():
                    return esc[k].strip()
    return None


def run_task(
    task: dict[str, Any],
    config: ReflexConfig | None = None,
    default_max_steps: int = 25,
    base_url: str | None = None,
    budget_usd: float | None = None,
) -> dict[str, Any]:
    """Execute a single evaluation task in ReflexSession with complete error shielding.

    Tracks step count, actual model IDs used per step (S1, S2, text helper),
    latency, cost, and evaluates the objective completion state via JS check expression.
    """
    task_id = task["id"]
    raw_url = task["url"]
    if "{BASE}" in raw_url and base_url is None:
        with managed_web_server() as local_base:
            return run_task(
                task,
                config=config,
                default_max_steps=default_max_steps,
                base_url=local_base,
                budget_usd=budget_usd,
            )
    url = raw_url.replace("{BASE}", base_url) if base_url else raw_url
    goal = task["goal"]
    max_steps = int(task.get("max_steps") or default_max_steps)
    user_replies: list[str] = list(task.get("user_replies", []))
    auto_confirm: bool = bool(task.get("auto_confirm", False))
    check_expr: str | None = task.get("check")
    check_kind: str = str(task.get("check_kind", "correctness"))

    cfg = (config or ReflexConfig()).model_copy(deep=True)
    cfg.max_steps = max(cfg.max_steps, max_steps + 10)

    start_time = time.perf_counter()
    session: ReflexSession | None = None
    step_details: list[dict[str, Any]] = []
    task_status: str = "ready"
    last_error: str | None = None
    error_step: int | None = None
    asked_question: str | None = None
    check_result: Any = None
    check_error: str | None = None

    try:
        # 1. Initialize session with exception shielding
        try:
            session = ReflexSession(url=url, goal=goal, config=cfg)
        except Exception as exc:
            task_status = "error"
            error_step = 0
            last_error = f"Session initialization failed: {exc}"
            elapsed_ms = round((time.perf_counter() - start_time) * 1000)
            return {
                "id": task_id,
                "url": url,
                "goal": goal,
                "success": False,
                "score": 0.0,
                "status": task_status,
                "check_kind": check_kind,
                "steps": 0,
                "s1_steps": 0,
                "s2_escalations": 0,
                "s1_errors": 0,
                "cost_usd": 0.0,
                "latencies_ms": [],
                "median_latency_ms": 0,
                "models_used": [],
                "step_details": [],
                "asked_question": None,
                "error_step": error_step,
                "last_error": last_error,
                "check_result": None,
                "check_error": None,
                "elapsed_ms": elapsed_ms,
            }

        if session.status == "error":
            error_step = len(session.history)

        # Check if initial session setup (e.g. constraint extraction) already exceeded budget or has unknown cost
        if budget_usd is not None:
            if _session_has_unknown_cost(session):
                task_status = "budget_unknown"
                session.status = "budget_unknown"
                session.last_error = (
                    "Budget unknown: call with unknown cost ('bilinmiyor') encountered while --budget-usd is set"
                )
            elif _session_known_cost_usd(session) > budget_usd:
                task_status = "budget_exceeded"
                session.status = "budget_exceeded"
                session.last_error = (
                    f"Budget exceeded: ${_session_known_cost_usd(session):.6f} > ${budget_usd:.6f}"
                )

        # 2. Execution step loop
        while (
            session.status not in {"done", "blocked", "error", "budget_exceeded", "budget_unknown"}
            and len(session.history) < max_steps
        ):
            if budget_usd is not None:
                if _session_has_unknown_cost(session):
                    task_status = "budget_unknown"
                    session.status = "budget_unknown"
                    session.last_error = (
                        "Budget unknown: call with unknown cost ('bilinmiyor') encountered while --budget-usd is set"
                    )
                    break
                if _session_known_cost_usd(session) > budget_usd:
                    task_status = "budget_exceeded"
                    session.status = "budget_exceeded"
                    session.last_error = (
                        f"Budget exceeded: ${_session_known_cost_usd(session):.6f} > ${budget_usd:.6f}"
                    )
                    break

            attempt_step = len(session.history) + 1
            if session.status == "waiting_user":
                if not user_replies:
                    task_status = "asked_user"
                    session.status = "asked_user"
                    break
                reply = user_replies.pop(0)
                try:
                    session.reply_user(reply)
                except Exception as exc:
                    task_status = "error"
                    session.status = "error"
                    error_step = attempt_step
                    session.last_error = f"reply_user failed: {exc}"
                    break
                if session.status == "error" and error_step is None:
                    error_step = attempt_step
            elif session.status == "waiting_confirmation":
                try:
                    session.confirm_action(approved=auto_confirm)
                except Exception as exc:
                    task_status = "error"
                    session.status = "error"
                    error_step = attempt_step
                    session.last_error = f"confirm_action failed: {exc}"
                    break
                if session.status == "error" and error_step is None:
                    error_step = attempt_step
            elif session.status == "ready":
                try:
                    session.step()
                except Exception as exc:
                    task_status = "error"
                    session.status = "error"
                    error_step = attempt_step
                    session.last_error = f"step failed: {exc}"
                    break
                if session.status == "error" and error_step is None:
                    error_step = attempt_step
            else:
                break

            if budget_usd is not None:
                if _session_has_unknown_cost(session):
                    task_status = "budget_unknown"
                    session.status = "budget_unknown"
                    session.last_error = (
                        "Budget unknown: call with unknown cost ('bilinmiyor') encountered while --budget-usd is set"
                    )
                    break
                if _session_known_cost_usd(session) > budget_usd:
                    task_status = "budget_exceeded"
                    session.status = "budget_exceeded"
                    session.last_error = (
                        f"Budget exceeded: ${_session_known_cost_usd(session):.6f} > ${budget_usd:.6f}"
                    )
                    break

        # Check termination status
        if task_status in {"asked_user", "budget_exceeded", "budget_unknown"}:
            pass
        elif (
            session.status not in {"done", "blocked", "error", "budget_exceeded", "budget_unknown"}
            and len(session.history) >= max_steps
        ):
            task_status = "max_steps_exceeded"
        else:
            task_status = session.status

        last_error = session.last_error

        # 3. Evaluate JS check expression if provided
        if check_expr and session.browser and task_status not in {"budget_exceeded", "budget_unknown"}:
            try:
                check_result = session.browser.evaluate(check_expr)
            except Exception as exc:
                check_error = f"Check evaluation failed: {exc}"
                check_result = None

        # 4. Determine success and score
        if task_status in {"error", "asked_user", "budget_exceeded", "budget_unknown"}:
            success = False
            score = 0.0
        elif check_expr is not None:
            if isinstance(check_result, dict):
                is_done = bool(check_result.get("done", False))
                score_val = check_result.get("score")
                if score_val is not None:
                    score = float(score_val)
                    success = is_done and (score > 0.0)
                else:
                    score = 1.0 if is_done else 0.0
                    success = is_done
            elif isinstance(check_result, bool):
                success = check_result
                score = 1.0 if success else 0.0
            elif isinstance(check_result, (int, float)):
                score = float(check_result)
                success = score > 0.0
            else:
                success = bool(check_result)
                score = 1.0 if success else 0.0
        else:
            # Fallback when no check expression is specified
            success = task_status == "done"
            score = 1.0 if success else 0.0

        # 5. Extract snapshot metrics and per-step model telemetry
        snap = session.snapshot()
        history = snap.get("history", [])
        decisions = snap.get("decisions", [])
        escalations = snap.get("escalations", [])
        text_calls = snap.get("text_calls", [])
        interpret_calls = snap.get("interpret_calls", [])

        if task_status == "asked_user":
            asked_question = _extract_asked_question(session, snap)
        if task_status == "error" and error_step is None:
            error_step = len(history)

        # Track models used across every step and tier
        models_used_set: set[str] = set()
        for d in decisions:
            if d.get("model"):
                models_used_set.add(str(d["model"]))
        for e in escalations:
            if e.get("model"):
                models_used_set.add(str(e["model"]))
        for t in text_calls:
            if t.get("model"):
                models_used_set.add(str(t["model"]))
        for ic in interpret_calls:
            if ic.get("model"):
                models_used_set.add(str(ic["model"]))
        for h in history:
            if h.get("model"):
                models_used_set.add(str(h["model"]))
            if h.get("text_helper"):
                models_used_set.add(str(h["text_helper"]))

        # Build detailed per-step records
        for idx, h in enumerate(history, start=1):
            step_details.append(
                {
                    "step": idx,
                    "actor": h.get("actor"),
                    "operation": h.get("operation"),
                    "target": h.get("target"),
                    "model": h.get("model"),
                    "text_helper": h.get("text_helper"),
                    "latency_ms": h.get("latency_ms"),
                    "s1_ms": h.get("s1_ms", 0),
                    "s2_ms": h.get("s2_ms", 0),
                    "act_ms": h.get("act_ms", 0),
                    "observe_ms": h.get("observe_ms", 0),
                    "step_ms": h.get("step_ms", h.get("latency_ms", 0)),
                    "url": h.get("url"),
                }
            )

        latencies = [
            h["latency_ms"]
            for h in history
            if isinstance(h.get("latency_ms"), (int, float))
        ]
        median_lat = sorted(latencies)[len(latencies) // 2] if latencies else 0

        elapsed_ms = round((time.perf_counter() - start_time) * 1000)
        metrics = snap.get("metrics", {})
        raw_total_cost = metrics.get("total_cost_usd", 0.0)
        cost_usd_out: float | str = (
            "bilinmiyor"
            if raw_total_cost == "bilinmiyor"
            else round(float(raw_total_cost), 6)
        )
        known_cost_out = round(
            float(
                metrics.get(
                    "known_cost_usd",
                    0.0 if raw_total_cost == "bilinmiyor" else raw_total_cost,
                )
            ),
            6,
        )

        return {
            "id": task_id,
            "url": url,
            "goal": goal,
            "success": bool(success),
            "score": round(float(score), 4),
            "status": task_status,
            "check_kind": check_kind,
            "steps": len(history),
            "s1_steps": metrics.get("s1_steps", 0),
            "s2_escalations": len(escalations),
            "s1_errors": metrics.get("s1_errors", 0),
            "cost_usd": cost_usd_out,
            "known_cost_usd": known_cost_out,
            "usage": metrics.get("usage", {}),
            "latencies_ms": latencies,
            "median_latency_ms": median_lat,
            "models_used": sorted(models_used_set),
            "step_details": step_details,
            "asked_question": asked_question,
            "error_step": error_step,
            "last_error": last_error,
            "check_result": check_result,
            "check_error": check_error,
            "elapsed_ms": elapsed_ms,
        }

    except Exception as exc:
        task_status = "error"
        last_error = f"Unexpected runner error: {exc}"
        elapsed_ms = round((time.perf_counter() - start_time) * 1000)
        return {
            "id": task_id,
            "url": url,
            "goal": goal,
            "success": False,
            "score": 0.0,
            "status": task_status,
            "check_kind": check_kind,
            "steps": len(step_details),
            "s1_steps": 0,
            "s2_escalations": 0,
            "s1_errors": 0,
            "cost_usd": 0.0,
            "latencies_ms": [],
            "median_latency_ms": 0,
            "models_used": [],
            "step_details": step_details,
            "asked_question": None,
            "error_step": error_step if error_step is not None else len(step_details),
            "last_error": last_error,
            "check_result": check_result,
            "check_error": check_error,
            "elapsed_ms": elapsed_ms,
        }
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def compute_summary(
    task_results: list[dict[str, Any]],
    budget_usd: float | None = None,
) -> dict[str, Any]:
    """Aggregate individual task results into a comprehensive evaluation summary."""
    total_tasks = len(task_results)
    successful_tasks = sum(1 for r in task_results if r.get("success", False))
    failed_tasks = total_tasks - successful_tasks
    error_tasks = sum(1 for r in task_results if r.get("status") == "error")
    asked_user_tasks = sum(1 for r in task_results if r.get("status") == "asked_user")
    budget_exceeded_tasks = sum(1 for r in task_results if r.get("status") == "budget_exceeded")
    budget_unknown_tasks = sum(1 for r in task_results if r.get("status") == "budget_unknown")

    success_rate = (successful_tasks / total_tasks) if total_tasks > 0 else 0.0

    total_steps = sum(r.get("steps", 0) for r in task_results)
    avg_steps = (total_steps / total_tasks) if total_tasks > 0 else 0.0

    total_s2 = sum(r.get("s2_escalations", 0) for r in task_results)
    s2_rate = (total_s2 / total_steps) if total_steps > 0 else 0.0

    has_unknown_cost = any(r.get("cost_usd") == "bilinmiyor" for r in task_results)
    known_cost = 0.0
    for r in task_results:
        kc = r.get("known_cost_usd")
        if isinstance(kc, (int, float)):
            known_cost += float(kc)
        else:
            c = r.get("cost_usd", 0.0)
            if isinstance(c, (int, float)):
                known_cost += float(c)

    total_cost_out: float | str = "bilinmiyor" if has_unknown_cost else round(known_cost, 6)
    known_cost_out = round(known_cost, 6)

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "total_tokens": 0,
    }
    for r in task_results:
        u = r.get("usage")
        if isinstance(u, dict):
            total_usage["prompt_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            total_usage["completion_tokens"] += int(u.get("completion_tokens", 0) or 0)
            total_usage["cached_tokens"] += int(u.get("cached_tokens", 0) or 0)
            total_usage["total_tokens"] += int(u.get("total_tokens", 0) or 0)

    all_latencies: list[float | int] = []
    for r in task_results:
        all_latencies.extend(r.get("latencies_ms", []))

    if all_latencies:
        sorted_lat = sorted(all_latencies)
        p50_lat_ms = sorted_lat[len(sorted_lat) // 2]
    else:
        p50_lat_ms = 0

    p50_lat_sec = round(p50_lat_ms / 1000.0, 4)

    all_models: set[str] = set()
    for r in task_results:
        for m in r.get("models_used", []):
            if m:
                all_models.add(str(m))

    by_check_kind: dict[str, dict[str, int]] = {}
    for r in task_results:
        kind = r.get("check_kind", "correctness")
        if kind not in by_check_kind:
            by_check_kind[kind] = {"total": 0, "successful": 0}
        by_check_kind[kind]["total"] += 1
        if r.get("success", False):
            by_check_kind[kind]["successful"] += 1

    if budget_exceeded_tasks > 0 or (budget_usd is not None and known_cost > budget_usd):
        run_status = "budget_exceeded"
    elif budget_unknown_tasks > 0 or (budget_usd is not None and has_unknown_cost):
        run_status = "budget_unknown"
    else:
        run_status = "completed"

    total_elapsed_ms = sum(int(r.get("elapsed_ms", 0) or 0) for r in task_results)
    avg_elapsed_ms = round(total_elapsed_ms / total_tasks, 2) if total_tasks > 0 else 0.0
    total_elapsed_sec = round(total_elapsed_ms / 1000.0, 2)
    avg_elapsed_sec = round(avg_elapsed_ms / 1000.0, 2)
    task_elapsed_ms = {
        str(r.get("id", f"task_{idx}")): int(r.get("elapsed_ms", 0) or 0)
        for idx, r in enumerate(task_results, start=1)
    }

    return {
        "status": run_status,
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "error_tasks": error_tasks,
        "asked_user_tasks": asked_user_tasks,
        "budget_exceeded_tasks": budget_exceeded_tasks,
        "budget_unknown_tasks": budget_unknown_tasks,
        "success_rate": round(success_rate, 4),
        "total_steps": total_steps,
        "avg_steps_per_task": round(avg_steps, 2),
        "total_s2_escalations": total_s2,
        "s2_escalation_rate": round(s2_rate, 4),
        "total_cost_usd": total_cost_out,
        "known_cost_usd": known_cost_out,
        "usage": total_usage,
        "p50_latency_ms": p50_lat_ms,
        "p50_latency_seconds": p50_lat_sec,
        "total_elapsed_ms": total_elapsed_ms,
        "total_elapsed_seconds": total_elapsed_sec,
        "avg_elapsed_ms_per_task": avg_elapsed_ms,
        "avg_elapsed_seconds_per_task": avg_elapsed_sec,
        "task_elapsed_ms": task_elapsed_ms,
        "models_used": sorted(all_models),
        "by_check_kind": by_check_kind,
    }


def format_summary_report(summary: dict[str, Any]) -> str:
    """Format evaluation summary into a readable report banner."""
    cost_val = summary.get("total_cost_usd", 0.0)
    cost_str = "bilinmiyor" if cost_val == "bilinmiyor" else f"${float(cost_val):.6f}"
    lines = [
        "=" * 64,
        "                   REFLEX EVALUATION REPORT",
        "=" * 64,
        f"Status:                  {summary.get('status', 'completed')}",
        f"Total Tasks:             {summary['total_tasks']}",
        f"Success Rate:            {summary['success_rate'] * 100:.2f}% ({summary['successful_tasks']}/{summary['total_tasks']})",
        f"Failed Tasks:            {summary['failed_tasks']} (Errors: {summary['error_tasks']}, Asked User: {summary.get('asked_user_tasks', 0)}, Budget Exceeded: {summary.get('budget_exceeded_tasks', 0)}, Budget Unknown: {summary.get('budget_unknown_tasks', 0)})",
        f"Total Steps:             {summary['total_steps']} (Avg / Task: {summary['avg_steps_per_task']:.2f})",
        f"S2 Escalations:          {summary['total_s2_escalations']}",
        f"S2 Escalation Rate:      {summary['s2_escalation_rate'] * 100:.2f}%",
        f"Total Cost:              {cost_str}",
        f"p50 Latency:             {summary['p50_latency_ms']} ms ({summary['p50_latency_seconds']} s)",
        f"Total Elapsed:           {summary.get('total_elapsed_ms', 0)} ms ({summary.get('total_elapsed_seconds', 0.0):.2f} s | Avg / Task: {summary.get('avg_elapsed_ms_per_task', 0.0):.0f} ms / {summary.get('avg_elapsed_seconds_per_task', 0.0):.2f} s)",
        f"Models Used:             {', '.join(summary['models_used']) if summary['models_used'] else 'None'}",
    ]
    if summary.get("task_elapsed_ms"):
        t_strs = [f"{tid}: {ms} ms ({ms / 1000.0:.1f} s)" for tid, ms in summary["task_elapsed_ms"].items()]
        lines.append(f"Task Durations:          {', '.join(t_strs)}")
    if summary.get("by_check_kind"):
        kind_strs = [
            f"{k}: {v['successful']}/{v['total']}"
            for k, v in summary["by_check_kind"].items()
        ]
        lines.append(f"Check Kinds:             {', '.join(kind_strs)}")
    lines.append("=" * 64)
    return "\n".join(lines)


def run_eval(
    tasks_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: ReflexConfig | None = None,
    default_max_steps: int = 25,
    verbose: bool = True,
    base_url: str | None = None,
    web_dir: Path | str | None = None,
    budget_usd: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute complete evaluation suite, save JSONL results and return summary."""
    tasks = load_tasks(tasks_path)
    if verbose:
        print(f"[eval] Loaded {len(tasks)} tasks from {tasks_path}", flush=True)

    has_base = any("{BASE}" in t["url"] for t in tasks)
    if has_base and base_url is None:
        with managed_web_server(web_dir=web_dir) as local_base:
            return run_eval(
                tasks_path=tasks_path,
                output_path=output_path,
                summary_path=summary_path,
                config=config,
                default_max_steps=default_max_steps,
                verbose=verbose,
                base_url=local_base,
                web_dir=web_dir,
                budget_usd=budget_usd,
            )

    task_results: list[dict[str, Any]] = []
    output_file = Path(output_path) if output_path else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear or create file
        output_file.write_text("", encoding="utf-8")

    cumulative_cost = 0.0
    for idx, task in enumerate(tasks, start=1):
        if budget_usd is not None and cumulative_cost > budget_usd:
            break

        if verbose:
            task_display_url = task["url"].replace("{BASE}", base_url) if base_url else task["url"]
            print(f"[eval] [{idx}/{len(tasks)}] Starting task {task['id']} ({task_display_url})...", flush=True)

        remaining_budget = (budget_usd - cumulative_cost) if budget_usd is not None else None
        res = run_task(
            task,
            config=config,
            default_max_steps=default_max_steps,
            base_url=base_url,
            budget_usd=remaining_budget,
        )

        task_known_cost = res.get("known_cost_usd")
        if isinstance(task_known_cost, (int, float)):
            cumulative_cost += float(task_known_cost)
        elif isinstance(res.get("cost_usd"), (int, float)):
            cumulative_cost += float(res["cost_usd"])

        if budget_usd is not None and res.get("cost_usd") == "bilinmiyor" and res.get("status") != "budget_unknown":
            res["status"] = "budget_unknown"
            res["success"] = False
            res["score"] = 0.0
            res["last_error"] = (
                "Budget unknown: call with unknown cost ('bilinmiyor') encountered while --budget-usd is set"
            )
        elif budget_usd is not None and cumulative_cost > budget_usd and res.get("status") != "budget_exceeded":
            res["status"] = "budget_exceeded"
            res["success"] = False
            res["score"] = 0.0
            res["last_error"] = f"Budget exceeded: ${cumulative_cost:.6f} > ${budget_usd:.6f}"

        task_results.append(res)

        if output_file:
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()

        if verbose:
            status_symbol = "✓" if res["success"] else "✗"
            cost_disp = (
                "bilinmiyor"
                if res["cost_usd"] == "bilinmiyor"
                else f"${float(res['cost_usd']):.6f}"
            )
            print(
                f"[eval] [{idx}/{len(tasks)}] {status_symbol} {res['id']}: "
                f"status={res['status']}, steps={res['steps']}, "
                f"score={res['score']}, cost={cost_disp}",
                flush=True,
            )
            if res.get("asked_question"):
                print(f"       asked_question: {res['asked_question']}", flush=True)
            if res.get("last_error"):
                err_step_suffix = f" (step {res['error_step']})" if res.get("error_step") is not None else ""
                print(f"       last_error{err_step_suffix}: {res['last_error']}", flush=True)

        if res.get("status") == "budget_exceeded":
            if verbose:
                print(f"[eval] Stopping evaluation: budget_exceeded (${cumulative_cost:.6f} > ${budget_usd:.6f})", flush=True)
            break
        if res.get("status") == "budget_unknown":
            if verbose:
                print("[eval] Stopping evaluation: budget_unknown (unknown cost 'bilinmiyor' encountered with --budget-usd)", flush=True)
            break

    summary = compute_summary(task_results, budget_usd=budget_usd)

    if summary_path:
        sum_path = Path(summary_path)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("\n" + format_summary_report(summary), flush=True)

    return task_results, summary


def is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Check if a TCP port is currently open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@contextlib.contextmanager
def managed_isolated_browser(
    port: int = 9444,
    headless: bool = True,
) -> Generator[str, None, None]:
    """Provide an isolated Chromium browser instance and environment lifecycle for evaluation.

    Guarantees:
    - Rejects port 9333 with ValueError.
    - Sets BU_CDP_URL to http://127.0.0.1:{port} and BU_NAME to eval_{port}.
    - If no process is listening on {port}, launches an isolated Chromium subprocess
      with a dedicated temporary --user-data-dir.
    - Cleanly shuts down the subprocess and removes the temporary profile directory on exit.
    - Restores previous environment variables and CDP hooks in finally:.
    """
    env_cdp = os.environ.get("BU_CDP_URL")
    if env_cdp:
        try:
            parsed_port = urlsplit(env_cdp).port
            if parsed_port == 9333:
                raise ValueError("Port 9333 is reserved for external monitoring; eval runner must not use port 9333.")
            if parsed_port and int(port) == 9444:
                port = parsed_port
        except ValueError:
            raise
        except Exception:
            pass

    if int(port) == 9333:
        raise ValueError("Port 9333 is reserved for external monitoring; eval runner must not use port 9333.")

    cdp_url = f"http://127.0.0.1:{port}"
    old_cdp_url = os.environ.get("BU_CDP_URL")
    old_bu_name = os.environ.get("BU_NAME")
    os.environ["BU_CDP_URL"] = cdp_url
    os.environ["BU_NAME"] = f"eval_{port}"

    old_ensure = None
    try:
        import reflex.browser.cdp as r_cdp
        old_ensure = r_cdp.ensure_chromium_cdp
        r_cdp.ensure_chromium_cdp = lambda *args, **kwargs: None
    except Exception:
        pass

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    proc: subprocess.Popen[Any] | None = None

    try:
        if not is_port_open("127.0.0.1", port):
            binary = (
                shutil.which("chromium")
                or shutil.which("chromium-browser")
                or shutil.which("google-chrome")
            )
            if binary:
                temp_dir_obj = tempfile.TemporaryDirectory(prefix="reflex_eval_profile_")
                user_data_dir = temp_dir_obj.name

                cmd = [
                    binary,
                    f"--remote-debugging-port={port}",
                    f"--user-data-dir={user_data_dir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                    "--disk-cache-size=1048576",
                    "--media-cache-size=1048576",
                    "--window-size=1280,900",
                    "--lang=tr-TR",
                ]
                if headless:
                    cmd.insert(1, "--headless=new")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if is_port_open("127.0.0.1", port):
                        time.sleep(0.2)
                        break
                    if proc.poll() is not None:
                        logger.warning("Chromium process exited prematurely with code %s", proc.returncode)
                        break
                    time.sleep(0.1)
            else:
                logger.warning("No Chromium executable found in PATH; running without spawned browser.")

        yield cdp_url

    finally:
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            except Exception as exc:
                logger.warning("Error terminating Chromium subprocess: %s", exc)

        if temp_dir_obj is not None:
            try:
                temp_dir_obj.cleanup()
            except Exception as exc:
                logger.warning("Error cleaning up temporary directory: %s", exc)

        try:
            from browser_harness import admin
            if hasattr(admin, "stop_remote_daemon"):
                admin.stop_remote_daemon(f"eval_{port}")
        except Exception:
            pass

        if old_ensure is not None:
            try:
                import reflex.browser.cdp as r_cdp
                r_cdp.ensure_chromium_cdp = old_ensure
            except Exception:
                pass

        if old_cdp_url is not None:
            os.environ["BU_CDP_URL"] = old_cdp_url
        else:
            os.environ.pop("BU_CDP_URL", None)

        if old_bu_name is not None:
            os.environ["BU_NAME"] = old_bu_name
        else:
            os.environ.pop("BU_NAME", None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reflex Autonomous Agent Evaluation Runner")
    parser.add_argument("tasks", nargs="?", help="Path to JSON or JSONL tasks file")
    parser.add_argument("--tasks", dest="tasks_flag", help="Alternative flag for tasks file")
    parser.add_argument("-o", "--output", help="Path to write JSONL results")
    parser.add_argument("-s", "--summary", help="Path to write summary JSON")
    parser.add_argument("--decision-backend", choices=["local_turnike", "openrouter_jev"], help="Override S1 decision backend")
    parser.add_argument("--supervisor-model", help="Override S2 supervisor model")
    parser.add_argument("--allow-model-fallback", action="store_true", help="Allow fallback models on error")
    parser.add_argument("--max-steps", type=int, default=25, help="Default maximum steps per task (default: 25)")
    parser.add_argument("--screenshots", action="store_true", help="Capture screenshots during observe")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--web-port", type=int, default=0, help="Port for local fixture web server (default: 0 for ephemeral port)")
    parser.add_argument("--cdp-port", type=int, default=9444, help="CDP port for isolated Chromium instance (default: 9444; 9333 is forbidden)")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode (default: headless)")
    parser.add_argument("--budget-usd", type=float, default=None, help="Maximum budget in USD; stops with budget_exceeded if exceeded")

    args = parser.parse_args(argv)
    tasks_file = args.tasks or args.tasks_flag
    if not tasks_file:
        parser.error("A tasks file must be provided either as positional argument or via --tasks.")

    if args.cdp_port == 9333:
        raise ValueError("Port 9333 is reserved for external monitoring; eval runner must not use port 9333.")

    cfg = ReflexConfig()
    if args.decision_backend:
        cfg.decision_backend = args.decision_backend
    if args.supervisor_model:
        cfg.supervisor_model = args.supervisor_model
    if args.allow_model_fallback:
        cfg.allow_model_fallback = True
    if args.screenshots:
        cfg.screenshots = True

    try:
        with (
            managed_web_server(port=args.web_port) as local_base_url,
            managed_isolated_browser(port=args.cdp_port, headless=not args.headed),
        ):
            _, summary = run_eval(
                tasks_path=tasks_file,
                output_path=args.output,
                summary_path=args.summary,
                config=cfg,
                default_max_steps=args.max_steps,
                verbose=not args.quiet,
                base_url=local_base_url,
                budget_usd=args.budget_usd,
            )
            if summary.get("status") in {"budget_exceeded", "budget_unknown"}:
                return 1
            return 0 if summary["failed_tasks"] == 0 else 1
    except Exception as exc:
        print(f"[eval] Fatal evaluation runner error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
