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
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig

logger = logging.getLogger("reflex.eval")


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

        normalized.append(
            {
                "id": task_id,
                "url": str(url),
                "goal": str(goal),
                "max_steps": int(max_steps) if max_steps is not None else None,
                "user_replies": user_replies_list,
                "auto_confirm": bool(t.get("auto_confirm", False)),
                "check": str(t["check"]) if t.get("check") else None,
            }
        )

    return normalized


def run_task(
    task: dict[str, Any],
    config: ReflexConfig | None = None,
    default_max_steps: int = 25,
) -> dict[str, Any]:
    """Execute a single evaluation task in ReflexSession with complete error shielding.

    Tracks step count, actual model IDs used per step (S1, S2, text helper),
    latency, cost, and evaluates the objective completion state via JS check expression.
    """
    task_id = task["id"]
    url = task["url"]
    goal = task["goal"]
    max_steps = int(task.get("max_steps") or default_max_steps)
    user_replies: list[str] = list(task.get("user_replies", []))
    auto_confirm: bool = bool(task.get("auto_confirm", False))
    check_expr: str | None = task.get("check")

    cfg = (config or ReflexConfig()).model_copy(deep=True)
    cfg.max_steps = max(cfg.max_steps, max_steps + 10)

    start_time = time.perf_counter()
    session: ReflexSession | None = None
    step_details: list[dict[str, Any]] = []
    task_status: str = "ready"
    last_error: str | None = None
    check_result: Any = None
    check_error: str | None = None

    try:
        # 1. Initialize session with exception shielding
        try:
            session = ReflexSession(url=url, goal=goal, config=cfg)
        except Exception as exc:
            task_status = "error"
            last_error = f"Session initialization failed: {exc}"
            elapsed_ms = round((time.perf_counter() - start_time) * 1000)
            return {
                "id": task_id,
                "url": url,
                "goal": goal,
                "success": False,
                "score": 0.0,
                "status": task_status,
                "steps": 0,
                "s1_steps": 0,
                "s2_escalations": 0,
                "s1_errors": 0,
                "cost_usd": 0.0,
                "latencies_ms": [],
                "median_latency_ms": 0,
                "models_used": [],
                "step_details": [],
                "last_error": last_error,
                "check_result": None,
                "check_error": None,
                "elapsed_ms": elapsed_ms,
            }

        # 2. Execution step loop
        while session.status not in {"done", "blocked", "error"} and len(session.history) < max_steps:
            if session.status == "waiting_user":
                reply = user_replies.pop(0) if user_replies else ""
                try:
                    session.reply_user(reply)
                except Exception as exc:
                    session.status = "error"
                    session.last_error = f"reply_user failed: {exc}"
                    break
            elif session.status == "waiting_confirmation":
                try:
                    session.confirm_action(approved=auto_confirm)
                except Exception as exc:
                    session.status = "error"
                    session.last_error = f"confirm_action failed: {exc}"
                    break
            elif session.status == "ready":
                try:
                    session.step()
                except Exception as exc:
                    session.status = "error"
                    session.last_error = f"step failed: {exc}"
                    break
            else:
                break

        # Check termination status
        if session.status not in {"done", "blocked", "error"} and len(session.history) >= max_steps:
            task_status = "max_steps_exceeded"
        else:
            task_status = session.status

        last_error = session.last_error

        # 3. Evaluate JS check expression if provided
        if check_expr and session.browser:
            try:
                check_result = session.browser.evaluate(check_expr)
            except Exception as exc:
                check_error = f"Check evaluation failed: {exc}"
                check_result = None

        # 4. Determine success and score
        if task_status == "error":
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

        return {
            "id": task_id,
            "url": url,
            "goal": goal,
            "success": bool(success),
            "score": round(float(score), 4),
            "status": task_status,
            "steps": len(history),
            "s1_steps": snap.get("metrics", {}).get("s1_steps", 0),
            "s2_escalations": len(escalations),
            "s1_errors": snap.get("metrics", {}).get("s1_errors", 0),
            "cost_usd": round(float(snap.get("metrics", {}).get("total_cost_usd", 0.0)), 6),
            "latencies_ms": latencies,
            "median_latency_ms": median_lat,
            "models_used": sorted(models_used_set),
            "step_details": step_details,
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
            "steps": len(step_details),
            "s1_steps": 0,
            "s2_escalations": 0,
            "s1_errors": 0,
            "cost_usd": 0.0,
            "latencies_ms": [],
            "median_latency_ms": 0,
            "models_used": [],
            "step_details": step_details,
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


def compute_summary(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate individual task results into a comprehensive evaluation summary."""
    total_tasks = len(task_results)
    successful_tasks = sum(1 for r in task_results if r.get("success", False))
    failed_tasks = total_tasks - successful_tasks
    error_tasks = sum(1 for r in task_results if r.get("status") == "error")

    success_rate = (successful_tasks / total_tasks) if total_tasks > 0 else 0.0

    total_steps = sum(r.get("steps", 0) for r in task_results)
    avg_steps = (total_steps / total_tasks) if total_tasks > 0 else 0.0

    total_s2 = sum(r.get("s2_escalations", 0) for r in task_results)
    s2_rate = (total_s2 / total_steps) if total_steps > 0 else 0.0

    total_cost = sum(r.get("cost_usd", 0.0) for r in task_results)

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

    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "failed_tasks": failed_tasks,
        "error_tasks": error_tasks,
        "success_rate": round(success_rate, 4),
        "total_steps": total_steps,
        "avg_steps_per_task": round(avg_steps, 2),
        "total_s2_escalations": total_s2,
        "s2_escalation_rate": round(s2_rate, 4),
        "total_cost_usd": round(total_cost, 6),
        "p50_latency_ms": p50_lat_ms,
        "p50_latency_seconds": p50_lat_sec,
        "models_used": sorted(all_models),
    }


def format_summary_report(summary: dict[str, Any]) -> str:
    """Format evaluation summary into a readable report banner."""
    lines = [
        "=" * 64,
        "                   REFLEX EVALUATION REPORT",
        "=" * 64,
        f"Total Tasks:             {summary['total_tasks']}",
        f"Success Rate:            {summary['success_rate'] * 100:.2f}% ({summary['successful_tasks']}/{summary['total_tasks']})",
        f"Failed Tasks:            {summary['failed_tasks']} (Errors: {summary['error_tasks']})",
        f"Total Steps:             {summary['total_steps']} (Avg / Task: {summary['avg_steps_per_task']:.2f})",
        f"S2 Escalations:          {summary['total_s2_escalations']}",
        f"S2 Escalation Rate:      {summary['s2_escalation_rate'] * 100:.2f}%",
        f"Total Cost:              ${summary['total_cost_usd']:.6f}",
        f"p50 Latency:             {summary['p50_latency_ms']} ms ({summary['p50_latency_seconds']} s)",
        f"Models Used:             {', '.join(summary['models_used']) if summary['models_used'] else 'None'}",
        "=" * 64,
    ]
    return "\n".join(lines)


def run_eval(
    tasks_path: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    config: ReflexConfig | None = None,
    default_max_steps: int = 25,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute complete evaluation suite, save JSONL results and return summary."""
    tasks = load_tasks(tasks_path)
    if verbose:
        print(f"[eval] Loaded {len(tasks)} tasks from {tasks_path}")

    task_results: list[dict[str, Any]] = []
    output_file = Path(output_path) if output_path else None
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear or create file
        output_file.write_text("", encoding="utf-8")

    for idx, task in enumerate(tasks, start=1):
        if verbose:
            print(f"[eval] [{idx}/{len(tasks)}] Starting task {task['id']} ({task['url']})...")

        res = run_task(task, config=config, default_max_steps=default_max_steps)
        task_results.append(res)

        if output_file:
            with output_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        if verbose:
            status_symbol = "✓" if res["success"] else "✗"
            print(
                f"[eval] [{idx}/{len(tasks)}] {status_symbol} {res['id']}: "
                f"status={res['status']}, steps={res['steps']}, "
                f"score={res['score']}, cost=${res['cost_usd']:.6f}"
            )
            if res.get("last_error"):
                print(f"       last_error: {res['last_error']}")

    summary = compute_summary(task_results)

    if summary_path:
        sum_path = Path(summary_path)
        sum_path.parent.mkdir(parents=True, exist_ok=True)
        sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("\n" + format_summary_report(summary))

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
        r_cdp.ensure_chromium_cdp = lambda p=port: None
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
    parser.add_argument("--cdp-port", type=int, default=9444, help="CDP port for isolated Chromium instance (default: 9444; 9333 is forbidden)")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed mode (default: headless)")

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
        with managed_isolated_browser(port=args.cdp_port, headless=not args.headed):
            _, summary = run_eval(
                tasks_path=tasks_file,
                output_path=args.output,
                summary_path=args.summary,
                config=cfg,
                default_max_steps=args.max_steps,
                verbose=not args.quiet,
            )
            return 0 if summary["failed_tasks"] == 0 else 1
    except Exception as exc:
        print(f"[eval] Fatal evaluation runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
