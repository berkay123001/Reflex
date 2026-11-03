"""Run Reflex on Online-Mind2Web tasks and save results in the benchmark's submission layout.

Each task gets `<out>/<task_id>/result.json` (answer, action history, status, cost) and
`<out>/<task_id>/trajectory/<step>_full_screenshot.png`. Every task runs in its own isolated Chromium.
Irreversible actions that ask for confirmation are declined: nobody is there to approve them.

    python -m scripts.benchmark.run_online_mind2web pilot40.json runs/pilot40 --worker-id 0 --total-workers 3
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from reflex.agent import ReflexSession
from reflex.browser.isolated import managed_isolated_browser
from reflex.config import ReflexConfig

MAX_STEPS = 40


def _history_entry(step_no: int, h: dict[str, Any]) -> dict[str, Any]:
    target = h.get("target") if isinstance(h.get("target"), dict) else {}
    action = h.get("action") if isinstance(h.get("action"), dict) else {}
    progress = (h.get("verification") or {}).get("effective_progress")
    return {
        "step": step_no,
        "operation": h.get("operation"),
        "target_label": target.get("label") or action.get("target_label"),
        "text": h.get("typed_text"),
        "action_status": "SUCCESS" if progress else ("FAILED" if progress is False else None),
        "thought": h.get("escalation_note") or str(h.get("reasoning") or ""),
        "url": str(h.get("url") or ""),
        "screenshot": f"trajectory/{step_no}_full_screenshot.png",
    }


def run_task(task: dict[str, Any], out: Path, port: int, wall_limit_s: int) -> dict[str, Any]:
    task_dir = out / task["task_id"]
    (task_dir / "trajectory").mkdir(parents=True, exist_ok=True)
    cfg = ReflexConfig(s2_every_step=True, max_steps=MAX_STEPS, max_escalations=1000, screenshots=True, browser_language="en-US")
    cfg.trace_path = str(task_dir / "trace.jsonl")
    started = time.time()
    record: dict[str, Any] = {"task_id": task["task_id"], "task": task["confirmed_task"], "website": task["website"], "level": task.get("level")}
    session = None
    try:
        with managed_isolated_browser(port=port):
            session = ReflexSession(url=task["website"], goal=task["confirmed_task"], config=cfg)
            while time.time() - started < wall_limit_s:
                if session.status == "waiting_confirmation":
                    session.confirm_action(approved=False)
                    continue
                if session.status != "ready" or len(session.history) >= MAX_STEPS:
                    break
                session.step()
                try:
                    shot = session.browser.call("Page.captureScreenshot", format="png").get("data")
                except Exception:  # noqa: BLE001 - a missing screenshot must not end the task
                    shot = None
                if shot:
                    (task_dir / "trajectory" / f"{len(session.history) - 1}_full_screenshot.png").write_bytes(base64.b64decode(shot))
            snap = session.snapshot()
            chat = (snap.get("context") or {}).get("chat_log") or []
            answer = next((m.get("content") for m in reversed(chat) if m.get("role") == "assistant" and m.get("kind", "chat") == "chat"), None)
            status = snap.get("status", "ready")
            if status != "done" and len(session.history) >= MAX_STEPS:
                status = "max_steps"
            record.update(
                final_result_response=answer,
                action_history=[_history_entry(i, h) for i, h in enumerate(snap.get("history", []))],
                status=status,
                steps=len(session.history),
                cost_usd=float((snap.get("metrics") or {}).get("total_cost_usd") or 0.0),
                answer_checks=[{"verdict": c.get("verdict"), "unmet": c.get("unmet")} for c in session.answer_checks],
                answer_check_rejections=session._answer_check_rejections,
            )
    except Exception as exc:  # noqa: BLE001 - one broken site must not stop the batch
        record.update(final_result_response=None, action_history=[], status="error", error=f"{exc}\n{traceback.format_exc()[-800:]}",
                      steps=len(session.history) if session else 0, cost_usd=0.0)
    finally:
        if session is not None:
            session.close()
    record["elapsed_s"] = round(time.time() - started, 1)
    record["domain"] = urlsplit(task["website"]).hostname or ""
    (task_dir / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks")
    parser.add_argument("out")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=10600)
    parser.add_argument("--wall-limit-s", type=int, default=600)
    args = parser.parse_args()
    os.environ.setdefault("REFLEX_BROWSER_LANG", "en-US")
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    out = Path(args.out)
    port = args.port_base + args.worker_id * 20
    for i, task in enumerate(tasks):
        if i % args.total_workers != args.worker_id or (out / task["task_id"] / "result.json").exists():
            continue
        rec = run_task(task, out, port, args.wall_limit_s)
        print(f"[{i + 1}/{len(tasks)}] {task['task_id']} {rec['status']} steps={rec['steps']} ${rec['cost_usd']:.4f}", flush=True)


if __name__ == "__main__":
    main()
