"""Reflex Hybrid Browser Agent Runtime (`ReflexSession`).

Combines System 1 (OpenRouter Jev / Local Turnike-Web 4B) with System 2 Supervisor
(`qwen/qwen3.8-omni-flash`, Meta Muse Spark, etc.), 3-layer verification, calibrated
escalation gates, irreversible action safety guards, and interactive human-in-the-loop chat.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _is_raw_dom(text: str) -> bool:
    """Check if text looks like raw DOM/HTML rather than distilled factual knowledge."""
    s = text.strip()
    if not s:
        return True
    s_low = s.lower()
    if s_low.startswith(("<html", "<!doctype", "<body", "<div", "<table", "<section", "<header", "<nav", "<form")):
        return True
    return bool(
        ("</div>" in s_low and "<div" in s_low)
        or ("</span>" in s_low and "<span" in s_low)
        or ("</p>" in s_low and "<p" in s_low)
    )

from .browser import Browser, NavigationTimeout, StalePage, TargetCoveredError
from .browser.cdp import canonical_url
from .config import ReflexConfig, run_mode_of
from .context import DualTierContext, _url_match_keys
from .gate import GateEvaluation, evaluate_decision
from .memory import SiteMemoryStore
from .package import (
    ActionPackage,
    compile_system1_instruction,
    create_action_package,
    extract_s1_target_index,
    format_package_abort_reason,
    match_step_element,
)
from .planner import Subgoal, _extract_domain
from .safety import check_action_safety, extract_origin
from .supervisor import (
    SupervisorDecision,
    generate_field_text,
    interpret_user_message,
    resolve_escalation,
)
from .system1 import MAX_SUPERVISOR_ELEMENTS, MAX_SYSTEM1_ELEMENTS, action_space, choose_action, ground_target
from .verifier import VerificationResult, verify_step


def repeated_type_text_nodes(page: dict[str, Any], history: list[dict[str, Any]]) -> set[int]:
    """Fields whose last step already typed text that is still in the control.

    Only the latest history step counts. A field whose current value no longer
    contains that text is left available.
    """
    if not history:
        return set()
    last = history[-1]
    if not isinstance(last, dict):
        return set()
    if last.get("kind") != "fill" and str(last.get("operation") or "") != "TYPE_TEXT":
        return set()
    typed = str(last.get("text") or "")
    if not typed:
        return set()
    last_node = last.get("node")
    last_label = str(last.get("action") or "").strip()
    masked: set[int] = set()
    for action in page.get("actions") or []:
        if action.get("kind") != "fill":
            continue
        if typed not in str(action.get("value") or ""):
            continue
        node = action.get("node")
        if type(node) is not int:
            continue
        same_node = node == last_node
        same_label = bool(last_label) and last_label == str(action.get("label") or "").strip()
        if same_node or same_label:
            masked.add(node)
    return masked

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)


def _extract_candidate_url(sup: SupervisorDecision, current_url: str) -> str | None:
    """Extract a target navigation URL from SupervisorDecision fields if different from `current_url`."""
    candidates: list[str] = []
    if sup.url:
        candidates.append(sup.url.strip())
    if (sup.operation or "").upper() in {"GOTO_URL", "NAVIGATE", "OPEN_URL"}:
        for field_val in (sup.text, sup.target):
            if field_val:
                candidates.append(field_val.strip())
    for text_field in (sup.tactical_hint, sup.updated_subgoal, sup.reasoning):
        if text_field:
            candidates.extend(_URL_RE.findall(text_field))
    curr_norm = (current_url or "").strip().rstrip("/").lower()
    for raw in candidates:
        cleaned = raw.rstrip(".,;:!?)']\"")
        if not cleaned.startswith(("http://", "https://")):
            if (
                "." in cleaned
                and " " not in cleaned
                and (sup.operation or "").upper() in {"GOTO_URL", "NAVIGATE", "OPEN_URL"}
            ):
                cleaned = "https://" + cleaned
            else:
                continue
        if cleaned.rstrip("/").lower() != curr_norm:
            return cleaned
    return None


class ReflexSession:
    """Stateful, multi-turn hybrid browser agent session."""

    def __init__(
        self,
        url: str,
        goal: str,
        config: ReflexConfig | None = None,
        constraints: list[str] | None = None,
        allow_arbitrary_choices: bool = False,
        session_id: str | None = None,
    ):
        if not goal or not goal.strip():
            raise ValueError("Supply a non-empty goal")
        self.config = config or ReflexConfig()
        clean_url = url.strip()
        self.url = clean_url
        self.session_id: str = (session_id or "").strip() or uuid.uuid4().hex[:12]
        self.trace_path: str | None = self.config.trace_path
        self.trace_write_errors: int = 0
        self._terminal_trace_logged: bool = False
        self.ctx = DualTierContext(
            primary_goal=goal.strip(),
            active_subgoal=goal.strip(),
            allow_arbitrary_choices=allow_arbitrary_choices,
            constraints=list(constraints or []),
            user_present=self.config.interactive,
        )
        self.ctx.add_user_message(goal.strip(), kind="chat")
        self.site_memory = SiteMemoryStore()

        self.status: str = "ready"  # "ready" | "waiting_user" | "waiting_confirmation" | "done" | "blocked" | "error"
        self.started_at: float = time.perf_counter()
        self.elapsed_ms: int = 0
        self.history: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []
        self.interpret_calls: list[dict[str, Any]] = []

        # Pending state for pause/resume & post-verification escalation
        self.pending_fact_key: str | None = None
        self.last_asked_question: str | None = None
        self.pending_confirmation: dict[str, Any] | None = None
        self.pending_escalation_reason: tuple[str, str] | None = None  # (trigger_type, reason)
        self._noop_escalation_counts: dict[str, int] = {}
        self._done_rejected_url: tuple[str, int] | None = None
        self._ineffective_url: str = ""
        self._ineffective_fp: str = ""
        self.last_error: str | None = None
        self.s1_errors: int = 0
        self._s1_steps_since_s2: int = 0
        self._stuck_counter: int = 0
        self._plan_done_rejections: int = 0
        self.unsettled_count: int = 0

        # Action package execution state (G10 R3 & R4)
        self.active_package: ActionPackage | None = None
        self.packages_started: int = 0
        self.packages_completed: int = 0
        self.package_steps_executed: int = 0
        self.package_aborts: Counter[str] = Counter()
        self.s1_pkg_step1_agreed: int = 0
        self.s1_pkg_step1_total: int = 0
        self.s1_pkg_stepk_agreed: int = 0
        self.s1_pkg_stepk_total: int = 0

        initial_page_stub: dict[str, Any] = {
            "url": self.url,
            "title": "",
            "text": "",
            "full_text": "",
            "headings": [],
        }
        executor = ThreadPoolExecutor(max_workers=1)
        interp_future = executor.submit(
            interpret_user_message,
            user_message=goal.strip(),
            ctx=self.ctx,
            page=initial_page_stub,
            history=self.history,
            pending_fact_key=None,
            supervisor_model=self.config.supervisor_model,
            allow_model_fallback=self.config.allow_model_fallback,
            supervisor_timeout_s=self.config.supervisor_timeout_s,
            reasoning_effort=self.config.supervisor_reasoning_effort,
        )
        try:
            self.browser = Browser(self.url)
            try:
                self.page = self.browser.observe(screenshot=self.config.screenshots)
            except Exception:
                self.browser.close()
                raise

            init_url = str(self.page.get("url", "")) if self.page else ""
            if not init_url and self.url:
                init_url = self.url
            if init_url and init_url not in self.ctx.visited_urls:
                self.ctx.visited_urls.append(init_url)

            try:
                interp = interp_future.result()
                self.interpret_calls.append(interp.model_dump())
                self.ctx.allow_arbitrary_choices = interp.allow_arbitrary_choices
                self.ctx.constraints = list(interp.constraints or [])
                if constraints:
                    for c in constraints:
                        if c not in self.ctx.constraints:
                            self.ctx.constraints.append(c)
                if allow_arbitrary_choices:
                    self.ctx.allow_arbitrary_choices = True

                if interp.plan and interp.plan.needs_plan:
                    self.ctx.plan = interp.plan
                    active_sg = self.ctx.plan.get_active_subgoal()
                    if active_sg:
                        self.ctx.active_subgoal = active_sg.text
                    if self.ctx.plan.clarify:
                        clarify = self.ctx.plan.clarify
                        if self.config.interactive:
                            self.status = "waiting_user"
                            self.pending_fact_key = "clarify_scope"
                            opts_str = f" ({', '.join(clarify.options)})" if clarify.options else ""
                            def_str = f" [Varsayılan: {clarify.default}]" if clarify.default else ""
                            q_msg = f"{clarify.question}{opts_str}{def_str}"
                            self.last_asked_question = q_msg
                            self.ctx.add_agent_message(q_msg, kind="ask_user")
                        else:
                            if clarify.default:
                                self.ctx.user_facts["clarify_scope"] = clarify.default
                                match = re.search(r"\b\d+\b", clarify.default)
                                if match:
                                    try:
                                        self.ctx.plan.budget.sources = max(1, int(match.group(0)))
                                    except (ValueError, TypeError):
                                        pass
                                self.ctx.add_agent_message(
                                    f"Varsayılan kapsam kabul edildi: {clarify.default} ({clarify.question})",
                                    kind="chat",
                                )
                            self.status = "ready"
            except Exception as exc:
                self.status = "error"
                self.last_error = f"Goal constraint extraction failed: {exc}"
        finally:
            executor.shutdown(wait=False)

    def elapsed(self) -> int:
        self.elapsed_ms = round((time.perf_counter() - self.started_at) * 1000)
        return self.elapsed_ms

    @property
    def ineffective_clicks(self) -> list[tuple[str, str, str]]:
        return self.ctx.ineffective_clicks

    def _clear_ineffective_clicks(self) -> None:
        """Clear recorded ineffective click targets when URL or page state changes."""
        self.ctx.ineffective_clicks.clear()
        self._ineffective_url = ""
        self._ineffective_fp = ""

    def _sync_ineffective_clicks(self, current_url: str = "", page: dict[str, Any] | None = None) -> None:
        """Clear ineffective clicks if canonical URL or meaningful page fingerprint changed."""
        if not self.ctx.ineffective_clicks:
            return
        page_dict = page if isinstance(page, dict) else (self.page if isinstance(getattr(self, "page", None), dict) else {})
        raw_url = current_url or str(page_dict.get("url", "") or "")
        cur_norm = canonical_url(raw_url) if raw_url else ""
        cur_fp = str(page_dict.get("fingerprint", "") or "")

        last_url = getattr(self, "_ineffective_url", "")
        last_fp = getattr(self, "_ineffective_fp", "")

        if cur_norm and last_url and cur_norm != last_url:
            self._clear_ineffective_clicks()
            return
        if cur_fp and last_fp and cur_fp != last_fp:
            self._clear_ineffective_clicks()
            return
        if cur_norm and all(canonical_url(u) != cur_norm for u, _, _ in self.ctx.ineffective_clicks):
            self._clear_ineffective_clicks()

    def _record_ineffective_click(self, action: dict[str, Any], current_url: str, page_fp: str = "") -> None:
        """Record `(canonical_url, label, role)` for a click that produced no progress or triggered a loop/oscillation."""
        if action.get("kind") != "click":
            return
        cur_norm = canonical_url(current_url or "")
        lbl = str(action.get("label", "") or "").strip()
        role = str(action.get("role", "") or "button").strip().lower() or "button"
        if not cur_norm or not lbl:
            return
        self._sync_ineffective_clicks(current_url=cur_norm)
        entry = (cur_norm, lbl, role)
        if not any(
            canonical_url(u) == cur_norm and l.strip().lower() == lbl.lower() and r.strip().lower() == role
            for u, l, r in self.ctx.ineffective_clicks
        ):
            self.ctx.ineffective_clicks.append(entry)
        self._ineffective_url = cur_norm
        if page_fp:
            self._ineffective_fp = str(page_fp)
        elif isinstance(getattr(self, "page", None), dict) and self.page.get("fingerprint"):
            self._ineffective_fp = str(self.page.get("fingerprint", ""))

    def _is_ineffective_click(self, action: dict[str, Any], current_url: str) -> bool:
        """Return True if `action` matches a previously ineffective click `(canonical_url, label, role)` on the current page."""
        if action.get("kind") != "click":
            return False
        self._sync_ineffective_clicks(current_url=current_url)
        if not self.ctx.ineffective_clicks:
            return False
        cur_norm = canonical_url(current_url or "")
        lbl = str(action.get("label", "") or "").strip().lower()
        role = str(action.get("role", "") or "button").strip().lower() or "button"
        if not cur_norm or not lbl:
            return False
        for u, rec_lbl, rec_role in self.ctx.ineffective_clicks:
            if (
                canonical_url(u) == cur_norm
                and rec_lbl.strip().lower() == lbl
                and (not rec_role or rec_role.strip().lower() == role)
            ):
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        """Return complete UI/telemetry state for the session."""
        elements, _, _ = action_space(self.page.get("actions", []))
        s1_calls = sum(1 for h in self.history if h.get("actor") == "S1")
        s2_calls = len(self.escalations)
        cost_items: list[Any] = []
        for d in self.decisions:
            d_usage = d.get("usage") or {}
            if "cost" in d_usage and d_usage["cost"] is not None:
                cost_items.append(d_usage["cost"])
        for e in self.escalations:
            if "cost_usd" in e and e["cost_usd"] is not None:
                cost_items.append(e["cost_usd"])
        for t in self.text_calls:
            if "cost_usd" in t and t["cost_usd"] is not None:
                cost_items.append(t["cost_usd"])
        for ic in self.interpret_calls:
            if "cost_usd" in ic and ic["cost_usd"] is not None:
                cost_items.append(ic["cost_usd"])

        has_unknown_cost = any(isinstance(c, str) and c in {"bilinmiyor", "unknown"} for c in cost_items)
        known_cost_sum = round(
            sum(float(c) for c in cost_items if isinstance(c, (int, float))),
            6,
        )
        total_cost: float | str = "bilinmiyor" if has_unknown_cost else known_cost_sum

        usage_records = (
            [(d.get("usage") or {}) for d in self.decisions]
            + [(e.get("usage") or {}) for e in self.escalations]
            + [(t.get("usage") or {}) for t in self.text_calls]
            + [(ic.get("usage") or {}) for ic in self.interpret_calls]
        )
        prompt_tokens = sum(int(u.get("prompt_tokens", 0) or 0) for u in usage_records if isinstance(u, dict))
        completion_tokens = sum(int(u.get("completion_tokens", 0) or 0) for u in usage_records if isinstance(u, dict))
        cached_tokens = sum(
            int(
                (
                    (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                    if isinstance(u.get("prompt_tokens_details"), dict)
                    else None
                )
                or u.get("cached_tokens", 0)
                or 0
            )
            for u in usage_records
            if isinstance(u, dict)
        )
        total_tokens = sum(
            int(u.get("total_tokens") or ((u.get("prompt_tokens", 0) or 0) + (u.get("completion_tokens", 0) or 0)))
            for u in usage_records
            if isinstance(u, dict)
        )
        cache_hit_ratio = round(cached_tokens / prompt_tokens, 4) if prompt_tokens > 0 else 0.0

        latencies = [h["latency_ms"] for h in self.history if isinstance(h.get("latency_ms"), (int, float))]
        median_ms = sorted(latencies)[len(latencies) // 2] if latencies else 0
        step_durations = [
            h.get("step_ms", h.get("latency_ms", 0))
            for h in self.history
            if isinstance(h.get("step_ms", h.get("latency_ms")), (int, float))
        ]
        median_step_ms = sorted(step_durations)[len(step_durations) // 2] if step_durations else 0

        return {
            "session_id": self.session_id,
            "status": self.status,
            "goal": self.ctx.primary_goal,
            "compiled_goal": self.ctx.compile_system1_goal(
                self.site_memory.get_domain_hint(self.page.get("url", "")) if self.config.enable_site_memory else ""
            ),
            "page": self.page,
            "elements": elements,
            "history": self.history,
            "decisions": self.decisions,
            "escalations": self.escalations,
            "text_calls": self.text_calls,
            "interpret_calls": self.interpret_calls,
            "context": self.ctx.snapshot_dict(),
            "config": self.config.model_dump(),
            "pending_confirmation": (
                {
                    "summary": self.pending_confirmation["summary"],
                    "reason": self.pending_confirmation["reason"],
                    "action_label": self.pending_confirmation["action"].get("label"),
                }
                if self.pending_confirmation
                else None
            ),
            "pending_fact_key": self.pending_fact_key,
            "asked_question": self.last_asked_question,
            "last_error": self.last_error,
            "metrics": {
                "elapsed_ms": self.elapsed(),
                "steps": len(self.history),
                "s1_steps": s1_calls,
                "s2_escalations": s2_calls,
                "s2_calls": s2_calls,
                "packages_started": self.packages_started,
                "packages_completed": self.packages_completed,
                "package_steps_executed": self.package_steps_executed,
                "package_aborts": dict(self.package_aborts),
                "s1_package_agreement": {
                    "step_1": (
                        round(self.s1_pkg_step1_agreed / self.s1_pkg_step1_total, 4)
                        if self.s1_pkg_step1_total > 0
                        else 0.0
                    ),
                    "step_k": (
                        round(self.s1_pkg_stepk_agreed / self.s1_pkg_stepk_total, 4)
                        if self.s1_pkg_stepk_total > 0
                        else 0.0
                    ),
                },
                "s1_steps_since_s2": self._s1_steps_since_s2,
                "stuck_counter": self._stuck_counter,
                "s1_errors": self.s1_errors,
                "text_helper_calls": len(self.text_calls),
                "trace_write_errors": self.trace_write_errors,
                "visit_queue_rejected_count": self.ctx.visit_queue_rejected_count,
                "median_decision_ms": median_ms,
                "median_step_ms": median_step_ms,
                "total_cost_usd": total_cost,
                "known_cost_usd": known_cost_sum,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "total_tokens": total_tokens,
                    "cache_hit_ratio": cache_hit_ratio,
                },
            },
        }

    def _observe_settled(self, deadline_s: float | None = None) -> dict[str, Any] | None:
        """Observe page with retries until DOM settles or deadline expires.

        Catches StalePage ("Document is navigating", etc.).
        If deadline expires, does not raise; records 'page_unsettled' note,
        sets status='ready', and returns None so the step finishes cleanly.
        """
        effective_deadline = (
            self.config.observe_deadline_s if deadline_s is None else float(deadline_s)
        )
        start = time.monotonic()
        screenshots = bool(getattr(getattr(self, "config", None), "screenshots", False))
        while True:
            try:
                return self.browser.observe(screenshot=screenshots)
            except StalePage:
                elapsed = time.monotonic() - start
                if elapsed >= effective_deadline:
                    break
                time.sleep(0.06)

        # Deadline reached without settling
        if hasattr(self, "ctx") and hasattr(self.ctx, "add_agent_message"):
            self.ctx.add_agent_message("page_unsettled", kind="escalation_note")
        self.status = "ready"
        return None

    def step(self) -> dict[str, Any]:
        """Execute a single hybrid step (Observe -> S1 or S2 Escalate -> Safety Guard -> Act -> Verify)."""
        if self.status in {"waiting_user", "waiting_confirmation", "done", "blocked", "error"}:
            return self.snapshot()

        if len(self.history) >= self.config.max_steps:
            self.status = "blocked"
            self.last_error = f"Reached max steps limit ({self.config.max_steps})"
            self.ctx.add_agent_message(self.last_error, kind="escalation_note")
            self._log_trace_step(
                actor="system",
                action={"id": "max_steps", "label": "Max steps limit reached", "kind": "blocked"},
                operation="BLOCKED",
                target=None,
                typed_text=None,
                verification={"effective_progress": False, "flags": {}, "reason": "Max steps limit reached"},
                task_status="blocked",
                page=self.page,
                step_no=len(self.history),
            )
            return self.snapshot()

        step_t0 = time.perf_counter()

        # Action package execution cycle (G10 R3)
        if self.active_package is not None and not self.active_package.is_finished:
            return self._execute_package_step(step_t0)

        pre_obs_ms = 0
        try:
            needs_obs = (not self.page) or not self.browser.fresh(self.page)
        except StalePage:
            needs_obs = True

        prior_pending_escalation = self.pending_escalation_reason

        if needs_obs:
            obs_t0 = time.perf_counter()
            settled_page = self._observe_settled()
            pre_obs_ms += round((time.perf_counter() - obs_t0) * 1000)
            if settled_page is None:
                self.unsettled_count += 1
                wall_step_ms = round((time.perf_counter() - step_t0) * 1000)
                step_record = {
                    "step": len(self.history) + 1,
                    "actor": "system",
                    "action": "observe",
                    "kind": "observe",
                    "node": None,
                    "choice": "",
                    "operation": "OBSERVE",
                    "target": None,
                    "rect": None,
                    "href": None,
                    "confidence": 0.0,
                    "target_confidence": None,
                    "joint_confidence": 0.0,
                    "target_margin": None,
                    "latency_ms": pre_obs_ms,
                    "s1_ms": 0,
                    "s2_ms": 0,
                    "act_ms": 0,
                    "observe_ms": pre_obs_ms,
                    "step_ms": wall_step_ms,
                    "model": "",
                    "text": "",
                    "text_helper": None,
                    "text_latency_ms": 0,
                    "page_changed": False,
                    "url_changed": False,
                    "element_state_changed": False,
                    "effective_progress": False,
                    "verification": {
                        "effective_progress": False,
                        "page_changed": False,
                        "url_changed": False,
                        "element_state_changed": False,
                        "loop_detected": False,
                        "oscillation_detected": False,
                        "reason": "page_unsettled",
                        "note": "page_unsettled",
                        "status": "bilinmiyor",
                    },
                    "note": "page_unsettled",
                    "escalation_note": "page_unsettled",
                    "before_url": str(self.page.get("url", "")) if isinstance(self.page, dict) else "",
                    "url": str(self.page.get("url", "")) if isinstance(self.page, dict) else "",
                    "after_fingerprint": "",
                    "usage": {},
                    "elapsed_ms": self.elapsed(),
                }
                self.history.append(step_record)
                self._log_trace_step(
                    actor="system",
                    action={"id": "observe", "label": "observe", "kind": "observe"},
                    operation="OBSERVE",
                    target=None,
                    typed_text=None,
                    goto_url=None,
                    verification={"effective_progress": False, "flags": {}, "reason": "page_unsettled"},
                    task_status="blocked" if len(self.history) >= self.config.max_steps else None,
                    page=self.page,
                    step_no=len(self.history),
                )
                if len(self.history) >= self.config.max_steps:
                    self.status = "blocked"
                    self.last_error = f"Reached max steps limit ({self.config.max_steps})"
                    self.ctx.add_agent_message(self.last_error, kind="escalation_note")
                    return self.snapshot()
                if prior_pending_escalation is not None and prior_pending_escalation[0] == "page_unsettled":
                    trig, reason = prior_pending_escalation
                    self.pending_escalation_reason = None
                    self.unsettled_count = 0
                    return self._handle_escalation(
                        trigger_type=trig,
                        reason=reason,
                        s1_decision=None,
                        step_t0=step_t0,
                        s1_ms=0,
                        pre_obs_ms=pre_obs_ms,
                    )
                if self.unsettled_count >= 3:
                    self.pending_escalation_reason = (
                        "page_unsettled",
                        f"Page remained unsettled for {self.unsettled_count} consecutive steps",
                    )
                return self.snapshot()
            self.unsettled_count = 0
            if self.pending_escalation_reason is not None and self.pending_escalation_reason[0] == "page_unsettled":
                self.pending_escalation_reason = None
            self.page = settled_page

        self._sync_ineffective_clicks(current_url=str(self.page.get("url", "")), page=self.page)

        # 1. Check if post-step verification from the previous action scheduled an escalation
        if self.pending_escalation_reason is not None:
            trig, reason = self.pending_escalation_reason
            self.pending_escalation_reason = None
            return self._handle_escalation(
                trigger_type=trig,
                reason=reason,
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=0,
                pre_obs_ms=pre_obs_ms,
            )

        # 1b. Check consecutive S1 steps milestone trigger (only when planned task)
        if self.ctx.plan and self.ctx.plan.needs_plan and self._s1_steps_since_s2 >= 12:
            return self._handle_escalation(
                trigger_type="progress_milestone",
                reason=f"12 consecutive S1 steps without S2 ({self._s1_steps_since_s2} steps)",
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=0,
                pre_obs_ms=pre_obs_ms,
            )

        # Evaluation baseline: System 2 decides every step and System 1 is never called, to measure
        # what the hybrid saves. Bypasses the gate entirely (threshold tweaks cannot: scroll/wait and
        # arbitrary-choice clicks have fixed gate rules).
        # In action package mode, when no package is active, S2 plans the next package/step (S1 does not act unguided).
        if self.config.s2_every_step or self.config.s2_action_packages:
            return self._handle_escalation(
                trigger_type="s2_every_step",
                reason=(
                    "Action package mode: System 2 plans next action package"
                    if self.config.s2_action_packages
                    else "Evaluation baseline: System 2 decides every step"
                ),
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=0,
                pre_obs_ms=pre_obs_ms,
            )

        # 2. Query System 1 (OpenRouter Jev or Local Turnike)
        cur_url_str = str(self.page.get("url", ""))
        cur_norm_url = cur_url_str.split("#")[0].rstrip("/")
        site_hint = (
            self.site_memory.get_domain_hint(cur_url_str)
            if self.config.enable_site_memory
            else ""
        )
        compiled_goal = self.ctx.compile_system1_goal(site_hint)
        suppress_done = (
            self._done_rejected_url is not None
            and self._done_rejected_url[0] == cur_norm_url
            and (len(self.history) - self._done_rejected_url[1]) <= 8
        )

        s1_t0 = time.perf_counter()
        try:
            s1_decision = choose_action(
                self.page,
                compiled_goal,
                self.history,
                self.config,
                suppress_done=suppress_done,
            )
            repeat_nodes = repeated_type_text_nodes(self.page, self.history)
            if repeat_nodes and s1_decision.get("operation") == "TYPE_TEXT":
                chosen = next(
                    (item for item in self.page.get("actions", []) if item.get("id") == s1_decision.get("choice")),
                    None,
                )
                if chosen is not None and chosen.get("node") in repeat_nodes:
                    s1_decision = choose_action(
                        self.page,
                        compiled_goal,
                        self.history,
                        self.config,
                        suppress_done=suppress_done,
                        mask_fill_nodes=repeat_nodes,
                    )
        except ValueError as exc:
            s1_err_ms = round((time.perf_counter() - s1_t0) * 1000)
            self.s1_errors += 1
            return self._handle_escalation(
                trigger_type="s1_error",
                reason=f"System 1 error: {exc}",
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=s1_err_ms,
                pre_obs_ms=pre_obs_ms,
            )
        except RuntimeError as exc:
            if self.config.decision_backend == "local_turnike":
                self.status = "error"
                self.last_error = str(exc)
                return self.snapshot()
            s1_err_ms = round((time.perf_counter() - s1_t0) * 1000)
            self.s1_errors += 1
            return self._handle_escalation(
                trigger_type="s1_error",
                reason=f"System 1 error: {exc}",
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=s1_err_ms,
                pre_obs_ms=pre_obs_ms,
            )
        except Exception as exc:
            self.status = "error"
            self.last_error = str(exc)
            return self.snapshot()

        s1_ms = int(s1_decision.get("latency_ms", round((time.perf_counter() - s1_t0) * 1000)) or 0)
        selected_id = s1_decision.get("choice")
        selected_action = next((a for a in self.page.get("actions", []) if a.get("id") == selected_id), None)

        gate_eval = evaluate_decision(
            s1_decision,
            self.config.thresholds,
            step_count=len(self.history),
            verify_done_with_supervisor=self.config.verify_done_with_supervisor,
            allow_arbitrary_choices=self.ctx.allow_arbitrary_choices,
            constraints=self.ctx.constraints,
            action=selected_action,
            page_actions=self.page.get("actions") or [],
            strict=self.config.s1_strict,
        )

        self.decisions.append(
            {
                "step": len(self.history) + 1,
                "actor": "S1",
                "operation": s1_decision["operation"],
                "target": s1_decision["target"],
                "choice": s1_decision["choice"],
                "confidence": s1_decision["confidence"],
                "target_confidence": s1_decision["target_confidence"],
                "joint_confidence": gate_eval.joint_confidence,
                "target_margin": gate_eval.target_margin,
                "gate": gate_eval.model_dump(),
                "latency_ms": s1_decision["latency_ms"],
                "model": s1_decision["model"],
                "usage": s1_decision.get("usage", {}),
                "elapsed_ms": self.elapsed(),
            }
        )

        # All subgoals completed and System 1 says DONE: System 2 verifies on the page and writes the
        # final answer from the ledger; the ledger check result is passed along as evidence.
        if (
            s1_decision.get("operation") == "DONE"
            and self.ctx.plan
            and self.ctx.plan.needs_plan
            and (
                not self.ctx.active_subgoal
                or self.ctx.plan.get_active_subgoal() is None
                or all(sg.status == "completed" for sg in self.ctx.plan.subgoals)
            )
        ):
            _, ledger_reason, _ = self.ctx.plan.verify_done()
            return self._handle_escalation(
                trigger_type="verify_done",
                reason=(
                    f"System 1 signaled DONE and all subgoals are completed. Ledger check: {ledger_reason}. "
                    "Verify on the page; if complete, write the final answer from the ledger results in summary_for_user."
                ),
                s1_decision=s1_decision,
                gate_eval=gate_eval,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        # 3. Escalation Gate Check
        if gate_eval.should_escalate:
            trig_type = gate_eval.trigger_type
            esc_reason = gate_eval.reason
            if self.ctx.plan and self.ctx.plan.needs_plan and trig_type == "verify_done":
                trig_type = "progress_milestone"
                esc_reason = "System 1 signaled subgoal completion (DONE verification)"
            return self._handle_escalation(
                trigger_type=trig_type,
                reason=esc_reason,
                s1_decision=s1_decision,
                gate_eval=gate_eval,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        # 4. High-Confidence DONE from System 1
        if s1_decision["operation"] == "DONE":
            if self.ctx.plan and self.ctx.plan.needs_plan:
                return self._handle_escalation(
                    trigger_type="progress_milestone",
                    reason="System 1 signaled subgoal completion (DONE operation)",
                    s1_decision=s1_decision,
                    gate_eval=gate_eval,
                    step_t0=step_t0,
                    s1_ms=s1_ms,
                    pre_obs_ms=pre_obs_ms,
                )
            self.status = "done"
            self.ctx.add_agent_message(
                "Goal completed by System 1 (high confidence verification passed).",
                kind="chat",
            )
            self._log_trace_step(
                actor="S1",
                action={"id": "done", "label": "DONE", "kind": "done"},
                operation="DONE",
                target=None,
                typed_text=None,
                verification={"effective_progress": True, "flags": {}, "reason": "System 1 high confidence DONE"},
                task_status="done",
                page=self.page,
                step_no=len(self.history) + 1,
            )
            return self.snapshot()

        # 5. Execute System 1 Selected Action
        selected_id = s1_decision["choice"]
        action = next((a for a in self.page.get("actions", []) if a["id"] == selected_id), None)
        if action is None:
            return self._handle_escalation(
                trigger_type="missing_action",
                reason=f"Selected action ID {selected_id} not found on page",
                s1_decision=s1_decision,
                gate_eval=gate_eval,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        if s1_decision["operation"] == "CLICK" and self._is_ineffective_click(action, self.page.get("url", "")):
            lbl = str(action.get("label", "")).strip()
            role = str(action.get("role", "button")).strip().lower() or "button"
            return self._handle_escalation(
                trigger_type="no_effect_repeat",
                reason=f"Target '{lbl}' ({role}) had no effect on this page — bu buton işe yaramadı",
                s1_decision=s1_decision,
                gate_eval=gate_eval,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        if s1_decision["operation"] == "CLICK" and self._is_rejected_target(action, self.page.get("url", "")):
            scroll_act = next((a for a in self.page.get("actions", []) if a.get("id") == "scroll_down"), None)
            if scroll_act is not None:
                action = scroll_act
                s1_decision["operation"] = "SCROLL_DOWN"
                s1_decision["target"] = None
            else:
                return self._handle_escalation(
                    trigger_type="revisit_blocked",
                    reason=f"Target '{action.get('label', '')}' was already visited and rejected",
                    s1_decision=s1_decision,
                    gate_eval=gate_eval,
                    step_t0=step_t0,
                    s1_ms=s1_ms,
                    pre_obs_ms=pre_obs_ms,
                )

        return self._execute_and_verify(
            action=action,
            actor="S1",
            operation=s1_decision["operation"],
            target=s1_decision["target"],
            confidence=s1_decision["confidence"],
            target_confidence=s1_decision["target_confidence"],
            joint_confidence=gate_eval.joint_confidence,
            target_margin=gate_eval.target_margin,
            latency_ms=s1_decision["latency_ms"],
            model_name=s1_decision["model"],
            usage=s1_decision.get("usage", {}),
            preset_text=None,
            escalation_note=None,
            step_t0=step_t0,
            s1_ms=s1_ms,
            s2_ms=0,
            pre_obs_ms=pre_obs_ms,
        )

    def _is_subgoal_completion_signaled(
        self,
        active_sg: Subgoal | None,
        verification: VerificationResult | None,
        page: dict[str, Any],
    ) -> bool:
        """Check whether active subgoal completion criteria appear met from verifier or page state."""
        if not active_sg:
            return False
        if verification is not None and (
            getattr(verification, "subgoal_done", False)
            or (isinstance(verification.details, dict) and verification.details.get("subgoal_done"))
        ):
            return True
        if active_sg.done_when:
            dw = active_sg.done_when.strip().lower()
            if dw:
                page_text = str(page.get("text") or "").lower()
                page_full = str(page.get("full_text") or "").lower()
                headings = " ".join(str(h) for h in page.get("headings") or []).lower()
                if dw in page_text or dw in page_full or dw in headings:
                    return True
        return False

    def _is_rejected_target(self, action: dict[str, Any], current_url: str) -> bool:
        """Return True if clicking `action` would revisit a navigation link/URL that was already visited and abandoned,
        or re-click an already-selected radio button.
        """
        if action.get("kind") != "click":
            return False
        role = str(action.get("role", "")).lower()
        lbl = str(action.get("label", "")).strip().lower()
        href = str(action.get("href", "")).strip().rstrip("/").lower()

        # 1. Never re-click a radio option that is already checked="true"
        if role in {"radio", "menuitemradio"} and str(action.get("checked", "")).lower() == "true":
            return True

        # 3. Never block unselected in-page form controls (radio, checkbox, option, button without href)
        if role in {"radio", "checkbox", "switch", "option", "menuitemradio", "tab", "combobox", "textbox", "searchbox"}:
            return False
        if not href and role != "link":
            return False
        href_norm = canonical_url(href)
        for rej in self.ctx.rejected_links:
            rej_l = rej.strip().lower()
            if " (" in rej_l and rej_l.endswith(")"):
                rej_lbl, rej_href = rej_l.rsplit(" (", 1)
                rej_href_norm = canonical_url(rej_href[:-1].strip())
            else:
                rej_lbl, rej_href_norm = rej_l, ""
            if href_norm and len(href_norm) >= 8 and rej_href_norm and href_norm == rej_href_norm:
                return True
            if lbl and len(lbl) >= 4 and lbl == rej_lbl and (not href_norm or not rej_href_norm or href_norm == rej_href_norm):
                return True
        # Also check if we already clicked this exact link on this exact URL, navigated to a DIFFERENT URL, and later returned
        cur_norm = canonical_url(current_url or "")
        clicked_idx = -1
        for idx, h in enumerate(self.history):
            h_before = canonical_url(str(h.get("before_url") or ""))
            if (
                h.get("kind") == "click"
                and h.get("url_changed")
                and (not h_before or h_before == cur_norm)
                and str(h.get("action", "")).strip().lower() == lbl
            ):
                clicked_idx = idx
        if clicked_idx >= 0 and clicked_idx < len(self.history) - 1:
            for later in self.history[clicked_idx + 1 :]:
                if later.get("kind") == "go_back" or canonical_url(str(later.get("url", ""))) == cur_norm:
                    return True
        return False

    def _record_rejected_from_history(self, reason_url: str) -> None:
        """Mark the most recently clicked link before `reason_url` as rejected so it won't be clicked again."""
        for h in reversed(self.history):
            if h.get("kind") == "click":
                raw_href = str(h.get("href", "") or "").strip()
                if not raw_href and not h.get("url_changed"):
                    continue
                lbl = str(h.get("action", "")).strip()
                href = raw_href or (str(h.get("url", "") or reason_url).strip() if h.get("url_changed") else "")
                entry = f"{lbl} ({href})" if href else lbl
                if entry and entry not in self.ctx.rejected_links:
                    self.ctx.rejected_links.append(entry)
                break

    def _verify_and_complete_task_from_plan(self, summary: str | None = None) -> dict[str, Any]:
        """Finish a planned task after System 2 returned COMPLETE_TASK.

        The ledger check is advisory: it sends System 2 back once, but a repeated COMPLETE_TASK is final,
        so an over-strict budget (e.g. sources=2 on a single-site task) cannot lock the task forever.
        System 2's own summary is the answer; the ledger template is only a fallback.
        """
        if not self.ctx.plan or not self.ctx.plan.needs_plan:
            self.status = "done"
            self.ctx.add_agent_message(summary or "Task completed.", kind="chat")
            self._log_trace_step(
                actor="S2",
                action={"id": "complete_task", "label": "COMPLETE_TASK", "kind": "complete_task"},
                operation="DONE",
                target=None,
                typed_text=None,
                verification={"effective_progress": True, "flags": {}, "reason": "Supervisor COMPLETE_TASK"},
                task_status="done",
                page=self.page,
                step_no=len(self.history) + 1,
            )
            return self.snapshot()

        # System 2 judged the task complete, so the subgoal it was working on is complete too. Planners often
        # end with a reporting subgoal ("compare and report") that no page visit can satisfy; its result is the
        # summary itself.
        active_sg = self.ctx.plan.get_active_subgoal()
        if summary and active_sg is not None:
            if not active_sg.results:
                self.ctx.plan.add_result(
                    fact=summary[:500],
                    source_url=str(self.page.get("url", "")),
                    step=len(self.history),
                    subgoal_id=active_sg.id,
                )
            active_sg.status = "completed"

        is_done, reason, missing_sg = self.ctx.plan.verify_done()

        if not is_done and self._plan_done_rejections < 1:
            self._plan_done_rejections += 1
            self.ctx.tactical_hint = f"Ledger check before finishing: {reason}"
            # Reject DONE: do NOT set self.status = "done", remain in ready
            self.status = "ready"
            if missing_sg is not None:
                for s in self.ctx.plan.subgoals:
                    if s.id != missing_sg.id and s.status == "active":
                        s.status = "pending"
                missing_sg.status = "active"
                self.ctx.active_subgoal = missing_sg.text
            page_dict = self.page if isinstance(self.page, dict) else {}
            cur_norm = str(page_dict.get("url", "")).split("#")[0].rstrip("/")
            self._done_rejected_url = (cur_norm, len(self.history))
            self.ctx.add_agent_message(
                f"DONE rejected by plan verification: {reason}. Reactivated subgoal: {self.ctx.active_subgoal}",
                kind="escalation_note",
            )
            return self.snapshot()

        if not is_done:
            self.ctx.add_agent_message(
                f"Ledger check still open ({reason}); System 2 confirmed completion again, accepting.",
                kind="escalation_note",
            )
        self.status = "done"
        final_answer = summary or self.ctx.plan.synthesize_final_answer(self.ctx.primary_goal)
        self.ctx.add_agent_message(final_answer, kind="chat")
        self._log_trace_step(
            actor="S2",
            action={"id": "complete_task", "label": "COMPLETE_TASK", "kind": "complete_task"},
            operation="DONE",
            target=None,
            typed_text=None,
            verification={"effective_progress": True, "flags": {}, "reason": "Plan completed and verified"},
            task_status="done",
            page=self.page,
            step_no=len(self.history) + 1,
        )
        return self.snapshot()

    def _abort_package(
        self,
        pkg: ActionPackage,
        trigger: str,
        detail: str,
        *,
        step_t0: float,
        pre_obs_ms: int = 0,
    ) -> dict[str, Any]:
        """Abort the active package before its current step runs and hand the page back to System 2."""
        self.package_aborts[trigger] += 1
        pkg.aborted = True
        pkg.abort_trigger = trigger
        pkg.abort_reason = detail
        reason = format_package_abort_reason(
            pkg.current_step_index, pkg.total_steps, trigger, detail, pkg.remaining_instructions()
        )
        self.active_package = None
        return self._handle_escalation(
            trigger_type=trigger, reason=reason, s1_decision=None, step_t0=step_t0, pre_obs_ms=pre_obs_ms
        )

    def _finish_package_step(self, pkg: ActionPackage) -> None:
        """Advance, complete or abort the package after one executed step (step 1 and step k alike)."""
        last_h = self.history[-1] if self.history else {}
        v = last_h.get("verification", {})
        trigger: str | None = None
        detail = ""
        if last_h.get("url_changed"):
            trigger, detail = "url_changed", f"URL changed to {last_h.get('url')}"
        elif not v.get("effective_progress", True):
            trigger, detail = "no_progress", "Validator observed no effective progress"
        elif v.get("loop_detected"):
            trigger, detail = "loop_detected", "Validator detected loop"
        elif v.get("oscillation_detected"):
            trigger, detail = "oscillation_detected", "Validator detected oscillation"
        elif self.pending_escalation_reason is not None:
            trigger, detail = "pending_escalation", f"Pending escalation: {self.pending_escalation_reason}"
        elif self.status != "ready":
            trigger, detail = "status_not_ready", f"Status changed to {self.status}"

        is_last = pkg.current_step_index >= pkg.total_steps
        # A page change after the final step (typically submit/next) is the expected end, not an abort.
        if trigger is None or (is_last and trigger == "url_changed"):
            pkg.current_step_index += 1
            if pkg.current_step_index > pkg.total_steps:
                self.packages_completed += 1
                self.active_package = None
            return

        self.package_aborts[trigger] += 1
        pkg.aborted = True
        pkg.abort_trigger = trigger
        pkg.abort_reason = detail
        if "package" in last_h:
            last_h["package"]["abort_reason"] = detail
        reason = format_package_abort_reason(
            pkg.current_step_index, pkg.total_steps, trigger, detail, pkg.remaining_instructions()[1:]
        )
        self.pending_escalation_reason = (trigger, reason)
        self.active_package = None

    def _execute_package_step(self, step_t0: float) -> dict[str, Any]:
        """Execute step k (k >= 2) of an active action package."""
        pkg = self.active_package
        if pkg is None or pkg.is_finished:
            self.active_package = None
            return self.snapshot()

        step = pkg.current_step
        if step is None:
            self.active_package = None
            return self.snapshot()

        # 1. Observe settled DOM
        pre_obs_ms = 0
        obs_t0 = time.perf_counter()
        settled_page = self._observe_settled()
        pre_obs_ms += round((time.perf_counter() - obs_t0) * 1000)

        if settled_page is None:
            self.package_aborts["page_unsettled"] += 1
            pkg.aborted = True
            pkg.abort_trigger = "page_unsettled"
            pkg.abort_reason = "Page unsettled"
            reason = format_package_abort_reason(
                pkg.current_step_index,
                pkg.total_steps,
                "page_unsettled",
                "Page unsettled",
                pkg.remaining_instructions(),
            )
            self.active_package = None
            return self._handle_escalation(
                trigger_type="page_unsettled",
                reason=reason,
                s1_decision=None,
                step_t0=step_t0,
                pre_obs_ms=pre_obs_ms,
            )

        self.page = settled_page
        self._sync_ineffective_clicks(current_url=str(self.page.get("url", "")), page=self.page)

        # 2. Pre-execution abort conditions
        current_url = str(self.page.get("url", ""))
        if pkg.initial_url and current_url != pkg.initial_url:
            self.package_aborts["url_changed"] += 1
            pkg.aborted = True
            pkg.abort_trigger = "url_changed"
            pkg.abort_reason = f"URL changed from {pkg.initial_url} to {current_url}"
            reason = format_package_abort_reason(
                pkg.current_step_index,
                pkg.total_steps,
                "url_changed",
                pkg.abort_reason,
                pkg.remaining_instructions(),
            )
            self.active_package = None
            return self._handle_escalation(
                trigger_type="url_changed",
                reason=reason,
                s1_decision=None,
                step_t0=step_t0,
                pre_obs_ms=pre_obs_ms,
            )

        if self.pending_escalation_reason is not None:
            trig, p_reason = self.pending_escalation_reason
            self.pending_escalation_reason = None
            self.package_aborts["pending_escalation"] += 1
            pkg.aborted = True
            pkg.abort_trigger = "pending_escalation"
            pkg.abort_reason = p_reason
            reason = format_package_abort_reason(
                pkg.current_step_index,
                pkg.total_steps,
                "pending_escalation",
                p_reason,
                pkg.remaining_instructions(),
            )
            self.active_package = None
            return self._handle_escalation(
                trigger_type=trig,
                reason=reason,
                s1_decision=None,
                step_t0=step_t0,
                pre_obs_ms=pre_obs_ms,
            )

        if self.status != "ready":
            self.package_aborts["status_not_ready"] += 1
            pkg.aborted = True
            pkg.abort_trigger = "status_not_ready"
            self.active_package = None
            return self.snapshot()

        # 3. System 1 sees at most MAX_SYSTEM1_ELEMENTS; a target System 2 picked beyond that window
        # cannot be confirmed, so hand the step back instead of counting a false disagreement.
        page_actions = self.page.get("actions", [])
        expected_node = step.expected.node
        if expected_node is not None and any(a.get("node") == expected_node for a in page_actions):
            _, s1_window, _ = action_space(page_actions)
            if not any(a.get("node") == expected_node for a in s1_window.get(step.operation, {}).values()):
                return self._abort_package(
                    pkg,
                    "package_target_out_of_window",
                    f"Expected node {expected_node} is outside System 1's {MAX_SYSTEM1_ELEMENTS}-element window",
                    step_t0=step_t0,
                    pre_obs_ms=pre_obs_ms,
                )

        # 4. Query System 1 (target question only) with error shielding
        compiled_instruction = compile_system1_instruction(step.instruction)
        s1_t0 = time.perf_counter()
        try:
            s1_decision = ground_target(
                self.page,
                compiled_instruction,
                step.operation,
                self.history,
                self.config,
            )
            s1_ms = round((time.perf_counter() - s1_t0) * 1000)
        except Exception as exc:
            s1_ms = round((time.perf_counter() - s1_t0) * 1000)
            self.s1_errors += 1
            self.package_aborts["s1_error"] += 1
            pkg.aborted = True
            pkg.abort_trigger = "s1_error"
            pkg.abort_reason = str(exc)
            reason = format_package_abort_reason(
                pkg.current_step_index,
                pkg.total_steps,
                "s1_error",
                str(exc),
                pkg.remaining_instructions(),
            )
            self.active_package = None
            return self._handle_escalation(
                trigger_type="s1_error",
                reason=reason,
                s1_decision=None,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        # 4. Extract S1 target from raw_answers
        s1_target_idx = extract_s1_target_index(
            s1_decision.get("raw_answers", {}), step.operation
        )
        current_targets = s1_decision.get("targets", {})
        if not current_targets:
            _, current_targets, _ = action_space(
                self.page.get("actions", []), max_elements=MAX_SUPERVISOR_ELEMENTS
            )

        # 5. Matching Decision Tree
        matched, abort_trig, chosen_act = match_step_element(
            step.expected,
            s1_target_idx,
            current_targets,
            self.page.get("actions", []),
        )

        self.s1_pkg_stepk_total += 1
        if matched and chosen_act is not None:
            self.s1_pkg_stepk_agreed += 1
        else:
            trig_name = abort_trig or "package_mismatch"
            self.package_aborts[trig_name] += 1
            pkg.aborted = True
            pkg.abort_trigger = trig_name
            detail = (
                f"S1 target '{s1_target_idx}' did not match expected node {step.expected.node}"
                if trig_name == "package_mismatch"
                else f"Expected node {step.expected.node} missing and label/role mismatched"
            )
            pkg.abort_reason = detail
            reason = format_package_abort_reason(
                pkg.current_step_index,
                pkg.total_steps,
                trig_name,
                detail,
                pkg.remaining_instructions(),
            )
            self.active_package = None
            return self._handle_escalation(
                trigger_type=trig_name,
                reason=reason,
                s1_decision=s1_decision,
                step_t0=step_t0,
                s1_ms=s1_ms,
                pre_obs_ms=pre_obs_ms,
            )

        # 6. TYPE_TEXT text comes from S2 package
        preset_text = step.text if step.operation == "TYPE_TEXT" else None

        # 7. Execute via _execute_and_verify
        pkg_history_meta = {
            "id": pkg.id,
            "step": pkg.current_step_index,
            "of": pkg.total_steps,
            "agreed": True,
            "abort_reason": None,
        }
        self.package_steps_executed += 1
        self._execute_and_verify(
            action=chosen_act,
            actor="S1",
            operation=step.operation,
            target=s1_target_idx,
            confidence=float(s1_decision.get("confidence") or 1.0),
            target_confidence=s1_decision.get("target_confidence"),
            joint_confidence=float(s1_decision.get("joint_confidence") or 1.0),
            target_margin=float(s1_decision.get("target_margin") or 1.0),
            latency_ms=s1_decision.get("latency_ms", 0),
            model_name=str(s1_decision.get("model", "")),
            usage=s1_decision.get("usage", {}),
            preset_text=preset_text,
            escalation_note=f"[package step {pkg.current_step_index}/{pkg.total_steps}] {step.instruction}",
            step_t0=step_t0,
            s1_ms=s1_ms,
            s2_ms=0,
            pre_obs_ms=pre_obs_ms,
            package_meta=pkg_history_meta,
        )

        self._finish_package_step(pkg)
        return self.snapshot()  # reflects the package bookkeeping above

    def _handle_escalation(
        self,
        *,
        trigger_type: str,
        reason: str,
        s1_decision: dict[str, Any] | None,
        gate_eval: GateEvaluation | None = None,
        step_t0: float | None = None,
        s1_ms: int = 0,
        pre_obs_ms: int = 0,
    ) -> dict[str, Any]:
        """Run System 2 Supervisor (`qwen/qwen3.8-omni-flash`) to resolve an escalation."""
        self.active_package = None
        if trigger_type == "page_unsettled":
            self.unsettled_count = 0
        if step_t0 is None:
            step_t0 = time.perf_counter()
        if len(self.escalations) >= self.config.escalation_limit:
            self.status = "blocked"
            self.last_error = f"Reached max supervisor escalations ({self.config.escalation_limit}): {reason}"
            self.ctx.add_agent_message(self.last_error, kind="escalation_note")
            return self.snapshot()

        elements, targets, controls = action_space(
            self.page.get("actions", []),
            max_elements=MAX_SUPERVISOR_ELEMENTS,
        )
        existing_href_keys: set[str] = set()
        for el in elements:
            existing_href_keys.update(_url_match_keys(str(el.get("href") or "")))
        for pl in self.page.get("page_links") or []:
            if len(elements) >= MAX_SUPERVISOR_ELEMENTS or not isinstance(pl, dict):
                break
            pl_href = str(pl.get("href") or "").strip()
            pl_keys = _url_match_keys(pl_href)
            if not pl_keys or (pl_keys & existing_href_keys):
                continue
            existing_href_keys.update(pl_keys)
            idx = str(len(elements) + 1)
            pl_lbl = str(pl.get("label") or pl_href).strip()
            elements.append(
                {
                    "index": idx,
                    "role": "link",
                    "label": pl_lbl,
                    "href": pl_href,
                    "operations": ["GOTO_URL"],
                }
            )
            targets.setdefault("CLICK", {})[idx] = {
                "id": "goto_url",
                "kind": "goto_url",
                "label": f"Navigate to {pl_href}",
                "url": pl_href,
                "href": pl_href,
            }
        s2_t0 = time.perf_counter()
        try:
            sup: SupervisorDecision = resolve_escalation(
                ctx=self.ctx,
                page=self.page,
                elements=elements,
                history=self.history,
                escalation_reason=reason,
                trigger_type=trigger_type,
                s1_decision=s1_decision,
                supervisor_model=self.config.supervisor_model,
                allow_model_fallback=self.config.allow_model_fallback,
                supervisor_timeout_s=self.config.supervisor_timeout_s,
                reasoning_effort=self.config.supervisor_reasoning_effort,
                package_mode=self.config.s2_action_packages,
            )
        except Exception as exc:
            self.status = "error"
            self.last_error = str(exc)
            return self.snapshot()

        s2_ms = int(sup.latency_ms or round((time.perf_counter() - s2_t0) * 1000))
        cur_url = str(self.page.get("url", "")).strip()
        step_num = len(self.history) + 1
        self._s1_steps_since_s2 = 0

        if self.ctx.plan and self.ctx.plan.needs_plan:
            # Plan-as-Memory (Anti-Clobbering): Do NOT overwrite with 15-word sup.memory.
            # Instead, accumulate distilled results (0-3 facts) additively to active subgoal.
            facts_to_add: list[tuple[str, str]] = []

            raw_results = (
                getattr(sup, "extracted_results", None)
                or sup.raw.get("new_results")
                or sup.raw.get("extracted_results")
                or sup.raw.get("results")
                or []
            )
            if isinstance(raw_results, (list, tuple)):
                for r in raw_results:
                    if len(facts_to_add) >= 3:
                        break
                    if isinstance(r, dict):
                        f_text = str(r.get("fact") or "").strip()
                        f_url = str(r.get("source_url") or cur_url).strip()
                        if f_text and not _is_raw_dom(f_text):
                            facts_to_add.append((f_text, f_url))
                    elif isinstance(r, str) and r.strip() and not _is_raw_dom(r):
                        facts_to_add.append((r.strip(), cur_url))

            if len(facts_to_add) < 3 and sup.verified_fact_key and sup.verified_fact_value:
                vf_text = f"{sup.verified_fact_key}: {sup.verified_fact_value}".strip()
                if not _is_raw_dom(vf_text) and not any(vf_text == f[0] for f in facts_to_add):
                    facts_to_add.append((vf_text, cur_url))

            new_notes = getattr(sup, "new_notes", None) or sup.raw.get("new_notes")
            if len(facts_to_add) < 3 and isinstance(new_notes, str) and new_notes.strip() and not _is_raw_dom(new_notes):
                clean_notes = new_notes.strip()
                if not any(clean_notes == f[0] for f in facts_to_add):
                    facts_to_add.append((clean_notes, cur_url))

            for f_text, f_url in facts_to_add[:3]:
                self.ctx.plan.add_result(fact=f_text, source_url=f_url, step=step_num)

            raw_cands = (
                getattr(sup, "extracted_candidates", None)
                or sup.raw.get("new_candidates")
                or sup.raw.get("extracted_candidates")
                or sup.raw.get("candidates")
                or []
            )
            if isinstance(raw_cands, dict):
                raw_cands = [raw_cands]
            if isinstance(raw_cands, (list, tuple)):
                for c in raw_cands:
                    if isinstance(c, dict):
                        item_name = c.get("item") or c.get("name")
                        if item_name and str(item_name).strip():
                            c_item = str(item_name).strip()
                            c_url = str(c.get("source_url") or cur_url).strip()
                            c_attrs = dict(c.get("attributes") or {}) if isinstance(c.get("attributes"), dict) else {}
                            self.ctx.plan.add_candidate(item=c_item, source_url=c_url, attributes=c_attrs)

            # Evaluate progress check response
            subgoal_done = getattr(sup, "subgoal_done", False) or bool(sup.raw.get("subgoal_done", False))
            next_subgoal = getattr(sup, "next_subgoal", None) or sup.raw.get("next_subgoal")
            stuck = getattr(sup, "stuck", False) or bool(sup.raw.get("stuck", False))

            if subgoal_done:
                if next_subgoal:
                    for s in self.ctx.plan.subgoals:
                        if s.status == "active":
                            s.status = "completed"
                    target_sg = next(
                        (s for s in self.ctx.plan.subgoals if s.id == next_subgoal or s.text == next_subgoal),
                        None,
                    )
                    if target_sg is not None:
                        target_sg.status = "active"
                        self.ctx.active_subgoal = target_sg.text
                    else:
                        next_sg = self.ctx.plan.advance_subgoal()
                        if next_sg is not None:
                            self.ctx.active_subgoal = next_sg.text
                        else:
                            self.ctx.active_subgoal = ""
                else:
                    next_sg = self.ctx.plan.advance_subgoal()
                    if next_sg is not None:
                        self.ctx.active_subgoal = next_sg.text
                    else:
                        self.ctx.active_subgoal = ""

                if self.ctx.active_subgoal:
                    self.ctx.add_agent_message(
                        f"Subgoal completed. Active subgoal advanced to: {self.ctx.active_subgoal}",
                        kind="escalation_note",
                    )
                else:
                    self.ctx.add_agent_message(
                        "Subgoal completed. All subgoals completed.",
                        kind="escalation_note",
                    )

            if stuck:
                self._stuck_counter += 1
                if self._stuck_counter >= 3:
                    recovery_hint = (
                        sup.tactical_hint
                        or "Progress stuck across multiple checks. Consider alternative search queries or navigation paths."
                    )
                    self.ctx.tactical_hint = recovery_hint
                    self.ctx.add_agent_message(
                        f"Stuck counter reached {self._stuck_counter}: applying recovery hint '{recovery_hint}'",
                        kind="escalation_note",
                    )
            else:
                self._stuck_counter = 0

            self.ctx.working_memory = self.ctx.plan.format_for_s2()
        else:
            if sup.memory and isinstance(sup.memory, str):
                self.ctx.add_working_memory_note(sup.memory)

        raw_queue_urls = (
            getattr(sup, "queue_urls", None)
            or (sup.raw.get("queue_urls") if isinstance(getattr(sup, "raw", None), dict) else None)
            or []
        )
        accepted_queue_urls: list[str] = []
        rejected_queue_delta = 0
        if isinstance(raw_queue_urls, (list, tuple)) and raw_queue_urls:
            rej_before = self.ctx.visit_queue_rejected_count
            accepted_queue_urls = self.ctx.enqueue_urls(raw_queue_urls, elements)
            rejected_queue_delta = self.ctx.visit_queue_rejected_count - rej_before

        esc_record = {
            "step": len(self.history) + 1,
            "trigger_type": trigger_type,
            "reason": reason,
            "action_type": sup.action_type,
            "evaluation_previous_goal": sup.evaluation_previous_goal,
            "memory": sup.memory,
            "next_goal": sup.next_goal,
            "reasoning": sup.reasoning,
            "operation": sup.operation,
            "target": sup.target,
            "text": sup.text,
            "url": sup.url,
            "queue_urls": accepted_queue_urls,
            "rejected_queue_urls": rejected_queue_delta,
            "updated_subgoal": sup.updated_subgoal,
            "tactical_hint": sup.tactical_hint,
            "question_for_user": sup.question_for_user,
            "model": sup.model,
            "latency_ms": sup.latency_ms,
            "cost_usd": sup.cost_usd,
            "usage": sup.usage,
            "retries": sup.retries,
            "retry_reason": sup.retry_reason,
            "finish_reason": sup.finish_reason,
            "elapsed_ms": self.elapsed(),
        }
        self.escalations.append(esc_record)

        if sup.verified_fact_key and sup.verified_fact_value:
            self.ctx.record_dom_fact(
                sup.verified_fact_key,
                sup.verified_fact_value,
                self.page.get("url", ""),
                len(self.history) + 1,
            )

        if self.config.enable_site_memory and sup.site_rule:
            self.site_memory.record_rule(self.page.get("url", ""), sup.site_rule, reason)

        if sup.updated_subgoal:
            self.ctx.active_subgoal = sup.updated_subgoal
            if self.ctx.plan and self.ctx.plan.needs_plan:
                active_sg = self.ctx.plan.get_active_subgoal()
                if active_sg:
                    active_sg.text = sup.updated_subgoal
        if sup.tactical_hint:
            self.ctx.tactical_hint = sup.tactical_hint

        # Case 1: Supervisor verifies the task is complete
        if sup.action_type == "COMPLETE_TASK":
            if self.ctx.plan and self.ctx.plan.needs_plan:
                return self._verify_and_complete_task_from_plan(sup.summary_for_user)
            self.status = "done"
            msg = sup.summary_for_user or sup.reasoning or "Task verified complete by Supervisor."
            self.ctx.add_agent_message(msg, kind="chat")
            self._log_trace_step(
                actor="S2",
                action={"id": "complete_task", "label": "COMPLETE_TASK", "kind": "complete_task"},
                operation="DONE",
                target=None,
                typed_text=None,
                verification={"effective_progress": True, "flags": {}, "reason": "Supervisor COMPLETE_TASK"},
                task_status="done",
                page=self.page,
                step_no=len(self.history) + 1,
            )
            return self.snapshot()

        if trigger_type == "verify_done":
            cur_norm = str(self.page.get("url", "")).split("#")[0].rstrip("/")
            self._done_rejected_url = (cur_norm, len(self.history))

        # Case 2: Supervisor asks the human user for missing info / clarification
        if sup.action_type == "ASK_USER":
            question = (sup.question_for_user or sup.reasoning or "Could you clarify how you want to proceed?").strip()

            if not getattr(self.ctx, "user_present", True):
                unattended = getattr(self.ctx, "unattended_questions", None)
                if unattended is None:
                    unattended = []
                    self.ctx.unattended_questions = unattended
                q_clean = question.strip()
                past_norm_questions = [q.strip().casefold() for q in unattended]
                if q_clean.casefold() not in past_norm_questions and len(unattended) < 2:
                    unattended.append(q_clean)
                    self.pending_fact_key = None
                    self.last_asked_question = None
                    hint = f"User is unavailable: choose a sensible default for: {question}, state it in the final answer"
                    self.ctx.tactical_hint = hint
                    self.pending_escalation_reason = (
                        "ask_user_default",
                        f"Unattended default for: {question}",
                    )
                    self.ctx.add_agent_message(
                        f"Unattended default: {hint}",
                        kind="escalation_note",
                    )
                    self.status = "ready"
                    return self.snapshot()

            self.status = "waiting_user"
            self.pending_fact_key = sup.fact_key or "user_clarification"
            self.last_asked_question = question
            self.ctx.add_agent_message(question, kind="ask_user")
            return self.snapshot()

        current_fp = str(self.page.get("fingerprint", ""))

        # Case 3: Supervisor updates subgoal/hint without forcing a direct browser action
        if sup.action_type == "UPDATE_SUBGOAL":
            # If System 1 was BLOCKED and Supervisor suggested navigating to a new URL in hint/url, execute GOTO_URL!
            nav_url = _extract_candidate_url(sup, self.page.get("url", "")) if trigger_type in {"blocked", "s1_blocked"} else None
            if nav_url is None:
                if trigger_type not in {"progress_milestone", "page_unsettled"}:
                    self._noop_escalation_counts[current_fp] = self._noop_escalation_counts.get(current_fp, 0) + 1
                    if self._noop_escalation_counts[current_fp] >= 2:
                        self.status = "waiting_user"
                        self.pending_fact_key = "user_guidance"
                        ask_msg = f"I am stuck on this page ({sup.reasoning}). Could you clarify or provide a different instruction?"
                        self.last_asked_question = ask_msg
                        self.ctx.add_agent_message(ask_msg, kind="ask_user")
                        return self.snapshot()
                self.ctx.add_agent_message(
                    f"Supervisor updated subgoal: {self.ctx.active_subgoal} ({sup.reasoning})",
                    kind="escalation_note",
                )
                self.status = "ready"
                return self.snapshot()

        # Case 4: Supervisor overrides/forces a specific action (or recovery primitive)
        action_dict = self._resolve_supervisor_action(sup, targets, controls)
        if action_dict is None:
            if trigger_type in {"progress_milestone", "page_unsettled"}:
                self.ctx.add_agent_message(
                    f"Page unsettled: {sup.reasoning}" if trigger_type == "page_unsettled" else f"Progress milestone verified: {sup.reasoning}",
                    kind="escalation_note",
                )
                self.status = "ready"
                return self.snapshot()
            self._noop_escalation_counts[current_fp] = self._noop_escalation_counts.get(current_fp, 0) + 1
            if self._noop_escalation_counts[current_fp] >= 2:
                self.status = "waiting_user"
                self.pending_fact_key = "user_guidance"
                ask_msg = f"Supervisor could not resolve a valid browser action on this page ({sup.reasoning}). How should I proceed?"
                self.last_asked_question = ask_msg
                self.ctx.add_agent_message(ask_msg, kind="ask_user")
                return self.snapshot()
            # If supervisor gave a subgoal update or invalid target once, let System 1 try with updated hint
            self.ctx.add_agent_message(
                f"Supervisor hint applied: {sup.reasoning}",
                kind="escalation_note",
            )
            self.status = "ready"
            return self.snapshot()

        esc_note = (
            f"[{trigger_type}] (queue +{len(accepted_queue_urls)}, rej {rejected_queue_delta}) {sup.reasoning}"
            if isinstance(raw_queue_urls, (list, tuple)) and raw_queue_urls
            else f"[{trigger_type}] {sup.reasoning}"
        )
        # Check if S2 produced an Action Package (G10 R3)
        raw_pkg_actions = getattr(sup, "actions", None)
        if self.config.s2_action_packages and raw_pkg_actions:
            package = create_action_package(raw_pkg_actions, self.page, targets)
            if package is not None and package.total_steps >= 2:
                self.active_package = package
                self.packages_started += 1
                step_1 = package.steps[0]

                # Step 1 is the Supervisor's own action. System 1 is asked the same grounding question
                # only to measure agreement; the probe never blocks the step. A failed probe or a target
                # outside System 1's window is not counted.
                agreed: bool | None = None
                try:
                    s1_resp = ground_target(
                        self.page,
                        compile_system1_instruction(step_1.instruction),
                        step_1.operation,
                        self.history,
                        self.config,
                    )
                    s1_t_idx = extract_s1_target_index(s1_resp.get("raw_answers", {}), step_1.operation)
                    s1_targets = (s1_resp.get("targets") or targets).get(step_1.operation, {})
                    if any(a.get("node") == step_1.expected.node for a in s1_targets.values()):
                        agreed = bool(s1_t_idx) and s1_targets.get(s1_t_idx, {}).get("node") == step_1.expected.node
                except Exception:
                    agreed = None

                if agreed is not None:
                    self.s1_pkg_step1_total += 1
                    if agreed:
                        self.s1_pkg_step1_agreed += 1

                pkg_history_meta = {
                    "id": package.id,
                    "step": 1,
                    "of": package.total_steps,
                    "agreed": agreed,
                    "abort_reason": None,
                }
                self.package_steps_executed += 1
                self._execute_and_verify(
                    action=action_dict,
                    actor="S2",
                    operation=sup.operation or action_dict["kind"].upper(),
                    target=sup.target,
                    confidence=1.0,
                    target_confidence=1.0,
                    joint_confidence=1.0,
                    target_margin=1.0,
                    latency_ms=sup.latency_ms,
                    model_name=sup.model,
                    usage={"cost": 0.0},
                    preset_text=sup.text,
                    escalation_note=esc_note,
                    step_t0=step_t0,
                    s1_ms=s1_ms,
                    s2_ms=s2_ms,
                    pre_obs_ms=pre_obs_ms,
                    package_meta=pkg_history_meta,
                )

                self._finish_package_step(package)
                return self.snapshot()  # reflects the package bookkeeping above

        return self._execute_and_verify(
            action=action_dict,
            actor="S2",
            operation=sup.operation or action_dict["kind"].upper(),
            target=sup.target,
            confidence=1.0,
            target_confidence=1.0,
            joint_confidence=1.0,
            target_margin=1.0,
            latency_ms=sup.latency_ms,
            model_name=sup.model,
            usage={"cost": 0.0},
            preset_text=sup.text,
            escalation_note=esc_note,
            step_t0=step_t0,
            s1_ms=s1_ms,
            s2_ms=s2_ms,
            pre_obs_ms=pre_obs_ms,
        )

    def _resolve_supervisor_action(
        self,
        sup: SupervisorDecision,
        targets: dict[str, dict[str, Any]],
        controls: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Map a SupervisorDecision (`operation`, `target`, `url`) to a concrete browser action dict."""
        op = (sup.operation or "").upper()
        if op == "FILL":
            op = "TYPE_TEXT"
            sup.operation = "TYPE_TEXT"
        target = sup.target
        cur_url = str(self.page.get("url", ""))

        if op in {"GOTO_URL", "NAVIGATE", "OPEN_URL"}:
            nav_url = _extract_candidate_url(sup, cur_url)
            if nav_url:
                sup.operation = "GOTO_URL"
                sup.url = nav_url
                return {"id": "goto_url", "kind": "goto_url", "label": f"Navigate to {nav_url}", "url": nav_url}

        if op in {"PRESS_ESCAPE", "PRESS_ENTER", "GO_BACK", "RELOAD"}:
            if op == "GO_BACK":
                self._record_rejected_from_history(cur_url)
            if op == "RELOAD":
                return {"id": "reload", "kind": "reload", "label": "Reload"}
            return {"id": op.lower(), "kind": op.lower(), "label": op.replace("_", " ").title()}

        if op in {"SCROLL_DOWN", "SCROLL_UP", "WAIT"}:
            if op in controls:
                return controls[op]
            if op == "WAIT":
                return {"id": "wait", "kind": "wait", "label": "Wait for the page to update"}
            delta = 560 if op == "SCROLL_DOWN" else -560
            return {"id": op.lower(), "kind": "scroll", "label": op.replace("_", " ").title(), "delta": delta}

        if op in targets and target:
            cand = targets[op].get(target)
            if cand:
                if op == "CLICK" and (
                    self._is_rejected_target(cand, cur_url)
                    or self._is_ineffective_click(cand, cur_url)
                ):
                    if "SCROLL_DOWN" in controls:
                        sup.operation = "SCROLL_DOWN"
                        sup.target = None
                        sup.reasoning = f"[Anti-Loop: skipped already-visited link '{cand.get('label', '')}' -> scrolling down] {sup.reasoning}"
                        return controls["SCROLL_DOWN"]
                    if self._is_ineffective_click(cand, cur_url):
                        return None
                return cand
            if op == "SELECT":
                options = [(k, v) for k, v in targets["SELECT"].items() if k.startswith(f"{target}:")]
                if sup.text:
                    needle = sup.text.strip().lower()
                    for _, v in options:
                        if needle in str(v.get("label", "")).lower() or needle == str(v.get("value", "")).lower():
                            return v
                if options:
                    return options[0][1]

        # Cross-group lookup if Supervisor omitted or mismatched operation name
        if target:
            ordered_ops = ["TYPE_TEXT", "CLICK", "SELECT"] if sup.text else ["CLICK", "TYPE_TEXT", "SELECT"]
            for cand_op in ordered_ops:
                group = targets.get(cand_op, {})
                if target in group:
                    cand = group[target]
                    if cand_op == "CLICK" and (
                        self._is_rejected_target(cand, cur_url)
                        or self._is_ineffective_click(cand, cur_url)
                    ):
                        if "SCROLL_DOWN" in controls:
                            sup.operation = "SCROLL_DOWN"
                            sup.target = None
                            return controls["SCROLL_DOWN"]
                        return None
                    sup.operation = cand_op
                    return cand

        # Fallback: if no element target resolved, check if Supervisor specified a new URL to navigate to
        fallback_url = _extract_candidate_url(sup, cur_url)
        if fallback_url:
            sup.operation = "GOTO_URL"
            sup.url = fallback_url
            return {
                "id": "goto_url",
                "kind": "goto_url",
                "label": f"Navigate to {fallback_url}",
                "url": fallback_url,
            }

        return None

    def _get_focused_form_submit_label(self) -> str | None:
        """Inspect the currently focused form in the browser to retrieve its submit button label."""
        if not hasattr(self, "browser") or not hasattr(self.browser, "evaluate"):
            return None
        js = (
            "(() => {"
            "  const active = document.activeElement;"
            "  if (!active) return null;"
            "  const form = active.closest('form');"
            "  if (!form) return null;"
            "  const btn = form.querySelector('button[type=\"submit\"], input[type=\"submit\"], button:not([type=\"button\"]):not([type=\"reset\"])');"
            "  if (!btn) return null;"
            "  return (btn.innerText || btn.value || btn.getAttribute('aria-label') || '').trim();"
            "})()"
        )
        try:
            res = self.browser.evaluate(js)
            if isinstance(res, str) and res.strip():
                return res.strip()
        except Exception:
            pass
        return None

    def _execute_and_verify(
        self,
        *,
        action: dict[str, Any],
        actor: str,
        operation: str,
        target: str | None,
        confidence: float,
        target_confidence: float | None,
        joint_confidence: float,
        target_margin: float,
        latency_ms: int,
        model_name: str,
        usage: dict[str, Any],
        preset_text: str | None,
        escalation_note: str | None,
        step_t0: float | None = None,
        s1_ms: int = 0,
        s2_ms: int = 0,
        pre_obs_ms: int = 0,
        package_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run safety check, text generation (if fill), CDP action execution, and 3-layer verification."""
        if step_t0 is None:
            step_t0 = time.perf_counter()
        # 1. Irreversible Action Safety Guard
        origin = extract_origin(action.get("origin")) or extract_origin(self.page.get("url"))
        if action.get("kind") == "press_enter":
            submit_label = action.get("submit_label") or self._get_focused_form_submit_label()
            if submit_label:
                action["submit_label"] = submit_label
                if not action.get("label") or action.get("label") in {"Press Enter", "press_enter"}:
                    action["label"] = submit_label

        safety = check_action_safety(
            action,
            origin=origin,
            approved_signatures=self.ctx.approved_actions,
            enabled=self.config.require_confirmation_for_destructive,
        )
        if safety.requires_confirmation:
            sig = safety.action_signature or (f"{action.get('kind')}:{action.get('label')}:{origin}" if origin else f"{action.get('kind')}:{action.get('label')}")
            self.pending_confirmation = {
                "action": action,
                "actor": actor,
                "operation": operation,
                "target": target,
                "confidence": confidence,
                "target_confidence": target_confidence,
                "joint_confidence": joint_confidence,
                "target_margin": target_margin,
                "latency_ms": latency_ms,
                "model_name": model_name,
                "usage": usage,
                "preset_text": preset_text,
                "escalation_note": escalation_note,
                "s1_ms": s1_ms,
                "s2_ms": s2_ms,
                "pre_obs_ms": pre_obs_ms,
                "summary": safety.action_summary,
                "reason": safety.reason,
                "signature": sig,
            }
            self.status = "waiting_confirmation"
            self.ctx.add_agent_message(
                f"Confirmation required before executing {safety.action_summary}: {safety.reason}",
                kind="confirm_action",
            )
            return self.snapshot()

        # 2. Text Generation for `fill` (`TYPE_TEXT`)
        typed_text = preset_text
        text_helper_model = None
        text_latency_ms = 0
        if action.get("kind") == "fill" and not typed_text:
            try:
                outcome = generate_field_text(
                    ctx=self.ctx,
                    action=action,
                    page=self.page,
                    history=self.history,
                    supervisor_model=self.config.supervisor_model,
                    allow_model_fallback=self.config.allow_model_fallback,
                    supervisor_timeout_s=self.config.supervisor_timeout_s,
                    reasoning_effort=self.config.supervisor_reasoning_effort,
                )
            except Exception as exc:
                self.status = "error"
                self.last_error = str(exc)
                return self.snapshot()
            self.text_calls.append(outcome.model_dump())
            text_helper_model = outcome.model
            text_latency_ms = int(outcome.latency_ms or 0)
            s2_ms += text_latency_ms
            if outcome.text is None:
                # Required value missing -> Pause and ask the user instead of crashing!
                self.status = "waiting_user"
                self.pending_fact_key = outcome.fact_key or "field_value"
                ask_msg = outcome.ask_user or f"What value should I enter into '{action.get('label')}'?"
                self.last_asked_question = ask_msg
                self.ctx.add_agent_message(ask_msg, kind="ask_user")
                return self.snapshot()
            typed_text = outcome.text

        # 3. Execute on Browser via CDP
        before_page = self.page
        act_res: dict[str, Any] = {}
        act_ms = 0
        stale_obs_ms = 0
        act_t0 = time.perf_counter()
        try:
            act_res = self.browser.act(action, before_page, text=typed_text) or {}
            act_ms += round((time.perf_counter() - act_t0) * 1000)
        except NavigationTimeout as exc:
            # A dead or hanging site is an observation for System 2 (pick another source), not a session crash.
            dead_url = str(action.get("url") or action.get("href") or typed_text or action.get("label", ""))
            self.ctx.rejected_links.append(f"site did not respond ({dead_url})")
            self.ctx.drop_queued_url(dead_url)
            self.pending_escalation_reason = ("navigation_timeout", str(exc))
            settled = self._observe_settled()
            if settled is not None:
                self.page = settled
            self.status = "ready"
            return self.snapshot()
        except TargetCoveredError as exc:
            act_ms += round((time.perf_counter() - act_t0) * 1000)
            if actor == "S1":
                return self._handle_escalation(
                    trigger_type="target_covered",
                    reason=str(exc),
                    s1_decision={"operation": operation, "target": target, "confidence": confidence},
                    step_t0=step_t0,
                    s1_ms=s1_ms,
                    pre_obs_ms=pre_obs_ms,
                )
            # If S2 also hit a covered target, automatically press Escape to dismiss the modal/overlay instead of looping
            try:
                self.browser._recovery_act("press_escape")
            except Exception:
                pass
            self.page = self.browser.observe(screenshot=self.config.screenshots)
            self.status = "ready"
            return self.snapshot()
        except StalePage as exc:
            act_ms += round((time.perf_counter() - act_t0) * 1000)
            obs_retry_t0 = time.perf_counter()
            stale_page = self._observe_settled()
            stale_obs_ms += round((time.perf_counter() - obs_retry_t0) * 1000)
            if stale_page is None:
                return self.snapshot()
            self.page = stale_page
            refreshed_act = next(
                (
                    a
                    for a in self.page.get("actions", [])
                    if a.get("kind") == action.get("kind")
                    and str(a.get("label", "")).strip() == str(action.get("label", "")).strip()
                ),
                None,
            )
            if refreshed_act is not None and refreshed_act.get("kind") == "click":
                # A raw name that still disagrees must not be clicked again.
                matcher = getattr(self.browser, "accessible_name_matches", None)
                if callable(matcher) and matcher(refreshed_act, self.page) is False:
                    refreshed_act = None
            if refreshed_act is not None:
                try:
                    before_page = self.page
                    action = refreshed_act
                    act_retry_t0 = time.perf_counter()
                    act_res = self.browser.act(action, before_page, text=typed_text) or {}
                    act_ms += round((time.perf_counter() - act_retry_t0) * 1000)
                except Exception as exc2:
                    self.pending_escalation_reason = ("stale_retry_failed", f"Retry after StalePage failed: {exc2}")
                    self.status = "ready"
                    return self.snapshot()
            else:
                if self.page.get("fingerprint") == before_page.get("fingerprint"):
                    self.pending_escalation_reason = ("stale_unchanged", f"Action could not execute on unchanged page: {exc}")
                self.status = "ready"
                return self.snapshot()

        obs_after_t0 = time.perf_counter()
        after_page = self._observe_settled()
        observe_ms = pre_obs_ms + stale_obs_ms + round((time.perf_counter() - obs_after_t0) * 1000)
        if after_page is None:
            self.unsettled_count += 1
            action_label = str(action.get("label", ""))
            if act_res.get("auto_submitted"):
                action_label = f"{action_label} (submitted search)"
            wall_step_ms = round((time.perf_counter() - step_t0) * 1000)
            step_ms = max(wall_step_ms, s1_ms + s2_ms + act_ms + observe_ms)
            step_record = {
                "step": len(self.history) + 1,
                "actor": actor,
                "action": action_label,
                "kind": action.get("kind", ""),
                "node": action.get("node"),
                "choice": action.get("id", ""),
                "operation": operation,
                "target": target,
                "rect": act_res.get("rect") or action.get("rect"),
                "href": action.get("href"),
                "confidence": confidence,
                "target_confidence": target_confidence,
                "joint_confidence": joint_confidence,
                "target_margin": target_margin,
                "latency_ms": latency_ms,
                "s1_ms": s1_ms,
                "s2_ms": s2_ms,
                "act_ms": act_ms,
                "observe_ms": observe_ms,
                "step_ms": step_ms,
                "model": model_name,
                "text": typed_text,
                "text_helper": text_helper_model,
                "text_latency_ms": text_latency_ms,
                "page_changed": False,
                "url_changed": False,
                "element_state_changed": False,
                "effective_progress": False,
                "verification": {
                    "effective_progress": False,
                    "page_changed": False,
                    "url_changed": False,
                    "element_state_changed": False,
                    "loop_detected": False,
                    "oscillation_detected": False,
                    "reason": "page_unsettled",
                    "note": "page_unsettled",
                    "status": "bilinmiyor",
                },
                "note": "page_unsettled",
                "escalation_note": "page_unsettled",
                "before_url": str(before_page.get("url", "")),
                "url": str(before_page.get("url", "")),
                "after_fingerprint": "",
                "usage": usage,
                "elapsed_ms": self.elapsed(),
            }
            self.history.append(step_record)
            self._log_trace_step(
                actor=actor,
                action=action,
                operation=operation,
                target=target,
                typed_text=typed_text,
                goto_url=None,
                verification={"effective_progress": False, "flags": {}, "reason": "page_unsettled"},
                task_status="blocked" if len(self.history) >= self.config.max_steps else None,
                page=before_page,
                step_no=len(self.history),
            )
            self.page = {}  # force subsequent step to re-observe without reusing stale page
            if self.unsettled_count >= 3:
                self.pending_escalation_reason = (
                    "page_unsettled",
                    f"Page remained unsettled for {self.unsettled_count} consecutive steps",
                )
            if len(self.history) >= self.config.max_steps:
                self.status = "blocked"
                self.last_error = f"Reached max steps limit ({self.config.max_steps})"
                self.ctx.add_agent_message(self.last_error, kind="escalation_note")
            return self.snapshot()
        self.unsettled_count = 0
        self.page = after_page
        after_url = str(after_page.get("url", ""))
        if after_url and after_url not in self.ctx.visited_urls:
            self.ctx.visited_urls.append(after_url)
        if action.get("kind") == "goto_url" and action.get("url"):
            nav_target_url = str(action.get("url")).strip()
            if nav_target_url and nav_target_url not in self.ctx.visited_urls:
                self.ctx.visited_urls.append(nav_target_url)
            self.ctx.drop_queued_url(nav_target_url)
        elif action.get("kind") == "click" and after_url != str(before_page.get("url", "")) and action.get("href"):
            self.ctx.drop_queued_url(str(action.get("href")))
        self.ctx.prune_visit_queue()

        # 4. Three-Layer Action Verification
        verification: VerificationResult = verify_step(
            action=action,
            before_page=before_page,
            after_page=after_page,
            typed_text=typed_text,
            history=self.history,
            max_no_progress_steps=self.config.thresholds.max_no_progress_steps,
        )

        # Clear one-step tactical hint once a System 1 action makes effective progress
        # (including scroll, so a "scroll down" hint never causes an infinite 13-step scroll streak!)
        if verification.effective_progress and actor == "S1":
            self.ctx.tactical_hint = ""

        if verification.url_changed:
            self._clear_ineffective_clicks()
        elif (
            action.get("kind") != "scroll"
            and verification.effective_progress
            and verification.page_changed
            and not verification.loop_detected
            and not verification.oscillation_detected
        ):
            self._clear_ineffective_clicks()
        elif (
            action.get("kind") == "scroll"
            and self.ctx.ineffective_clicks
            and before_page.get("text", "") == after_page.get("text", "")
        ):
            self._ineffective_fp = str(after_page.get("fingerprint", "") or "")

        if action.get("kind") == "click" and not verification.url_changed and (
            not verification.effective_progress
            or verification.loop_detected
            or verification.oscillation_detected
        ):
            self._record_ineffective_click(
                action,
                after_url or str(before_page.get("url", "")),
                str(after_page.get("fingerprint", "") or ""),
            )

        if actor == "S1":
            self._s1_steps_since_s2 += 1
        elif actor == "S2":
            self._s1_steps_since_s2 = 0

        action_label = str(action.get("label", ""))
        if act_res.get("auto_submitted"):
            action_label = f"{action_label} (submitted search)"

        wall_step_ms = round((time.perf_counter() - step_t0) * 1000)
        step_ms = max(wall_step_ms, s1_ms + s2_ms + act_ms + observe_ms)

        step_record = {
            "step": len(self.history) + 1,
            "actor": actor,  # "S1" (Jev/Turnike) or "S2" (Supervisor)
            "action": action_label,
            "kind": action.get("kind", ""),
            "node": action.get("node"),
            "choice": action.get("id", ""),
            "operation": operation,
            "target": target,
            "rect": act_res.get("rect") or action.get("rect"),
            "href": action.get("href"),
            "confidence": confidence,
            "target_confidence": target_confidence,
            "joint_confidence": joint_confidence,
            "target_margin": target_margin,
            "latency_ms": latency_ms,
            "s1_ms": s1_ms,
            "s2_ms": s2_ms,
            "act_ms": act_ms,
            "observe_ms": observe_ms,
            "step_ms": step_ms,
            "model": model_name,
            "text": typed_text,
            "text_helper": text_helper_model,
            "text_latency_ms": text_latency_ms,
            "page_changed": verification.page_changed,
            "url_changed": verification.url_changed,
            "element_state_changed": verification.element_state_changed,
            "effective_progress": verification.effective_progress,
            "verification": verification.model_dump(),
            "escalation_note": escalation_note,
            "before_url": str(before_page.get("url", "")),
            "url": after_url,
            "after_fingerprint": after_page.get("fingerprint", ""),
            "usage": usage,
            "elapsed_ms": self.elapsed(),
        }
        if package_meta is not None:
            step_record["package"] = package_meta
        self.history.append(step_record)
        self._log_trace_step(
            actor=actor,
            action=action,
            operation=operation,
            target=target,
            typed_text=typed_text,
            goto_url=None,
            verification=verification,
            task_status=None,
            page=before_page,
            step_no=len(self.history),
        )

        # 5. Schedule post-step escalation if a loop, oscillation, or scroll budget was reached
        if (
            verification.loop_detected
            or verification.oscillation_detected
            or verification.scroll_budget_exceeded
        ):
            if action.get("kind") == "click" and (verification.loop_detected or verification.oscillation_detected):
                self._record_rejected_from_history(after_url)
            if verification.oscillation_detected:
                trig = "oscillation"
            elif verification.scroll_budget_exceeded:
                trig = "scroll_budget"
            else:
                trig = "loop_detected"
            self.pending_escalation_reason = (trig, verification.reason)

        # 6. Milestone Triggers (only for planned tasks)
        if self.ctx.plan and self.ctx.plan.needs_plan and self.pending_escalation_reason is None:
            before_url_str = str(before_page.get("url", ""))
            b_dom = _extract_domain(before_url_str)
            a_dom = _extract_domain(after_url)
            site_changed = bool(b_dom and a_dom and b_dom != a_dom)

            twelve_s1 = self._s1_steps_since_s2 >= 12

            active_sg = self.ctx.plan.get_active_subgoal()
            subgoal_met = self._is_subgoal_completion_signaled(active_sg, verification, after_page)

            if site_changed:
                self.pending_escalation_reason = (
                    "progress_milestone",
                    f"Site changed from {b_dom} to {a_dom}",
                )
            elif twelve_s1:
                self.pending_escalation_reason = (
                    "progress_milestone",
                    f"12 consecutive S1 steps without S2 ({self._s1_steps_since_s2} steps)",
                )
            elif subgoal_met and active_sg:
                self.pending_escalation_reason = (
                    "progress_milestone",
                    f"Active subgoal criteria appear met for '{active_sg.text}'",
                )

        self.status = "ready"
        return self.snapshot()

    def reply_user(self, message: str) -> dict[str, Any]:
        """Handle a user message by letting the Supervisor model classify intent dynamically."""
        text = (message or "").strip()
        if not text:
            raise ValueError("Enter a non-empty message")

        try:
            interp = interpret_user_message(
                user_message=text,
                ctx=self.ctx,
                page=self.page,
                history=self.history,
                pending_fact_key=self.pending_fact_key,
                supervisor_model=self.config.supervisor_model,
                allow_model_fallback=self.config.allow_model_fallback,
                supervisor_timeout_s=self.config.supervisor_timeout_s,
                reasoning_effort=self.config.supervisor_reasoning_effort,
            )
            self.interpret_calls.append(interp.model_dump())
        except Exception as exc:
            self.status = "error"
            self.last_error = str(exc)
            return self.snapshot()

        # Case 1: User is asking the Supervisor a question or chatting (do NOT touch the browser!)
        if interp.intent == "CHAT_ONLY":
            self.ctx.add_user_message(text, kind="chat")
            reply_msg = interp.direct_reply or "Mevcut sayfa durumunu inceledim."
            self.ctx.add_agent_message(reply_msg, kind="chat")
            if self.status == "ready":
                self.status = "waiting_user"
                self.last_asked_question = reply_msg
            return self.snapshot()

        # Case 2: User is providing a requested fact (`ASK_USER` response)
        if interp.intent == "PROVIDE_FACT":
            fact_key = self.pending_fact_key or f"user_fact_{len(self.ctx.user_facts) + 1}"
            val = interp.fact_value or text
            self.pending_fact_key = None
            self.pending_confirmation = None
            self.pending_escalation_reason = None
            self._noop_escalation_counts.clear()
            self.ctx.add_user_message(text, kind="user_reply", fact_key=fact_key)
            self.ctx.user_facts[fact_key] = val
            if fact_key == "clarify_scope" and self.ctx.plan:
                match = re.search(r"\b\d+\b", val)
                if match:
                    try:
                        self.ctx.plan.budget.sources = max(1, int(match.group(0)))
                    except (ValueError, TypeError):
                        pass
            self.ctx.tactical_hint = interp.tactical_hint or f"User provided {fact_key}: {val}"
            self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
            self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)
            self.last_error = None
            self.status = "ready"
            return self.snapshot()

        if interp.allow_arbitrary_choices:
            self.ctx.allow_arbitrary_choices = True
        if interp.constraints:
            for c in interp.constraints:
                if c not in self.ctx.constraints:
                    self.ctx.constraints.append(c)

        # Case 3: User is switching to a new task/goal
        if interp.intent == "NEW_GOAL":
            self.ctx.add_user_message(text, kind="chat")
            clean_goal = interp.new_goal or text
            self.ctx.primary_goal = clean_goal
            self.ctx.active_subgoal = interp.updated_subgoal or clean_goal
            self.ctx.tactical_hint = interp.tactical_hint or ""
            self.ctx.working_memory = ""
            self.ctx.working_memory_notes.clear()
            if interp.plan and interp.plan.needs_plan:
                self.ctx.plan = interp.plan
                active_sg = self.ctx.plan.get_active_subgoal()
                if active_sg:
                    self.ctx.active_subgoal = active_sg.text
            elif interp.plan and not interp.plan.needs_plan:
                self.ctx.plan = None
            self.ctx.rejected_links.clear()
            self.ctx.visit_queue.clear()
            self.ctx.visit_queue_rejected_count = 0
            self._clear_ineffective_clicks()
            self.ctx.approved_actions.clear()
            self.ctx.constraints = list(interp.constraints)
            self._done_rejected_url = None
            self.pending_fact_key = None
            self.pending_confirmation = None
            self.pending_escalation_reason = None
            self._noop_escalation_counts.clear()
            self.last_error = None
            self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
            self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)

            if interp.target_url:
                if interp.target_url.rstrip("/").lower() != str(self.page.get("url", "")).rstrip("/").lower():
                    self.browser.navigate(interp.target_url)
                    self.page = self.browser.observe(screenshot=self.config.screenshots)
                nav_url = str(self.page.get("url", "")) if self.page else ""
                if not nav_url:
                    nav_url = interp.target_url
                if nav_url and nav_url not in self.ctx.visited_urls:
                    self.ctx.visited_urls.append(nav_url)

            if interp.direct_reply:
                self.ctx.add_agent_message(interp.direct_reply, kind="chat")
            self.status = "ready"
            return self.snapshot()

        # Case 4: Continue or steer current task
        self.ctx.add_user_message(text, kind="chat")
        if interp.new_goal:
            self.ctx.primary_goal = interp.new_goal

        if interp.plan and interp.plan.needs_plan:
            if self.ctx.plan is None:
                self.ctx.plan = interp.plan
                active_sg = self.ctx.plan.get_active_subgoal()
                if interp.updated_subgoal:
                    self.ctx.active_subgoal = interp.updated_subgoal
                elif active_sg:
                    self.ctx.active_subgoal = active_sg.text
                else:
                    self.ctx.active_subgoal = interp.new_goal or text
            else:
                # Preserve existing plan and completed results, append new subgoals
                existing_texts = {sg.text.casefold() for sg in self.ctx.plan.subgoals}
                existing_ids = {sg.id for sg in self.ctx.plan.subgoals}
                next_id_num = len(self.ctx.plan.subgoals) + 1
                for sg in interp.plan.subgoals:
                    if sg.text.casefold() not in existing_texts:
                        while f"sg_{next_id_num}" in existing_ids:
                            next_id_num += 1
                        new_id = f"sg_{next_id_num}"
                        next_id_num += 1
                        sg_copy = sg.model_copy(update={"id": new_id, "status": "pending"})
                        self.ctx.plan.subgoals.append(sg_copy)
                        existing_texts.add(sg.text.casefold())
                        existing_ids.add(new_id)
                if interp.updated_subgoal:
                    self.ctx.active_subgoal = interp.updated_subgoal
                else:
                    active_sg = self.ctx.plan.get_active_subgoal()
                    if active_sg:
                        self.ctx.active_subgoal = active_sg.text
                    else:
                        self.ctx.active_subgoal = interp.new_goal or text
        else:
            # If needs_plan is False or None, do NOT reset self.ctx.plan!
            if interp.updated_subgoal:
                self.ctx.active_subgoal = interp.updated_subgoal
            elif self.ctx.plan and self.ctx.plan.get_active_subgoal():
                self.ctx.active_subgoal = self.ctx.plan.get_active_subgoal().text
            else:
                self.ctx.active_subgoal = interp.new_goal or text

        self.ctx.tactical_hint = interp.tactical_hint or f"Focus on user instruction: {text}"
        self.pending_fact_key = None
        self.pending_confirmation = None
        self.pending_escalation_reason = None
        self._noop_escalation_counts.clear()
        self.last_error = None
        self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
        self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)

        if interp.target_url:
            if interp.target_url.rstrip("/").lower() != str(self.page.get("url", "")).rstrip("/").lower():
                self.browser.navigate(interp.target_url)
                self.page = self.browser.observe(screenshot=self.config.screenshots)
            nav_url = str(self.page.get("url", "")) if self.page else ""
            if not nav_url:
                nav_url = interp.target_url
            if nav_url and nav_url not in self.ctx.visited_urls:
                self.ctx.visited_urls.append(nav_url)
        if interp.direct_reply:
            self.ctx.add_agent_message(interp.direct_reply, kind="chat")
        self.status = "ready"
        return self.snapshot()

    def confirm_action(self, approved: bool, alternative_instruction: str = "") -> dict[str, Any]:
        """Approve or reject a pending irreversible action (`waiting_confirmation`)."""
        if not self.pending_confirmation:
            return self.snapshot()

        pending = self.pending_confirmation
        self.pending_confirmation = None

        if approved:
            sig = pending["signature"]
            self.ctx.approved_actions.add(sig)
            self.ctx.add_user_message(
                f"Approved action: {pending['summary']}",
                kind="user_reply",
            )
            self.status = "ready"
            res = self._execute_and_verify(
                action=pending["action"],
                actor=pending["actor"],
                operation=pending["operation"],
                target=pending["target"],
                confidence=pending["confidence"],
                target_confidence=pending["target_confidence"],
                joint_confidence=pending["joint_confidence"],
                target_margin=pending["target_margin"],
                latency_ms=pending["latency_ms"],
                model_name=pending["model_name"],
                usage=pending["usage"],
                preset_text=pending["preset_text"],
                escalation_note=pending["escalation_note"],
                s1_ms=int(pending.get("s1_ms", 0) or 0),
                s2_ms=int(pending.get("s2_ms", 0) or 0),
                pre_obs_ms=int(pending.get("pre_obs_ms", 0) or 0),
            )
            self.ctx.approved_actions.discard(sig)
            return res

        # User rejected the destructive action
        reason_msg = alternative_instruction.strip() or f"User declined executing {pending['summary']}."
        self.ctx.add_user_message(reason_msg, kind="user_reply")
        if alternative_instruction.strip():
            self.ctx.tactical_hint = f"Do NOT execute {pending['summary']}. Instead: {alternative_instruction.strip()}"
            self.status = "ready"
        else:
            self.status = "blocked"
            self.last_error = reason_msg
        return self.snapshot()

    def click_xy(self, x: float, y: float) -> dict[str, Any]:
        """Dispatch a manual user click at `(x, y)` (e.g., to pass a Cloudflare Turnstile / CAPTCHA box)."""
        self.browser.click_xy(float(x), float(y))
        self.page = self.browser.observe(screenshot=self.config.screenshots)
        if self.status in {"waiting_user", "blocked"}:
            self.status = "ready"
            self.last_error = None
        return self.snapshot()

    def _get_scrub_targets(self) -> tuple[list[str], list[str]]:
        """Collect user facts and environment secrets to redact, ordered by length descending."""
        facts_to_scrub: list[str] = []
        if hasattr(self, "ctx") and getattr(self.ctx, "user_facts", None):
            for raw_val in self.ctx.user_facts.values():
                if raw_val is not None:
                    s_val = str(raw_val).strip()
                    if s_val:
                        facts_to_scrub.append(s_val)
        facts_to_scrub = list(dict.fromkeys(facts_to_scrub))
        facts_to_scrub.sort(key=len, reverse=True)

        env_keys_to_scrub: list[str] = []
        for env_var in ("OPENROUTER_API_KEY", "META_API_KEY", "LLAMA_API_KEY", "BASETEN_API_KEY"):
            val = os.environ.get(env_var, "").strip()
            if val and len(val) >= 6:
                env_keys_to_scrub.append(val)
        for k, v in os.environ.items():
            if any(tok in k.upper() for tok in ("API_KEY", "SECRET", "AUTH_TOKEN", "BEARER")):
                clean_v = str(v).strip()
                if len(clean_v) >= 8:
                    env_keys_to_scrub.append(clean_v)
        env_keys_to_scrub = list(dict.fromkeys(env_keys_to_scrub))
        env_keys_to_scrub.sort(key=len, reverse=True)
        return facts_to_scrub, env_keys_to_scrub

    def _scrub_trace_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Scrub user facts and secrets from record dictionary while preserving schema integrity."""
        facts_to_scrub, env_keys_to_scrub = self._get_scrub_targets()

        def _scrub_str(val: Any, is_primary_fact_target: bool = False) -> Any:
            if not isinstance(val, str):
                return val
            res = val
            # 1. User facts scrubbing
            for s_val in facts_to_scrub:
                if len(s_val) >= 4 or is_primary_fact_target:
                    res = res.replace(s_val, "<redacted>")
            # 2. Active environment API keys & secrets
            for env_val in env_keys_to_scrub:
                res = res.replace(env_val, "<redacted>")
            # 3. Known API key patterns and Bearer tokens
            res = re.sub(r"sk-proj-[a-zA-Z0-9_\-]+", "<redacted>", res)
            res = re.sub(r"sk-or-v1-[a-zA-Z0-9_\-]+", "<redacted>", res)
            res = re.sub(r"sk-admin-[a-zA-Z0-9_\-]+", "<redacted>", res)
            res = re.sub(r"sk-ant-[a-zA-Z0-9_\-]+", "<redacted>", res)
            res = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "<redacted>", res)
            res = re.sub(r"LA-[a-zA-Z0-9_\-]+", "<redacted>", res)
            res = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\.\-]+", "Bearer <redacted>", res)
            return res

        rec = dict(record)

        # Scrub s1_input
        if "s1_input" in rec and isinstance(rec["s1_input"], dict):
            s1_in = dict(rec["s1_input"])
            if "compiled_goal" in s1_in and isinstance(s1_in["compiled_goal"], str):
                s1_in["compiled_goal"] = _scrub_str(s1_in["compiled_goal"], is_primary_fact_target=True)
            if "elements" in s1_in and isinstance(s1_in["elements"], list):
                clean_els = []
                for el in s1_in["elements"]:
                    if isinstance(el, dict):
                        el_copy = dict(el)
                        for k in ("label", "text", "value"):
                            if k in el_copy and isinstance(el_copy[k], str):
                                el_copy[k] = _scrub_str(el_copy[k], is_primary_fact_target=False)
                        if "options" in el_copy and isinstance(el_copy["options"], list):
                            clean_opts = []
                            for opt in el_copy["options"]:
                                if isinstance(opt, dict):
                                    opt_copy = dict(opt)
                                    for ok in ("label", "value"):
                                        if ok in opt_copy and isinstance(opt_copy[ok], str):
                                            opt_copy[ok] = _scrub_str(opt_copy[ok], is_primary_fact_target=False)
                                    clean_opts.append(opt_copy)
                                else:
                                    clean_opts.append(opt)
                            el_copy["options"] = clean_opts
                        clean_els.append(el_copy)
                    else:
                        clean_els.append(el)
                s1_in["elements"] = clean_els
            rec["s1_input"] = s1_in

        # Scrub typed_text (primary target for user facts)
        if "typed_text" in rec and isinstance(rec["typed_text"], str):
            rec["typed_text"] = _scrub_str(rec["typed_text"], is_primary_fact_target=True)

        # Scrub action and target_element
        if "action" in rec and isinstance(rec["action"], str):
            rec["action"] = _scrub_str(rec["action"], is_primary_fact_target=False)
        if "target_element" in rec and isinstance(rec["target_element"], dict):
            te = dict(rec["target_element"])
            if "label" in te and isinstance(te["label"], str):
                te["label"] = _scrub_str(te["label"], is_primary_fact_target=False)
            rec["target_element"] = te

        # Scrub URLs
        for url_field in ("url", "goto_url"):
            if url_field in rec and isinstance(rec[url_field], str):
                rec[url_field] = _scrub_str(rec[url_field], is_primary_fact_target=False)

        # Scrub validator reason
        if "validator" in rec and isinstance(rec["validator"], dict):
            v_dict = dict(rec["validator"])
            if "reason" in v_dict and isinstance(v_dict["reason"], str):
                v_dict["reason"] = _scrub_str(v_dict["reason"], is_primary_fact_target=False)
            rec["validator"] = v_dict

        return rec

    def _scrub_trace_text(self, text: str) -> str:
        """Redact sensitive user facts and API keys from a serialized JSON or text line."""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                scrubbed_dict = self._scrub_trace_record(parsed)
                return json.dumps(scrubbed_dict, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Raw string scrubbing fallback for non-JSON text
        result = text
        facts_to_scrub, env_keys_to_scrub = self._get_scrub_targets()

        for s_val in facts_to_scrub:
            result = result.replace(s_val, "<redacted>")
            try:
                encoded_s_val = json.dumps(s_val, ensure_ascii=False)[1:-1]
                if encoded_s_val != s_val:
                    result = result.replace(encoded_s_val, "<redacted>")
            except (TypeError, ValueError):
                pass

        for env_val in env_keys_to_scrub:
            result = result.replace(env_val, "<redacted>")

        result = re.sub(r"sk-proj-[a-zA-Z0-9_\-]+", "<redacted>", result)
        result = re.sub(r"sk-or-v1-[a-zA-Z0-9_\-]+", "<redacted>", result)
        result = re.sub(r"sk-admin-[a-zA-Z0-9_\-]+", "<redacted>", result)
        result = re.sub(r"sk-ant-[a-zA-Z0-9_\-]+", "<redacted>", result)
        result = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "<redacted>", result)
        result = re.sub(r"LA-[a-zA-Z0-9_\-]+", "<redacted>", result)
        result = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\.\-]+", "Bearer <redacted>", result)

        return result

    def _log_trace_step(
        self,
        *,
        actor: str = "S1",
        action: dict[str, Any] | None = None,
        operation: str = "",
        target: str | None = None,
        typed_text: str | None = None,
        goto_url: str | None = None,
        verification: Any = None,
        task_status: str | None = None,
        page: dict[str, Any] | None = None,
        step_no: int | None = None,
    ) -> None:
        """Write an execution trace step to JSONL if trace logging is enabled."""
        if not self.trace_path:
            return

        try:
            eff_page = page if page is not None else (self.page or {})
            page_url = str(eff_page.get("url") or getattr(self, "url", ""))
            page_actions = eff_page.get("actions", [])
            elements_raw, _, _ = action_space(page_actions, max_elements=MAX_SYSTEM1_ELEMENTS)
            clean_elements = [
                {k: v for k, v in el.items() if k not in ("rect", "node", "href")}
                for el in elements_raw
            ]
            site_hint = (
                self.site_memory.get_domain_hint(page_url)
                if getattr(self.config, "enable_site_memory", False)
                else ""
            )
            compiled_goal = self.ctx.compile_system1_goal(site_hint)
            s1_input = {
                "compiled_goal": compiled_goal,
                "elements": clean_elements,
            }

            eff_step = step_no if step_no is not None else max(1, len(self.history))
            mode_str = run_mode_of(self.config.s2_every_step, self.config.s1_strict, self.config.s2_action_packages)

            act = action or {}
            action_label = str(act.get("label") or act.get("action") or operation or "").strip()
            op_str = str(operation or act.get("kind") or "").upper()
            target_index = str(act.get("index") or act.get("id") or target or "") or None
            target_tag = act.get("role") or act.get("tag") or None
            target_href = act.get("href") or None
            eff_goto_url = goto_url or act.get("url") or (target if op_str in {"GOTO_URL", "NAVIGATE", "OPEN_URL"} else None)

            # Password redaction for typed text
            eff_typed_text = typed_text
            is_password = False
            type_attr = str(act.get("type") or act.get("input_type") or "").lower()
            role_attr = str(act.get("role") or "").lower()
            if type_attr == "password" or role_attr == "password":
                is_password = True

            pw_keywords = ("password", "passwd", "parola", "şifre", "sifre")
            candidate_strings = [
                str(act.get("label") or ""),
                str(act.get("aria-label") or ""),
                str(act.get("aria_label") or ""),
                str(act.get("placeholder") or ""),
                str(act.get("name") or ""),
                str(act.get("id") or ""),
                str(act.get("autocomplete") or ""),
                str(target or ""),
            ]
            if any(any(pw in s.lower() for pw in pw_keywords) for s in candidate_strings):
                is_password = True

            if is_password:
                eff_typed_text = "<redacted>"

            if isinstance(verification, VerificationResult):
                validator = {
                    "effective_progress": verification.effective_progress,
                    "flags": {
                        "page_changed": verification.page_changed,
                        "url_changed": verification.url_changed,
                        "element_state_changed": verification.element_state_changed,
                        "loop_detected": verification.loop_detected,
                        "oscillation_detected": verification.oscillation_detected,
                        "scroll_budget_exceeded": verification.scroll_budget_exceeded,
                    },
                    "reason": verification.reason,
                }
            elif isinstance(verification, dict):
                validator = verification
            else:
                validator = {
                    "effective_progress": bool(task_status == "done"),
                    "flags": {
                        "page_changed": False,
                        "url_changed": False,
                        "element_state_changed": False,
                        "loop_detected": False,
                        "oscillation_detected": False,
                        "scroll_budget_exceeded": False,
                    },
                    "reason": "terminal" if task_status else "none",
                }

            now_utc = datetime.now(timezone.utc)
            eff_task_status = task_status
            if eff_task_status is None and self.status in {"done", "blocked", "error"}:
                eff_task_status = self.status
            if eff_task_status is not None:
                self._terminal_trace_logged = True

            record = {
                "time": now_utc.isoformat(),
                "timestamp": now_utc.timestamp(),
                "session_id": getattr(self, "session_id", "default"),
                "step_no": eff_step,
                "url": page_url,
                "mode": mode_str,
                "s2_model": self.config.supervisor_model,
                "s1_input": s1_input,
                "decision_maker": actor,
                "actor": actor,
                "action": action_label,
                "operation": op_str,
                "target_index": target_index,
                "target_tag": target_tag,
                "target_href": target_href,
                "target_element": {
                    "role": target_tag,
                    "label": action_label,
                    "href": target_href,
                },
                "typed_text": eff_typed_text,
                "goto_url": eff_goto_url,
                "validator": validator,
                "progress_made": validator.get("effective_progress", False),
                "task_status": eff_task_status,
                "status": self.status,
            }

            scrubbed_record = self._scrub_trace_record(record)
            scrubbed_line = json.dumps(scrubbed_record, ensure_ascii=False)

            trace_file = Path(self.trace_path)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "a", encoding="utf-8") as f:
                f.write(scrubbed_line + "\n")
        except Exception:
            self.trace_write_errors += 1

    def _sync_trace_terminal_status(self) -> None:
        """Ensure trace_path reflects final session status via append-only logging without file rewrites."""
        if not self.trace_path:
            return
        if getattr(self, "_terminal_trace_logged", False):
            return
        if self.status in {"done", "blocked", "error", "waiting_user", "waiting_confirmation"}:
            self._log_trace_step(
                actor="system",
                action={"id": "terminal", "label": f"Session {self.status}", "kind": "terminal"},
                operation="TERMINAL",
                task_status=self.status,
            )
            self._terminal_trace_logged = True

    def close(self) -> None:
        try:
            self._sync_trace_terminal_status()
        except Exception:
            self.trace_write_errors += 1
        self.browser.close()

    def __enter__(self) -> ReflexSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


ReflexAgent = ReflexSession

