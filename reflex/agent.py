"""Reflex Hybrid Browser Agent Runtime (`ReflexSession`).

Combines System 1 (OpenRouter Jev / Local Turnike-Web 4B) with System 2 Supervisor
(`qwen/qwen3.8-omni-flash`, Meta Muse Spark, etc.), 3-layer verification, calibrated
escalation gates, irreversible action safety guards, and interactive human-in-the-loop chat.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .browser import Browser, StalePage, TargetCoveredError
from .config import ReflexConfig
from .context import DualTierContext
from .gate import GateEvaluation, evaluate_decision
from .memory import SiteMemoryStore
from .safety import check_action_safety
from .supervisor import (
    SupervisorDecision,
    generate_field_text,
    interpret_user_message,
    resolve_escalation,
)
from .system1 import action_space, choose_action
from .verifier import VerificationResult, verify_step

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

    def __init__(self, url: str, goal: str, config: ReflexConfig | None = None):
        if not goal or not goal.strip():
            raise ValueError("Supply a non-empty goal")
        self.config = config or ReflexConfig()
        clean_url = url.strip()
        goal_lower = goal.lower()
        if clean_url.rstrip("/") == "https://books.toscrape.com" and not any(
            kw in goal_lower for kw in ("toscrape", "sharp objects", "mystery", "books to scrape")
        ):
            clean_url = "https://www.google.com"
        self.url = clean_url
        self.ctx = DualTierContext(primary_goal=goal.strip(), active_subgoal=goal.strip())
        self.ctx.add_user_message(goal.strip(), kind="chat")
        self.site_memory = SiteMemoryStore()
        self.browser = Browser(self.url)
        try:
            self.page = self.browser.observe(screenshot=self.config.screenshots)
        except Exception:
            self.browser.close()
            raise

        self.status: str = "ready"  # "ready" | "waiting_user" | "waiting_confirmation" | "done" | "blocked" | "error"
        self.started_at: float = time.perf_counter()
        self.elapsed_ms: int = 0
        self.history: list[dict[str, Any]] = []
        self.decisions: list[dict[str, Any]] = []
        self.escalations: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

        # Pending state for pause/resume & post-verification escalation
        self.pending_fact_key: str | None = None
        self.pending_confirmation: dict[str, Any] | None = None
        self.pending_escalation_reason: tuple[str, str] | None = None  # (trigger_type, reason)
        self._noop_escalation_counts: dict[str, int] = {}
        self._done_rejected_url: tuple[str, int] | None = None
        self.last_error: str | None = None

    def elapsed(self) -> int:
        self.elapsed_ms = round((time.perf_counter() - self.started_at) * 1000)
        return self.elapsed_ms

    def snapshot(self) -> dict[str, Any]:
        """Return complete UI/telemetry state for the session."""
        elements, _, _ = action_space(self.page.get("actions", []))
        s1_calls = sum(1 for h in self.history if h.get("actor") == "S1")
        s2_calls = len(self.escalations)
        total_cost = round(
            sum(float((d.get("usage") or {}).get("cost", 0.0) or 0.0) for d in self.decisions)
            + sum(float(e.get("cost_usd", 0.0) or 0.0) for e in self.escalations)
            + sum(float(t.get("cost_usd", 0.0) or 0.0) for t in self.text_calls),
            6,
        )
        latencies = [h["latency_ms"] for h in self.history if isinstance(h.get("latency_ms"), (int, float))]
        median_ms = sorted(latencies)[len(latencies) // 2] if latencies else 0

        return {
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
            "last_error": self.last_error,
            "metrics": {
                "elapsed_ms": self.elapsed(),
                "steps": len(self.history),
                "s1_steps": s1_calls,
                "s2_escalations": s2_calls,
                "text_helper_calls": len(self.text_calls),
                "median_decision_ms": median_ms,
                "total_cost_usd": total_cost,
            },
        }

    def step(self) -> dict[str, Any]:
        """Execute a single hybrid step (Observe -> S1 or S2 Escalate -> Safety Guard -> Act -> Verify)."""
        if self.status in {"waiting_user", "waiting_confirmation", "done", "blocked", "error"}:
            return self.snapshot()

        if len(self.history) >= self.config.max_steps:
            self.status = "blocked"
            self.last_error = f"Reached max steps limit ({self.config.max_steps})"
            self.ctx.add_agent_message(self.last_error, kind="escalation_note")
            return self.snapshot()

        try:
            if not self.browser.fresh(self.page):
                self.page = self.browser.observe(screenshot=self.config.screenshots)
        except StalePage:
            self.page = self.browser.observe(screenshot=self.config.screenshots)

        # 1. Check if post-step verification from the previous action scheduled an escalation
        if self.pending_escalation_reason is not None:
            trig, reason = self.pending_escalation_reason
            self.pending_escalation_reason = None
            return self._handle_escalation(trigger_type=trig, reason=reason, s1_decision=None)

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

        try:
            s1_decision = choose_action(
                self.page,
                compiled_goal,
                self.history,
                self.config,
                suppress_done=suppress_done,
            )
        except Exception as exc:
            # If System 1 fails (e.g. local server down or transient API error), escalate to Supervisor!
            return self._handle_escalation(
                trigger_type="s1_error",
                reason=f"System 1 error: {exc}",
                s1_decision=None,
            )

        goal_l = (self.ctx.primary_goal + " " + self.ctx.active_subgoal).lower()
        page_meta_l = (str(self.page.get("title", "")) + " " + cur_norm_url).lower()
        is_ordering_puzzle = any(
            k in page_meta_l or k in goal_l
            for k in ("küçükten büyüğe", "büyükten küçüğe", "1-50", "1-100", "blind-ranking", "kor-siralama", "kör sıralama")
        )
        allow_subjective = (
            not is_ordering_puzzle
            and any(
                kw in goal_l
                for kw in ("kafana göre", "rastgele", "kendi yorumuna", "istediğin", "random", "tercih")
            )
        )

        gate_eval = evaluate_decision(
            s1_decision,
            self.config.thresholds,
            step_count=len(self.history),
            verify_done_with_supervisor=self.config.verify_done_with_supervisor,
            allow_subjective=allow_subjective,
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

        # 3. Escalation Gate Check
        if gate_eval.should_escalate:
            return self._handle_escalation(
                trigger_type=gate_eval.trigger_type,
                reason=gate_eval.reason,
                s1_decision=s1_decision,
                gate_eval=gate_eval,
            )

        # 4. High-Confidence DONE from System 1
        if s1_decision["operation"] == "DONE":
            self.status = "done"
            self.ctx.add_agent_message(
                "Goal completed by System 1 (high confidence verification passed).",
                kind="chat",
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
            )

        if s1_decision["operation"] == "CLICK" and str(action.get("label", "")).strip() in {str(i) for i in range(1, 11)}:
            slot_btn_count = sum(
                1
                for a in self.page.get("actions", [])
                if str(a.get("label", "")).strip() in {str(i) for i in range(1, 11)}
                or str(a.get("label", "")).startswith("[DOLU:")
            )
            rank_ctx = (self.ctx.primary_goal + " " + str(self.page.get("title", ""))).lower()
            if slot_btn_count >= 4 and any(k in rank_ctx for k in ("sıralama", "siralama", "ranking", "küçükten büyüğe", "büyükten küçüğe")):
                return self._handle_escalation(
                    trigger_type="ranking_logic",
                    reason=f"Blind ranking slot selection (S1 proposed slot '{action.get('label', '')}') requires System 2 interval reasoning",
                    s1_decision=s1_decision,
                    gate_eval=gate_eval,
                )

        if s1_decision["operation"] == "CLICK" and self._is_rejected_target(action, self.page.get("url", "")):
            if str(action.get("label", "")).startswith("[DOLU:"):
                return self._handle_escalation(
                    trigger_type="revisit_blocked",
                    reason=f"Slot '{action.get('label', '')}' is already occupied; pick an empty slot.",
                    s1_decision=s1_decision,
                    gate_eval=gate_eval,
                )
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
        )

    def _is_rejected_target(self, action: dict[str, Any], current_url: str) -> bool:
        """Return True if clicking `action` would revisit a navigation link/URL that was already visited and abandoned,
        re-click an already-selected radio button, or violate a 'do not submit / gönderme' goal constraint.
        """
        if action.get("kind") != "click":
            return False
        role = str(action.get("role", "")).lower()
        lbl = str(action.get("label", "")).strip().lower()
        href = str(action.get("href", "")).strip().rstrip("/").lower()

        # 1. Never click Submit / Gönder if the user's goal explicitly says not to submit
        goal_l = (self.ctx.primary_goal + " " + self.ctx.active_subgoal).lower()
        if any(neg in goal_l for neg in ("gönderme", "göndermeden", "do not submit", "don't submit", "without submitting")):
            if lbl in {"gönder", "submit", "formu gönder"} or lbl.endswith("→ gönder"):
                return True

        # 2. Never re-click a radio option that is already checked="true" or an occupied ranking slot
        if lbl.startswith("[dolu:"):
            return True
        if role in {"radio", "menuitemradio"} and str(action.get("checked", "")).lower() == "true":
            return True

        # 3. Never block unselected in-page form controls (radio, checkbox, option, button without href)
        if role in {"radio", "checkbox", "switch", "option", "menuitemradio", "tab", "combobox", "textbox", "searchbox"}:
            return False
        if not href and role != "link":
            return False
        href_norm = href.split("#")[0].rstrip("/").lower()
        for rej in self.ctx.rejected_links:
            rej_l = rej.strip().lower()
            if " (" in rej_l and rej_l.endswith(")"):
                rej_lbl, rej_href = rej_l.rsplit(" (", 1)
                rej_href_norm = rej_href[:-1].split("#")[0].rstrip("/").strip()
            else:
                rej_lbl, rej_href_norm = rej_l, ""
            if href_norm and len(href_norm) >= 8 and rej_href_norm and href_norm == rej_href_norm:
                return True
            if lbl and len(lbl) >= 4 and lbl == rej_lbl and (not href_norm or not rej_href_norm or href_norm == rej_href_norm):
                return True
        # Also check if we already clicked this exact link on this exact URL, navigated to a DIFFERENT URL, and later returned
        cur_norm = (current_url or "").split("#")[0].rstrip("/").lower()
        clicked_idx = -1
        for idx, h in enumerate(self.history):
            h_before = str(h.get("before_url") or "").split("#")[0].rstrip("/").lower()
            if (
                h.get("kind") == "click"
                and h.get("url_changed")
                and (not h_before or h_before == cur_norm)
                and str(h.get("action", "")).strip().lower() == lbl
            ):
                clicked_idx = idx
        if clicked_idx >= 0 and clicked_idx < len(self.history) - 1:
            for later in self.history[clicked_idx + 1 :]:
                if later.get("kind") == "go_back" or str(later.get("url", "")).split("#")[0].rstrip("/").lower() == cur_norm:
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

    def _handle_escalation(
        self,
        *,
        trigger_type: str,
        reason: str,
        s1_decision: dict[str, Any] | None,
        gate_eval: GateEvaluation | None = None,
    ) -> dict[str, Any]:
        """Run System 2 Supervisor (`qwen/qwen3.8-omni-flash`) to resolve an escalation."""
        if len(self.escalations) >= self.config.max_escalations:
            self.status = "blocked"
            self.last_error = f"Reached max supervisor escalations ({self.config.max_escalations}): {reason}"
            self.ctx.add_agent_message(self.last_error, kind="escalation_note")
            return self.snapshot()

        elements, targets, controls = action_space(self.page.get("actions", []))
        sup: SupervisorDecision = resolve_escalation(
            ctx=self.ctx,
            page=self.page,
            elements=elements,
            history=self.history,
            escalation_reason=reason,
            trigger_type=trigger_type,
            s1_decision=s1_decision,
            supervisor_model=self.config.supervisor_model,
        )

        if sup.memory:
            self.ctx.working_memory = sup.memory

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
            "updated_subgoal": sup.updated_subgoal,
            "tactical_hint": sup.tactical_hint,
            "question_for_user": sup.question_for_user,
            "model": sup.model,
            "latency_ms": sup.latency_ms,
            "cost_usd": sup.cost_usd,
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
        if sup.tactical_hint:
            self.ctx.tactical_hint = sup.tactical_hint

        # Case 1: Supervisor verifies the task is complete
        if sup.action_type == "COMPLETE_TASK":
            self.status = "done"
            msg = sup.summary_for_user or sup.reasoning or "Task verified complete by Supervisor."
            self.ctx.add_agent_message(msg, kind="chat")
            return self.snapshot()

        if trigger_type == "verify_done":
            cur_norm = str(self.page.get("url", "")).split("#")[0].rstrip("/")
            self._done_rejected_url = (cur_norm, len(self.history))

        # Case 2: Supervisor asks the human user for missing info / clarification
        if sup.action_type == "ASK_USER":
            self.status = "waiting_user"
            question = sup.question_for_user or sup.reasoning or "Could you clarify how you want to proceed?"
            self.pending_fact_key = sup.fact_key or "user_clarification"
            self.ctx.add_agent_message(question, kind="ask_user")
            return self.snapshot()

        current_fp = str(self.page.get("fingerprint", ""))

        # Case 3: Supervisor updates subgoal/hint without forcing a direct browser action
        if sup.action_type == "UPDATE_SUBGOAL":
            # If System 1 was BLOCKED and Supervisor suggested navigating to a new URL in hint/url, execute GOTO_URL!
            nav_url = _extract_candidate_url(sup, self.page.get("url", "")) if trigger_type in {"blocked", "s1_blocked"} else None
            if nav_url is None:
                self._noop_escalation_counts[current_fp] = self._noop_escalation_counts.get(current_fp, 0) + 1
                if self._noop_escalation_counts[current_fp] >= 2:
                    self.status = "waiting_user"
                    self.pending_fact_key = "user_guidance"
                    self.ctx.add_agent_message(
                        f"I am stuck on this page ({sup.reasoning}). Could you clarify or provide a different instruction?",
                        kind="ask_user",
                    )
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
            self._noop_escalation_counts[current_fp] = self._noop_escalation_counts.get(current_fp, 0) + 1
            if self._noop_escalation_counts[current_fp] >= 2:
                self.status = "waiting_user"
                self.pending_fact_key = "user_guidance"
                self.ctx.add_agent_message(
                    f"Supervisor could not resolve a valid browser action on this page ({sup.reasoning}). How should I proceed?",
                    kind="ask_user",
                )
                return self.snapshot()
            # If supervisor gave a subgoal update or invalid target once, let System 1 try with updated hint
            self.ctx.add_agent_message(
                f"Supervisor hint applied: {sup.reasoning}",
                kind="escalation_note",
            )
            self.status = "ready"
            return self.snapshot()

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
            escalation_note=f"[{trigger_type}] {sup.reasoning}",
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

        if op in {"PRESS_ESCAPE", "PRESS_ENTER", "GO_BACK"}:
            if op == "GO_BACK":
                self._record_rejected_from_history(cur_url)
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
                if op == "CLICK" and self._is_rejected_target(cand, cur_url):
                    if str(cand.get("label", "")).startswith("[DOLU:"):
                        for alt_idx, alt_cand in targets["CLICK"].items():
                            if str(alt_cand.get("label", "")).strip().isdigit():
                                sup.target = alt_idx
                                return alt_cand
                    if "SCROLL_DOWN" in controls:
                        sup.operation = "SCROLL_DOWN"
                        sup.target = None
                        sup.reasoning = f"[Anti-Loop: skipped already-visited link '{cand.get('label', '')}' -> scrolling down] {sup.reasoning}"
                        return controls["SCROLL_DOWN"]
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
                    if cand_op == "CLICK" and self._is_rejected_target(cand, cur_url):
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
    ) -> dict[str, Any]:
        """Run safety check, text generation (if fill), CDP action execution, and 3-layer verification."""
        # 1. Irreversible Action Safety Guard
        safety = check_action_safety(
            action,
            approved_signatures=self.ctx.approved_actions,
            enabled=self.config.require_confirmation_for_destructive,
        )
        if safety.requires_confirmation:
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
                "summary": safety.action_summary,
                "reason": safety.reason,
                "signature": f"{action.get('kind')}:{action.get('label')}",
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
            outcome = generate_field_text(
                ctx=self.ctx,
                action=action,
                page=self.page,
                history=self.history,
                supervisor_model=self.config.supervisor_model,
            )
            self.text_calls.append(outcome.model_dump())
            text_helper_model = outcome.model
            text_latency_ms = outcome.latency_ms
            if outcome.text is None:
                # Required value missing -> Pause and ask the user instead of crashing!
                self.status = "waiting_user"
                self.pending_fact_key = outcome.fact_key or "field_value"
                self.ctx.add_agent_message(
                    outcome.ask_user or f"What value should I enter into '{action.get('label')}'?",
                    kind="ask_user",
                )
                return self.snapshot()
            typed_text = outcome.text

        # 3. Execute on Browser via CDP
        before_page = self.page
        act_res: dict[str, Any] = {}
        try:
            act_res = self.browser.act(action, before_page, text=typed_text) or {}
        except TargetCoveredError as exc:
            if actor == "S1":
                return self._handle_escalation(
                    trigger_type="target_covered",
                    reason=str(exc),
                    s1_decision={"operation": operation, "target": target, "confidence": confidence},
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
            self.page = self.browser.observe(screenshot=self.config.screenshots)
            refreshed_act = next(
                (
                    a
                    for a in self.page.get("actions", [])
                    if a.get("kind") == action.get("kind")
                    and str(a.get("label", "")).strip() == str(action.get("label", "")).strip()
                ),
                None,
            )
            if refreshed_act is not None:
                try:
                    before_page = self.page
                    action = refreshed_act
                    act_res = self.browser.act(action, before_page, text=typed_text) or {}
                except Exception as exc2:
                    self.pending_escalation_reason = ("stale_retry_failed", f"Retry after StalePage failed: {exc2}")
                    self.status = "ready"
                    return self.snapshot()
            else:
                if self.page.get("fingerprint") == before_page.get("fingerprint"):
                    self.pending_escalation_reason = ("stale_unchanged", f"Action could not execute on unchanged page: {exc}")
                self.status = "ready"
                return self.snapshot()

        after_page = self.browser.observe(screenshot=self.config.screenshots)
        self.page = after_page
        after_url = str(after_page.get("url", ""))
        if after_url and after_url not in self.ctx.visited_urls:
            self.ctx.visited_urls.append(after_url)

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

        action_label = str(action.get("label", ""))
        if act_res.get("auto_submitted"):
            action_label = f"{action_label} (submitted search)"

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
        self.history.append(step_record)

        # 5. Schedule post-step escalation if a loop or oscillation was detected
        if verification.loop_detected or verification.oscillation_detected:
            if action.get("kind") == "click":
                self._record_rejected_from_history(after_url)
            trig = "oscillation" if verification.oscillation_detected else "loop_detected"
            self.pending_escalation_reason = (trig, verification.reason)

        self.status = "ready"
        return self.snapshot()

    def reply_user(self, message: str) -> dict[str, Any]:
        """Handle a user message by letting the Supervisor model classify intent dynamically."""
        text = (message or "").strip()
        if not text:
            raise ValueError("Enter a non-empty message")

        interp = interpret_user_message(
            user_message=text,
            ctx=self.ctx,
            page=self.page,
            history=self.history,
            pending_fact_key=self.pending_fact_key,
            supervisor_model=self.config.supervisor_model,
        )

        # Case 1: User is asking the Supervisor a question or chatting (do NOT touch the browser!)
        if interp.intent == "CHAT_ONLY":
            self.ctx.add_user_message(text, kind="chat")
            reply_msg = interp.direct_reply or "Mevcut sayfa durumunu inceledim."
            self.ctx.add_agent_message(reply_msg, kind="chat")
            if self.status == "ready":
                self.status = "waiting_user"
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
            self.ctx.tactical_hint = interp.tactical_hint or f"User provided {fact_key}: {val}"
            self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
            self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)
            self.last_error = None
            self.status = "ready"
            return self.snapshot()

        cur_url_lower = str(self.page.get("url", "")).lower()
        text_lower = text.lower()
        if not interp.target_url and any(
            kw in text_lower
            for kw in ("internette", "genel arat", "webde ara", "web'de ara", "google'da", "googleda", "search the web", "search online")
        ):
            if not any(se in cur_url_lower for se in ("google.", "bing.", "duckduckgo.")):
                interp.target_url = "https://www.google.com"

        # Case 3: User is switching to a new task/goal
        if interp.intent == "NEW_GOAL":
            self.ctx.add_user_message(text, kind="chat")
            clean_goal = interp.new_goal or text
            self.ctx.primary_goal = clean_goal
            self.ctx.active_subgoal = interp.updated_subgoal or clean_goal
            self.ctx.tactical_hint = interp.tactical_hint or ""
            self.ctx.working_memory = ""
            self.ctx.rejected_links.clear()
            self._done_rejected_url = None
            self.pending_fact_key = None
            self.pending_confirmation = None
            self.pending_escalation_reason = None
            self._noop_escalation_counts.clear()
            self.last_error = None
            self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
            self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)

            if interp.target_url and interp.target_url.rstrip("/").lower() != str(self.page.get("url", "")).rstrip("/").lower():
                self.browser.navigate(interp.target_url)
                self.page = self.browser.observe(screenshot=self.config.screenshots)

            if interp.direct_reply:
                self.ctx.add_agent_message(interp.direct_reply, kind="chat")
            self.status = "ready"
            return self.snapshot()

        # Case 4: Continue or steer current task
        self.ctx.add_user_message(text, kind="chat")
        if interp.new_goal:
            self.ctx.primary_goal = interp.new_goal
        self.ctx.active_subgoal = interp.updated_subgoal or interp.new_goal or text
        self.ctx.tactical_hint = interp.tactical_hint or f"Focus on user instruction: {text}"
        self.pending_fact_key = None
        self.pending_confirmation = None
        self.pending_escalation_reason = None
        self._noop_escalation_counts.clear()
        self.last_error = None
        self.config.max_steps = max(self.config.max_steps, len(self.history) + 60)
        self.config.max_escalations = max(self.config.max_escalations, len(self.escalations) + 30)
        if interp.target_url and interp.target_url.rstrip("/").lower() != str(self.page.get("url", "")).rstrip("/").lower():
            self.browser.navigate(interp.target_url)
            self.page = self.browser.observe(screenshot=self.config.screenshots)
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
            self.ctx.approved_actions.add(pending["signature"])
            self.ctx.add_user_message(
                f"Approved action: {pending['summary']}",
                kind="user_reply",
            )
            self.status = "ready"
            return self._execute_and_verify(
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
            )

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

    def close(self) -> None:
        self.browser.close()

    def __enter__(self) -> ReflexSession:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
