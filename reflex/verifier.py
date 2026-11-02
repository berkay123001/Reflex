"""Three-layer action verification and semantic loop / oscillation detector.

Addresses the limitation of relying solely on `page_changed == False`:
1. Technical Execution Check: Did CDP confirm the action without exception?
2. Element-Level State Diff: Did the target element update its value, selection, checked, or expanded state,
   or did new options/dialogs appear, or did page URL/content advance?
3. Semantic Loop & Ping-Pong Detection: Are we repeating the exact same (operation, target, value)
   or oscillating between fingerprints A -> B -> A -> B?
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured result of verifying a single browser step."""

    executed_ok: bool = True
    page_changed: bool = False
    url_changed: bool = False
    element_state_changed: bool = False
    effective_progress: bool = False
    loop_detected: bool = False
    oscillation_detected: bool = False
    scroll_budget_exceeded: bool = False
    reason: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


def _find_node_actions(page: dict[str, Any], node: int | None) -> list[dict[str, Any]]:
    if node is None:
        return []
    return [a for a in page.get("actions", []) if a.get("node") == node]


def verify_step(
    *,
    action: dict[str, Any],
    before_page: dict[str, Any],
    after_page: dict[str, Any],
    typed_text: str | None = None,
    history: list[dict[str, Any]] | None = None,
    max_no_progress_steps: int = 2,
) -> VerificationResult:
    """Evaluate a browser action across technical execution, element-level diff, and trajectory loops."""
    history = history or []
    before_fp = before_page.get("fingerprint", "")
    after_fp = after_page.get("fingerprint", "")
    page_changed = before_fp != after_fp
    url_changed = before_page.get("url", "") != after_page.get("url", "")

    kind = action.get("kind", "")
    node = action.get("node")
    before_nodes = _find_node_actions(before_page, node)
    after_nodes = _find_node_actions(after_page, node)

    element_state_changed = False
    details: dict[str, Any] = {"kind": kind, "node": node}

    if kind == "fill" and typed_text is not None:
        # Check if the target field's value now matches or contains the typed text, or if the form navigated
        after_vals = [str(a.get("value", "")).strip() for a in after_nodes]
        before_vals = [str(a.get("value", "")).strip() for a in before_nodes]
        target_norm = typed_text.strip().lower()
        if any(target_norm in v.lower() or v.lower() == target_norm for v in after_vals if v):
            element_state_changed = True
            details["fill_verified"] = True
        elif after_vals != before_vals:
            element_state_changed = True
            details["fill_verified"] = "value_changed"
        elif url_changed or not after_nodes:
            element_state_changed = True
            details["fill_verified"] = "submitted_or_unmounted"

    elif kind == "select":
        desired_val = str(action.get("value", ""))
        after_curr = [str(a.get("current_value", "")) for a in after_nodes]
        before_curr = [str(a.get("current_value", "")) for a in before_nodes]
        after_remaining_vals = {str(a.get("value", "")) for a in after_nodes if a.get("kind") == "select"}
        # In snapshot.js, selected <option> is excluded from remaining select actions
        if desired_val and desired_val not in after_remaining_vals:
            element_state_changed = True
            details["select_verified"] = True
        elif after_curr != before_curr:
            element_state_changed = True
            details["select_verified"] = "current_value_changed"

    elif kind == "click":
        # Check checkbox/radio/tab/expanded attribute transitions or newly opened options/dialogs
        before_meta = [(a.get("checked"), a.get("selected"), a.get("expanded"), a.get("value")) for a in before_nodes]
        after_meta = [(a.get("checked"), a.get("selected"), a.get("expanded"), a.get("value")) for a in after_nodes]
        before_action_ids = {
            (a.get("role"), a.get("label")) for a in before_page.get("actions", [])
        }
        after_action_ids = {
            (a.get("role"), a.get("label")) for a in after_page.get("actions", [])
        }
        if before_meta != after_meta:
            element_state_changed = True
            details["click_verified"] = "attribute_toggled"
        elif before_action_ids != after_action_ids:
            element_state_changed = True
            details["click_verified"] = "controls_updated"
        elif before_page.get("text", "") != after_page.get("text", ""):
            element_state_changed = True
            details["click_verified"] = "text_updated"

    elif kind == "scroll":
        before_y = int((before_page.get("scroll") or {}).get("y", 0) or 0)
        after_y = int((after_page.get("scroll") or {}).get("y", 0) or 0)
        scroll_moved = abs(after_y - before_y) >= 4

        # 1. Check for newly revealed interactive elements
        before_nodes = {a.get("node") for a in before_page.get("actions", []) if a.get("node") is not None}
        after_nodes = {a.get("node") for a in after_page.get("actions", []) if a.get("node") is not None}
        new_nodes = after_nodes - before_nodes

        before_sigs = {(a.get("role"), str(a.get("label", "")).strip(), a.get("href")) for a in before_page.get("actions", [])}
        after_sigs = {(a.get("role"), str(a.get("label", "")).strip(), a.get("href")) for a in after_page.get("actions", [])}
        new_sigs = after_sigs - before_sigs
        new_elements = bool(new_nodes or new_sigs)

        # 2. Check for newly revealed text
        before_lines = {l.strip() for l in before_page.get("text", "").splitlines() if len(l.strip()) >= 3}
        after_lines = {l.strip() for l in after_page.get("text", "").splitlines() if len(l.strip()) >= 3}
        new_lines = after_lines - before_lines

        before_words = set(re.findall(r"\b\w{3,}\b", before_page.get("text", "").lower()))
        after_words = set(re.findall(r"\b\w{3,}\b", after_page.get("text", "").lower()))
        new_words = after_words - before_words

        before_headings = set(before_page.get("headings", []))
        after_headings = set(after_page.get("headings", []))
        new_headings = after_headings - before_headings

        new_text = bool(new_lines or (len(new_words) >= 3) or new_headings)

        has_text_or_elements = bool(
            str(before_page.get("text", "") or "").strip()
            or str(after_page.get("text", "") or "").strip()
            or before_page.get("actions")
            or after_page.get("actions")
        )
        fingerprint_advanced = bool(
            not has_text_or_elements and before_fp and after_fp and before_fp != after_fp
        )
        scroll_revealed_content = bool(scroll_moved and (new_elements or new_text or fingerprint_advanced))
        element_state_changed = scroll_revealed_content
        if not scroll_revealed_content:
            page_changed = False

        details["scroll_moved"] = scroll_moved
        details["scroll_revealed_content"] = scroll_revealed_content
        details["new_elements_count"] = len(new_nodes or new_sigs)
        details["new_text_lines"] = len(new_lines)

    elif kind in {"wait", "press_escape", "press_enter", "go_back", "goto_url"}:
        element_state_changed = page_changed or url_changed

    effective_progress = url_changed or element_state_changed or page_changed

    # Check for repeated identical (kind, node, label, text) or consecutive no-progress steps
    sig = (action.get("kind"), action.get("node"), action.get("label"), typed_text)
    recent_sigs = [
        (h.get("kind"), h.get("node"), h.get("action"), h.get("text"))
        for h in history[-3:]
    ]
    same_action_count = sum(1 for s in recent_sigs if s == sig)

    no_progress_streak = 0
    if not effective_progress:
        no_progress_streak = 1
        for h in reversed(history):
            if not h.get("effective_progress", h.get("page_changed", False)):
                no_progress_streak += 1
            else:
                break

    scroll_streak = 0
    scroll_budget_exceeded = False
    if kind == "scroll":
        if effective_progress:
            scroll_streak = 1
            for h in reversed(history):
                if h.get("kind") == "scroll" and h.get("effective_progress", True):
                    scroll_streak += 1
                else:
                    break
            details["moving_scroll_streak"] = scroll_streak
            if scroll_streak >= 12 and scroll_streak % 12 == 0:
                scroll_budget_exceeded = True
        # Stuck at boundary is a repeated loop; moving without new content requires repeated attempts
        if not scroll_moved:
            repeated_without_progress = True
        else:
            repeated_without_progress = same_action_count >= 1 and not effective_progress
    else:
        repeated_without_progress = same_action_count >= 1 and not effective_progress

    loop_detected = repeated_without_progress or (no_progress_streak >= max_no_progress_steps)

    # Check for A -> B -> A -> B (2-step) or A -> B -> C -> A -> B -> C (3-step) fingerprint oscillation,
    # or repeating the exact same (kind, label, typed_text -> after_fingerprint) transition in recent history.
    # Pure in-page scroll steps are NEVER flagged as loops or oscillations as long as scroll.y changes.
    scroll_bounce = False
    if kind == "scroll":
        two_step_osc = False
        three_step_osc = False
        recent_scrolls = [str(h.get("choice") or h.get("action") or "").lower() for h in history[-2:] if h.get("kind") == "scroll"]
        cur_scroll = str(action.get("id") or action.get("label") or "").lower()
        if len(recent_scrolls) == 2 and recent_scrolls[0] == cur_scroll and recent_scrolls[1] != cur_scroll:
            scroll_bounce = True
    else:
        fps = [h.get("after_fingerprint") for h in history[-5:] if h.get("after_fingerprint")] + [after_fp]
        two_step_osc = (
            len(fps) >= 4
            and fps[-4] == fps[-2]
            and fps[-3] == fps[-1]
            and fps[-4] != fps[-3]
        )
        three_step_osc = (
            (
                len(fps) >= 6
                and fps[-6] == fps[-3]
                and fps[-5] == fps[-2]
                and fps[-4] == fps[-1]
                and len({fps[-6], fps[-5], fps[-4]}) >= 2
            )
            or (
                len(fps) >= 5
                and fps[-5] == fps[-2]
                and fps[-4] == fps[-1]
                and len({fps[-5], fps[-4], fps[-3]}) >= 3
            )
        )

    repeated_transition = False
    if kind not in ("scroll", "wait") and after_fp and len(history) >= 2:
        cur_trans = (kind, action.get("label"), typed_text, after_fp)
        action_label = action.get("label") or action.get("kind")
        prev_idx: int | None = None
        for idx in range(len(history) - 2, max(-1, len(history) - 9), -1):
            h = history[idx]
            h_trans = (h.get("kind"), h.get("action"), h.get("text"), h.get("after_fingerprint"))
            if h_trans == cur_trans or (
                h.get("kind"),
                h.get("action") or h.get("label"),
                h.get("text"),
                h.get("after_fingerprint"),
            ) == (kind, action_label, typed_text, after_fp):
                prev_idx = idx
                break

        if prev_idx is not None:
            intermediate_steps = history[prev_idx + 1 :]
            novel_page_found = False
            for offset, mid_step in enumerate(intermediate_steps):
                mid_idx = prev_idx + 1 + offset
                mid_fp = mid_step.get("after_fingerprint")
                if mid_fp:
                    prior_fps = {
                        h.get("after_fingerprint")
                        for h in history[:mid_idx]
                        if h.get("after_fingerprint")
                    }
                    if mid_fp not in prior_fps:
                        novel_page_found = True
                        break
            if not novel_page_found:
                repeated_transition = True

    oscillation_detected = two_step_osc or three_step_osc or repeated_transition or scroll_bounce

    reason = "ok"
    if oscillation_detected:
        if scroll_bounce:
            reason = "Scroll bounce detected (alternating scroll down and scroll up)"
        elif repeated_transition:
            reason = f"Navigation cycle detected: repeated '{action.get('label')}' returning to an already visited state"
        else:
            reason = "State oscillation (multi-page navigation cycle) detected"
    elif loop_detected:
        if same_action_count >= 1:
            reason = f"Repeated action ({action.get('kind')}: {action.get('label')}) without progress"
        else:
            reason = f"{no_progress_streak} consecutive steps with no element or page progress"
    elif scroll_budget_exceeded:
        reason = "12 scrolls performed; can goal be answered from page text?"
    elif not effective_progress:
        reason = "No observable element or page state change"

    return VerificationResult(
        executed_ok=True,
        page_changed=page_changed,
        url_changed=url_changed,
        element_state_changed=element_state_changed,
        effective_progress=effective_progress,
        loop_detected=loop_detected,
        oscillation_detected=oscillation_detected,
        scroll_budget_exceeded=scroll_budget_exceeded,
        reason=reason,
        details=details,
    )
