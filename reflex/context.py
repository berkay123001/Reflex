"""Dual-Tier Bounded Context Manager for Reflex.

Maintains strict separation between:
1. `user_facts`: Information explicitly supplied or confirmed by the human user.
2. `verified_dom_facts`: Provenance-tagged observations verified from the DOM.
3. `agent_hypotheses`: Tactical subgoals, supervisor hints, and active plan state.

Compiles a clean, unpolluted `Goal:` string for System 1 (Jev / Turnike) while keeping
the System 2 (Supervisor) prompt window strictly bounded.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from .safety import sanitize_untrusted_text


class ChatMessage(BaseModel):
    """A single message in the user <-> agent conversation."""

    role: str  # "user" | "assistant" | "system_notice"
    content: str
    kind: str = "chat"  # "chat" | "ask_user" | "confirm_action" | "user_reply" | "escalation_note"
    timestamp: float = Field(default_factory=time.time)


class VerifiedDomFact(BaseModel):
    """A piece of information verified directly from an observed webpage."""

    key: str
    value: str
    source_url: str
    step: int


class DualTierContext(BaseModel):
    """Session context separating System 1 reflex goal state from System 2 bounded memory."""

    primary_goal: str
    active_subgoal: str = ""
    tactical_hint: str = ""
    working_memory: str = ""
    subgoal_queue: list[str] = Field(default_factory=list)
    completed_subgoals: list[str] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    rejected_links: list[str] = Field(default_factory=list)

    # Strictly separated knowledge stores
    user_facts: dict[str, str] = Field(default_factory=dict)
    verified_dom_facts: list[VerifiedDomFact] = Field(default_factory=list)
    agent_hypotheses: dict[str, str] = Field(default_factory=dict)

    # Full chat log for UI display vs. rolling summary for bounded LLM context
    chat_log: list[ChatMessage] = Field(default_factory=list)
    older_chat_summary: str = ""
    compacted_up_to: int = 0
    max_chat_window: int = 8
    max_dom_facts: int = 15

    # Approved irreversible action signatures ("click:Place Order")
    approved_actions: set[str] = Field(default_factory=set)

    # Structured arbitrary/subjective choices permission
    allow_arbitrary_choices: bool = False

    def add_user_message(self, content: str, kind: str = "chat", fact_key: str | None = None) -> None:
        """Record a user message and store structured facts in `user_facts` when `fact_key` is provided."""
        text = content.strip()
        if not text:
            return
        self.chat_log.append(ChatMessage(role="user", content=text, kind=kind))
        if fact_key:
            self.user_facts[fact_key] = text
        self._compact_chat_if_needed()

    def add_agent_message(self, content: str, kind: str = "chat") -> None:
        """Record an assistant/supervisor message to the user."""
        text = content.strip()
        if not text:
            return
        self.chat_log.append(ChatMessage(role="assistant", content=text, kind=kind))
        self._compact_chat_if_needed()

    def record_dom_fact(self, key: str, value: str, source_url: str, step: int) -> None:
        """Record a verified observation from the DOM, keeping the list bounded."""
        self.verified_dom_facts = [f for f in self.verified_dom_facts if f.key != key]
        self.verified_dom_facts.append(
            VerifiedDomFact(key=key, value=value[:300], source_url=source_url[:200], step=step)
        )
        if len(self.verified_dom_facts) > self.max_dom_facts:
            self.verified_dom_facts = self.verified_dom_facts[-self.max_dom_facts :]

    def _compact_chat_if_needed(self) -> None:
        """Fold newly overflowed chat turns into a compact summary string once."""
        cutoff = len(self.chat_log) - self.max_chat_window
        if cutoff <= self.compacted_up_to:
            return
        new_overflow = self.chat_log[self.compacted_up_to : cutoff]
        self.compacted_up_to = cutoff
        snippets = [f"{m.role}: {sanitize_untrusted_text(m.content, 120)}" for m in new_overflow]
        addition = " | ".join(snippets)
        self.older_chat_summary = f"{self.older_chat_summary} | {addition}".strip(" |")[-800:]

    def compile_system1_goal(self, site_memory_hint: str = "") -> str:
        """Compile a clean, compact Goal string for System 1 (Jev / Turnike-Web).

        Preserves the training distribution of System 1 while injecting active subgoals,
        tactical recovery hints, and user-provided facts.
        """
        parts: list[str] = [self.primary_goal.strip()]
        if self.active_subgoal and self.active_subgoal.strip() != self.primary_goal.strip():
            parts.append(f"Current subgoal: {self.active_subgoal.strip()}")
        if self.tactical_hint:
            parts.append(f"Note: {self.tactical_hint.strip()}")
        if site_memory_hint:
            parts.append(f"Site rule: {site_memory_hint.strip()}")
        if self.user_facts:
            facts_str = "; ".join(f"{k}={v}" for k, v in list(self.user_facts.items())[-8:])
            parts.append(f"User-provided details: {facts_str}")
        return "\n".join(parts)

    def build_supervisor_context_block(self, history: list[dict[str, Any]]) -> str:
        """Build the bounded, provenance-separated context summary for System 2 (Supervisor)."""
        lines: list[str] = [
            f"Primary Goal: {self.primary_goal}",
            f"Active Subgoal: {self.active_subgoal or self.primary_goal}",
        ]
        if self.tactical_hint:
            lines.append(f"Current Tactical Hint: {self.tactical_hint}")
        if self.working_memory:
            lines.append(f"Cumulative Working Memory: {self.working_memory}")
        if self.rejected_links:
            lines.append(f"Already Visited & Rejected Links (DO NOT CLICK THESE AGAIN): {self.rejected_links[-10:]}")
        if self.subgoal_queue:
            lines.append(f"Remaining Subgoals: {self.subgoal_queue}")
        if self.completed_subgoals:
            lines.append(f"Completed Subgoals: {self.completed_subgoals[-5:]}")

        # 1. Explicit User Facts (Provenance: human)
        if self.user_facts:
            lines.append("User Facts (verified from human user):")
            for k, v in list(self.user_facts.items())[-10:]:
                lines.append(f"  - {k}: {v}")

        # 2. Verified DOM Facts (Provenance: page state, sanitized)
        if self.verified_dom_facts:
            lines.append("Verified DOM Observations:")
            for f in self.verified_dom_facts[-8:]:
                k_clean = sanitize_untrusted_text(f.key, 80)
                v_clean = sanitize_untrusted_text(f.value, 200)
                lines.append(f"  - [step {f.step}] {k_clean}: {v_clean}")

        # 3. Compact Trajectory Summary (no raw historical DOMs!)
        if history:
            lines.append("Recent Execution Trajectory (most recent last):")
            for h in history[-8:]:
                actor = h.get("actor", "S1")
                op = sanitize_untrusted_text(str(h.get("operation", h.get("kind", ""))), 40)
                lbl = sanitize_untrusted_text(str(h.get("action", "")), 120)
                txt = f" text={sanitize_untrusted_text(str(h['text']), 80)!r}" if h.get("text") else ""
                url_c = sanitize_untrusted_text(str(h.get("url", "")), 160)
                prog = "progressed" if h.get("effective_progress") else "no-change"
                lines.append(f"  - Step {h.get('step')} [{actor}] {op} '{lbl}'{txt} -> {prog} (url={url_c})")

        # 4. Bounded Chat Window
        if self.older_chat_summary:
            lines.append(f"Earlier Chat Summary: {self.older_chat_summary}")
        recent_chat = self.chat_log[-self.max_chat_window :]
        if recent_chat:
            lines.append("Recent User-Agent Conversation:")
            for m in recent_chat:
                lines.append(f"  [{m.role} ({m.kind})]: {m.content[:300]}")

        return "\n".join(lines)

    def snapshot_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the context for the frontend inspector."""
        return {
            "primary_goal": self.primary_goal,
            "active_subgoal": self.active_subgoal or self.primary_goal,
            "tactical_hint": self.tactical_hint,
            "working_memory": self.working_memory,
            "visited_urls": self.visited_urls[-15:],
            "rejected_links": self.rejected_links[-15:],
            "subgoal_queue": self.subgoal_queue,
            "completed_subgoals": self.completed_subgoals,
            "user_facts": self.user_facts,
            "verified_dom_facts": [f.model_dump() for f in self.verified_dom_facts],
            "agent_hypotheses": self.agent_hypotheses,
            "chat_log": [m.model_dump() for m in self.chat_log],
            "approved_actions": sorted(self.approved_actions),
        }
