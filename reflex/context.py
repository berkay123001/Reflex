"""Dual-Tier Bounded Context Manager for Reflex.

Maintains strict separation between:
1. `user_facts`: Information explicitly supplied or confirmed by the human user.
2. `verified_dom_facts`: Provenance-tagged observations verified from the DOM.
3. `agent_hypotheses`: Tactical subgoals, supervisor hints, and active plan state.

Compiles a clean, unpolluted `Goal:` string for System 1 (Jev / Turnike) while keeping
the System 2 (Supervisor) prompt window strictly bounded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from .browser.cdp import canonical_url
from .planner import TaskLedger
from .safety import sanitize_untrusted_text

MAX_VISIT_QUEUE_URLS = 30
MAX_VISIT_QUEUE_URL_CHARS = 200
_REJECTED_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)


def extract_domain(url: str | None) -> str:
    """Extract normalized hostname from URL, stripping port, scheme, 'www.', and trailing dots.

    Returns empty string for non-HTTP(S) schemes (about:, javascript:, data:, chrome:) or empty input.
    """
    if not url:
        return ""
    clean = str(url).strip()
    if not clean:
        return ""
    lower = clean.lower()
    if lower.startswith(("about:", "javascript:", "data:", "chrome:", "chrome-extension:", "file:", "mailto:", "tel:")):
        return ""
    if clean.startswith("#"):
        return ""
    if "://" not in clean and not clean.startswith("//"):
        clean = f"http://{clean}"
    try:
        parts = urlsplit(clean)
        if parts.scheme and parts.scheme.lower() not in ("http", "https"):
            return ""
        hostname = parts.hostname or ""
        if not hostname and parts.netloc:
            hostname = parts.netloc.split(":")[0].strip("[]")
        hostname = hostname.lower().strip().rstrip(".")
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def _url_match_keys(raw_url: str) -> set[str]:
    """Return canonical_url(raw_url) and HTTP(S)/www-normalized equivalent for queue matching."""
    c = canonical_url(raw_url)
    if not c:
        return set()
    keys = {c}
    try:
        parts = urlsplit(c)
        if parts.scheme in ("http", "https") and parts.netloc:
            host = parts.netloc.removeprefix("www.")
            keys.add(urlunsplit(("https", host, parts.path or "/", parts.query, "")))
    except Exception:
        pass
    return keys


def _extract_rejected_url_keys(rejected_entry: str) -> set[str]:
    """Extract canonical URL keys from a `rejected_links` entry (e.g. 'label (url)' or 'site did not respond (url)')."""
    entry = str(rejected_entry or "").strip()
    if not entry:
        return set()
    keys: set[str] = set()
    if " (" in entry and entry.endswith(")"):
        _, paren_url = entry.rsplit(" (", 1)
        keys.update(_url_match_keys(paren_url[:-1].strip()))
    for m in _REJECTED_URL_RE.findall(entry):
        keys.update(_url_match_keys(m.rstrip(".,;:!?)']\"")))
    if "://" in entry and " " not in entry:
        keys.update(_url_match_keys(entry))
    return keys


class _PruningList(list[str]):
    """List subclass that triggers visit-queue pruning when mutated in place."""

    def __init__(self, iterable: Iterable[str] = (), on_mutate: Callable[[], None] | None = None) -> None:
        super().__init__(iterable)
        self._on_mutate = on_mutate

    def _notify(self) -> None:
        if self._on_mutate is not None:
            self._on_mutate()

    def append(self, item: str) -> None:
        super().append(item)
        self._notify()

    def extend(self, iterable: Iterable[str]) -> None:
        super().extend(iterable)
        self._notify()

    def insert(self, index: Any, item: str) -> None:
        super().insert(index, item)
        self._notify()

    def __setitem__(self, index: Any, value: Any) -> None:
        super().__setitem__(index, value)
        self._notify()


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
    working_memory_notes: list[str] = Field(default_factory=list)
    plan: TaskLedger | None = None
    # False for unattended runs (eval, batch): System 2 must not pause for preferences nobody will answer.
    user_present: bool = True
    # Registrable site the task is bound to ("" = open web, e.g. the run started on a search engine).
    home_site: str = ""
    # Session-level tracking for questions converted in unattended mode
    unattended_questions: list[str] = Field(default_factory=list)
    subgoal_queue: list[str] = Field(default_factory=list)
    completed_subgoals: list[str] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    rejected_links: list[str] = Field(default_factory=list)
    visit_queue: list[str] = Field(default_factory=list)
    visit_queue_rejected_count: int = 0
    ineffective_clicks: list[tuple[str, str, str]] = Field(default_factory=list)

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

    # Goal & user constraints (e.g. "do not submit form", "only add to cart")
    constraints: list[str] = Field(default_factory=list)

    # Domain blocking, domain guidance, and scroll limiter state (G12 M3 / R2)
    blocked_domains: set[str] = Field(default_factory=set)
    domain_guidance: dict[str, str] = Field(default_factory=dict)
    scroll_warning: str = ""

    def is_domain_blocked(self, url: str) -> bool:
        """Return True if url's domain matches any blocked domain or subdomain thereof."""
        if not self.blocked_domains or not url:
            return False
        d = extract_domain(url)
        if not d:
            return False
        if d in self.blocked_domains:
            return True
        return any(d.endswith("." + b) for b in self.blocked_domains)

    def block_domain(self, domain: str, reason: str = "showing no content") -> None:
        """Mark a domain as blocked, record in rejected_links and S2 context, and prune visit queue."""
        clean_d = extract_domain(domain)
        if not clean_d:
            return
        if clean_d not in self.blocked_domains:
            self.blocked_domains.add(clean_d)
            rej_entry = f"domain {reason} ({clean_d})"
            if rej_entry not in self.rejected_links:
                self.rejected_links.append(rej_entry)
            self.prune_visit_queue()

    def model_post_init(self, __context: Any) -> None:
        self._wrap_pruning_lists()
        self.prune_visit_queue()

    def _wrap_pruning_lists(self) -> None:
        if not isinstance(self.visited_urls, _PruningList):
            object.__setattr__(self, "visited_urls", _PruningList(self.visited_urls, on_mutate=self.prune_visit_queue))
        else:
            self.visited_urls._on_mutate = self.prune_visit_queue
        if not isinstance(self.rejected_links, _PruningList):
            object.__setattr__(self, "rejected_links", _PruningList(self.rejected_links, on_mutate=self.prune_visit_queue))
        else:
            self.rejected_links._on_mutate = self.prune_visit_queue

    def _blocked_url_keys(self) -> set[str]:
        blocked: set[str] = set()
        for u in self.visited_urls:
            blocked.update(_url_match_keys(u))
        for rej in self.rejected_links:
            blocked.update(_extract_rejected_url_keys(rej))
        return blocked

    def prune_visit_queue(self) -> None:
        """Drop visited, rejected, or blocked-domain URLs and deduplicate `visit_queue` in place."""
        if not self.visit_queue:
            return
        blocked = self._blocked_url_keys()
        seen: set[str] = set()
        kept: list[str] = []
        for raw_u in self.visit_queue:
            clipped = str(raw_u or "").strip()[:MAX_VISIT_QUEUE_URL_CHARS]
            if not clipped:
                continue
            if self.is_domain_blocked(clipped):
                continue
            u_keys = _url_match_keys(clipped)
            if not u_keys or (u_keys & blocked) or (u_keys & seen):
                continue
            seen.update(u_keys)
            kept.append(clipped)
            if len(kept) >= MAX_VISIT_QUEUE_URLS:
                break
        self.visit_queue[:] = kept

    def drop_queued_url(self, url: str) -> None:
        """Explicitly remove `url` (matched via canonical_url) from `visit_queue`."""
        if not url or not self.visit_queue:
            return
        drop_keys = _url_match_keys(str(url).strip()[:MAX_VISIT_QUEUE_URL_CHARS]) | _url_match_keys(str(url).strip())
        if not drop_keys:
            return
        self.visit_queue[:] = [
            u for u in self.visit_queue if not (_url_match_keys(u) & drop_keys)
        ]

    def enqueue_urls(
        self,
        urls: Iterable[str],
        page_hrefs: Iterable[str | dict[str, Any]] | None = None,
    ) -> list[str]:
        """Ground candidate URLs against current step's observed `href`s and append valid ones to `visit_queue`."""
        self._wrap_pruning_lists()
        self.prune_visit_queue()

        allowed_keys: set[str] = set()
        for item in page_hrefs or ():
            raw_h = item.get("href", "") if isinstance(item, dict) else str(item or "")
            clean_h = str(raw_h or "").strip()
            if not clean_h or clean_h == "#" or clean_h.lower().startswith("javascript:"):
                continue
            allowed_keys.update(_url_match_keys(clean_h))
            allowed_keys.update(_url_match_keys(clean_h[:MAX_VISIT_QUEUE_URL_CHARS]))
            allowed_keys.update(_url_match_keys(clean_h[:180]))

        blocked = self._blocked_url_keys()
        for existing in self.visit_queue:
            blocked.update(_url_match_keys(existing))

        accepted: list[str] = []
        for raw_u in urls or ():
            if not isinstance(raw_u, str):
                continue
            clean_u = raw_u.strip()
            if not clean_u:
                continue
            clipped = clean_u[:MAX_VISIT_QUEUE_URL_CHARS]
            if self.is_domain_blocked(clean_u) or self.is_domain_blocked(clipped):
                self.visit_queue_rejected_count += 1
                continue
            u_keys = _url_match_keys(clipped) | _url_match_keys(clean_u)
            if not u_keys or not (u_keys & allowed_keys):
                self.visit_queue_rejected_count += 1
                continue
            if u_keys & blocked:
                continue
            if len(self.visit_queue) < MAX_VISIT_QUEUE_URLS:
                self.visit_queue.append(clipped)
                accepted.append(clipped)
                blocked.update(u_keys)
        return accepted

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return chat messages as dictionaries for backward and test compatibility."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "message": m.content,
                "kind": m.kind,
                "timestamp": m.timestamp,
            }
            for m in self.chat_log
        ]

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

    def add_working_memory_note(self, note: str) -> None:
        """Append an escalation memory note to the additive notebook when no structured plan exists."""
        if not isinstance(note, str):
            return
        text = note.strip()
        if not text:
            return
        if not self.working_memory_notes or text.casefold() != self.working_memory_notes[-1].casefold():
            self.working_memory_notes.append(text)
            if len(self.working_memory_notes) > 12:
                self.working_memory_notes = self.working_memory_notes[-12:]
            while len(self.working_memory_notes) > 1 and sum(len(n) for n in self.working_memory_notes) > 1500:
                self.working_memory_notes.pop(0)
            self.working_memory = "\n".join(self.working_memory_notes)

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
        if self.constraints:
            parts.append(f"Constraints: {'; '.join(self.constraints)}")
        if self.tactical_hint:
            parts.append(f"Note: {self.tactical_hint.strip()}")
        if site_memory_hint:
            parts.append(f"Site rule: {site_memory_hint.strip()}")
        if self.user_facts:
            facts_str = "; ".join(f"{k}={v}" for k, v in list(self.user_facts.items())[-8:])
            parts.append(f"User-provided details: {facts_str}")
        return "\n".join(parts)

    def build_supervisor_context_block(self, history: list[dict[str, Any]]) -> str:
        """Build the bounded, provenance-separated context summary for System 2 (Supervisor).

        Sections are ordered strictly by frequency of change to maximize prompt cache hits:
        - Tier 1: Static throughout task (Goal, Constraints, User availability, User facts)
        - Tier 2: Slow-changing (Task ledger/plan, Visited pages, Queue, Subgoals, Clicks, DOM facts, Chat)
        - Tier 3: Dynamic / every step changing (Tactical hint, Recent execution trajectory)
        """
        lines: list[str] = [
            f"Primary Goal: {self.primary_goal}",
            "Language Mandate: Answer and reasoning must strictly match the language of the Primary Goal / user task, regardless of internal context notes.",
        ]
        if self.constraints:
            lines.append(f"Active Constraints: {'; '.join(self.constraints)}")
        if self.home_site:
            lines.append(
                f"Task Website: {self.home_site}. Work on this website with its own search, menus and links; following links "
                "shown on pages is fine, but do not GOTO_URL guessed addresses of other websites unless the Primary Goal "
                "names them. If the task cannot be done here, use COMPLETE_TASK and explain why."
            )
        if not self.user_present:
            lines.append(
                "User Availability: the user is NOT available during this run. Do not ASK_USER for preferences, "
                "ranges or choices: pick a sensible default and state it in summary_for_user. ASK_USER only for "
                "data that must not be invented (login credentials, personal data, payment details)."
            )

        # 1. Explicit User Facts (Provenance: human, static throughout task)
        if self.user_facts:
            lines.append("User Facts (verified from human user):")
            for k, v in list(self.user_facts.items())[-10:]:
                lines.append(f"  - {k}: {v}")

        # 2. Slow-changing task state
        if self.plan and self.plan.needs_plan:
            lines.append("Structured Task Ledger:")
            lines.append(self.plan.format_for_s2())
            if self.working_memory and self.working_memory != self.plan.format_for_s2():
                lines.append(f"Cumulative Working Memory: {self.working_memory}")
        elif self.working_memory:
            lines.append(f"Cumulative Working Memory: {self.working_memory}")

        if self.blocked_domains:
            lines.append(
                f"Blocked Domains Showing No Content (DO NOT NAVIGATE OR CLICK LINKS TO THESE): {sorted(self.blocked_domains)}"
            )
        if self.scroll_warning:
            lines.append(f"Scroll Warning: {self.scroll_warning}")
        if self.domain_guidance:
            for dom, guide in sorted(self.domain_guidance.items()):
                lines.append(f"Domain Guidance ({dom}): {guide}")

        if self.rejected_links:
            lines.append(f"Already Visited & Rejected Links (DO NOT CLICK THESE AGAIN): {self.rejected_links[-10:]}")
        if self.ineffective_clicks:
            ineff_items = [
                f"{lbl} ({role})" if role else lbl
                for _, lbl, role in self.ineffective_clicks[-10:]
            ]
            lines.append(
                f"Ineffective Clicks on Current Page (had no effect; DO NOT CLICK THESE AGAIN): {ineff_items}"
            )

        if self.visited_urls:
            unique_urls: list[str] = []
            for u in self.visited_urls:
                if u and u not in unique_urls:
                    unique_urls.append(u)
            recent_urls = unique_urls[-40:]
            if recent_urls:
                lines.append("Visited Pages (already opened; re-open only if needed):")
                for u in recent_urls:
                    lines.append(f"  - {u[:150]}")

        self.prune_visit_queue()
        if self.visit_queue:
            lines.append(
                "Queued URLs (collected earlier; open with GOTO_URL directly, no need to return to the source page):"
            )
            for u in self.visit_queue[:MAX_VISIT_QUEUE_URLS]:
                lines.append(f"  - {u[:MAX_VISIT_QUEUE_URL_CHARS]}")

        lines.append(f"Active Subgoal: {self.active_subgoal or self.primary_goal}")
        if self.subgoal_queue:
            lines.append(f"Remaining Subgoals: {self.subgoal_queue}")
        if self.completed_subgoals:
            lines.append(f"Completed Subgoals: {self.completed_subgoals[-5:]}")

        # Verified DOM Facts (Provenance: page state, sanitized)
        if self.verified_dom_facts:
            lines.append("Verified DOM Observations:")
            for f in self.verified_dom_facts[-8:]:
                k_clean = sanitize_untrusted_text(f.key, 80)
                v_clean = sanitize_untrusted_text(f.value, 200)
                lines.append(f"  - [step {f.step}] {k_clean}: {v_clean}")

        # Bounded Chat Window
        if self.older_chat_summary:
            lines.append(f"Earlier Chat Summary: {self.older_chat_summary}")
        recent_chat = self.chat_log[-self.max_chat_window :]
        if recent_chat:
            lines.append("Recent User-Agent Conversation:")
            for m in recent_chat:
                lines.append(f"  [{m.role} ({m.kind})]: {m.content[:300]}")

        # 3. Dynamic / step-changing context
        if self.tactical_hint:
            lines.append(f"Current Tactical Hint: {self.tactical_hint}")

        # Compact Trajectory Summary (no raw historical DOMs!)
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

        return "\n".join(lines)

    def snapshot_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the context for the frontend inspector."""
        self.prune_visit_queue()
        return {
            "primary_goal": self.primary_goal,
            "active_subgoal": self.active_subgoal or self.primary_goal,
            "allow_arbitrary_choices": self.allow_arbitrary_choices,
            "constraints": self.constraints,
            "tactical_hint": self.tactical_hint,
            "working_memory": self.working_memory,
            "plan": self.plan.model_dump() if self.plan else None,
            "visited_urls": self.visited_urls[-15:],
            "rejected_links": self.rejected_links[-15:],
            "visit_queue": list(self.visit_queue[:MAX_VISIT_QUEUE_URLS]),
            "visit_queue_rejected_count": self.visit_queue_rejected_count,
            "ineffective_clicks": [list(item) for item in self.ineffective_clicks[-15:]],
            "subgoal_queue": self.subgoal_queue,
            "completed_subgoals": self.completed_subgoals,
            "user_facts": self.user_facts,
            "verified_dom_facts": [f.model_dump() for f in self.verified_dom_facts],
            "agent_hypotheses": self.agent_hypotheses,
            "chat_log": [m.model_dump() for m in self.chat_log],
            "approved_actions": sorted(self.approved_actions),
            "blocked_domains": sorted(self.blocked_domains),
            "domain_guidance": dict(self.domain_guidance),
            "scroll_warning": self.scroll_warning,
        }
