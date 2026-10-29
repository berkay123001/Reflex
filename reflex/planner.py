"""Task Ledger, Candidate Table, and Progress Verification for Reflex (R1-R4).

Inspired by Plan-and-Act (Erdogan et al. 2025), StructuredAgent (2026),
BrowseMaster (2025), and Magentic-One (Microsoft).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


class SubgoalResult(BaseModel):
    """A distilled factual finding associated with a specific subgoal and source."""

    fact: str
    source_url: str = ""
    step: int = 0


class Subgoal(BaseModel):
    """An individual actionable milestone within a task ledger."""

    id: str
    text: str
    done_when: str = ""
    status: str = "pending"  # "pending" | "active" | "completed" | "failed"
    results: list[SubgoalResult] = Field(default_factory=list)


class Candidate(BaseModel):
    """An entity, product, or option being tracked and compared across sources."""

    item: str
    source_url: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_urls: list[str] = Field(default_factory=list)


class Clarify(BaseModel):
    """A single clarification question generated when task scope or depth is ambiguous."""

    question: str
    options: list[str] = Field(default_factory=list)
    default: str = ""


class Budget(BaseModel):
    """Operational budget constraints for multi-source task execution."""

    sources: int = 1
    max_steps: int = 30


class ProgressCheckResult(BaseModel):
    """Milestone evaluation result for progress ledger updates."""

    subgoal_done: bool = False
    next_subgoal: str | None = None
    stuck: bool = False
    new_notes: str | None = None
    extracted_results: list[dict[str, Any]] = Field(default_factory=list)
    extracted_candidates: list[dict[str, Any]] = Field(default_factory=list)


class DoneVerificationResult(BaseModel):
    """Multi-criteria verification result determining whether a task is truly finished."""

    is_done: bool = False
    rejection_reason: str | None = None
    missing_subgoal_id: str | None = None
    updated_active_subgoal: str | None = None
    final_answer: str | None = None
    sources_used: list[str] = Field(default_factory=list)


def _extract_domain(url: str) -> str:
    """Extract normalized domain from a URL string, stripping 'www.' and lowercasing."""
    if not url:
        return ""
    try:
        clean = str(url).strip().lower()
        if clean.startswith(("about:", "javascript:", "chrome:", "data:", "blob:", "file:")):
            return ""
        if "://" not in clean and not clean.startswith("//"):
            clean = f"http://{clean}"
        parsed = urlsplit(clean)
        hostname = parsed.hostname or ""
        if not hostname and parsed.netloc:
            hostname = parsed.netloc.split(":")[0]
        return hostname.removeprefix("www.").strip()
    except (ValueError, TypeError, AttributeError):
        clean = str(url).strip().lower()
        if clean.startswith(("about:", "javascript:", "chrome:", "data:", "blob:", "file:")):
            return ""
        clean = clean.split("/")[0].removeprefix("www.")
        return clean.split(":")[0].strip()


class TaskLedger(BaseModel):
    """The central structured task ledger serving as additive persistent memory."""

    needs_plan: bool = False
    clarify: Clarify | None = None
    subgoals: list[Subgoal] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    finish_criteria: str = ""
    budget: Budget = Field(default_factory=Budget)

    def get_active_subgoal(self) -> Subgoal | None:
        """Return the currently active subgoal, or activate and return the first pending one."""
        for sg in self.subgoals:
            if sg.status == "active":
                return sg
        for sg in self.subgoals:
            if sg.status == "pending":
                sg.status = "active"
                return sg
        return None

    def advance_subgoal(self) -> Subgoal | None:
        """Mark active subgoal as completed and activate the next pending subgoal."""
        active = next((s for s in self.subgoals if s.status == "active"), None)
        if active:
            active.status = "completed"
        for sg in self.subgoals:
            if sg.status == "pending":
                sg.status = "active"
                return sg
        return None

    def add_result(
        self,
        fact: str,
        source_url: str = "",
        step: int = 0,
        subgoal_id: str | None = None,
    ) -> SubgoalResult:
        """Add a distilled result finding to the specified or active subgoal."""
        target_sg: Subgoal | None = None
        if subgoal_id:
            target_sg = next((s for s in self.subgoals if s.id == subgoal_id), None)
        if target_sg is None:
            target_sg = self.get_active_subgoal()
        if target_sg is None and self.subgoals:
            target_sg = self.subgoals[-1]

        res = SubgoalResult(
            fact=str(fact).strip(),
            source_url=str(source_url).strip(),
            step=step,
        )
        if target_sg is not None:
            # System 2 re-reports the same verified fact on later checks; keep the first record only.
            existing = next((r for r in target_sg.results if r.fact.casefold() == res.fact.casefold()), None)
            if existing is not None:
                return existing
            target_sg.results.append(res)
        return res

    def add_candidate(
        self,
        item: str,
        source_url: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> Candidate:
        """Add or update an item in the comparison candidates table with deduplication."""
        attr_dict = dict(attributes) if isinstance(attributes, dict) else {}
        clean_item = " ".join(item.strip().split())
        norm_item = clean_item.lower()
        clean_url = source_url.strip()

        for cand in self.candidates:
            if " ".join(cand.item.strip().split()).lower() == norm_item:
                if clean_url:
                    if not cand.source_url:
                        cand.source_url = clean_url
                    if clean_url not in cand.source_urls:
                        cand.source_urls.append(clean_url)
                cand.attributes.update(attr_dict)
                return cand

        initial_sources = [clean_url] if clean_url else []
        cand = Candidate(
            item=clean_item,
            source_url=clean_url,
            attributes=attr_dict,
            source_urls=initial_sources,
        )
        self.candidates.append(cand)
        return cand

    def total_results_chars(self) -> int:
        """Return the total character count across all facts in all subgoals."""
        return sum(len(r.fact) for sg in self.subgoals for r in sg.results)

    def summarize_completed_results(
        self,
        threshold: int = 2500,
        summarizer: Callable[[list[SubgoalResult]], list[SubgoalResult]] | None = None,
    ) -> bool:
        """Summarize completed subgoals' results if cumulative character length exceeds threshold.

        Strictly preserves all unique source_url references.
        """
        if self.total_results_chars() <= threshold:
            return False

        completed_sgs = [
            sg for sg in self.subgoals
            if sg.status == "completed" and len(sg.results) > 0
        ]
        if not completed_sgs:
            return False

        if summarizer is not None:
            for sg in completed_sgs:
                sg.results = summarizer(sg.results)
        else:
            for sg in completed_sgs:
                grouped: dict[str, list[str]] = {}
                min_step: dict[str, int] = {}
                for r in sg.results:
                    url = r.source_url.strip()
                    grouped.setdefault(url, []).append(r.fact.strip())
                    if url not in min_step or r.step < min_step[url]:
                        min_step[url] = r.step

                condensed: list[SubgoalResult] = []
                for url, facts in grouped.items():
                    seen: set[str] = set()
                    uniq_facts: list[str] = []
                    for f in facts:
                        if f and f not in seen:
                            seen.add(f)
                            uniq_facts.append(f)
                    merged = "; ".join(uniq_facts)
                    if len(merged) > 300:
                        merged = merged[:297] + "..."
                    condensed.append(
                        SubgoalResult(
                            fact=merged,
                            source_url=url,
                            step=min_step.get(url, 0),
                        )
                    )
                sg.results = condensed
        return True

    def format_for_s2(self) -> str:
        """Format the task ledger, subgoals, results, and candidate table for System 2."""
        lines: list[str] = [
            f"Plan Status: {'Active' if self.needs_plan else 'No Plan'}",
            f"Finish Criteria: {self.finish_criteria or 'Not specified'}",
            f"Budget: {self.budget.sources} sources, {self.budget.max_steps} max steps",
            "Subgoals & Extracted Results:",
        ]
        for sg in self.subgoals:
            status_tag = f"[{sg.status.upper()}]"
            done_when = f" (Done when: {sg.done_when})" if sg.done_when else ""
            lines.append(f"  {status_tag} {sg.id}: {sg.text}{done_when}")
            for r in sg.results:
                src = f" (Source: {r.source_url})" if r.source_url else ""
                lines.append(f"    - [Step {r.step}] {r.fact}{src}")
        if self.candidates:
            lines.append("Candidates Table:")
            for c in self.candidates:
                attr_str = ", ".join(f"{k}: {v}" for k, v in c.attributes.items())
                src = f" (Source: {c.source_url})" if c.source_url else ""
                lines.append(f"  - {c.item}: {attr_str}{src}")
        return "\n".join(lines)

    def verify_done(
        self,
        visited_urls: list[str] | None = None,
    ) -> tuple[bool, str, Subgoal | None]:
        """Verify whether task completion criteria are genuinely satisfied.

        Returns (is_done, reason_string, missing_subgoal_or_none).
        """
        # Bypass for simple single-site tasks without subgoals
        if not self.needs_plan and not self.subgoals:
            return True, "No plan required (simple task)", None

        # 1. Subgoal completion & non-empty results check. In comparison tasks the findings go to the
        # candidate table instead of subgoal results, so a filled table counts as evidence.
        for sg in self.subgoals:
            if sg.status == "completed" and not sg.results and not self.candidates:
                return False, f"Subgoal '{sg.text}' has no verified results", sg
        for sg in self.subgoals:
            if sg.status != "completed":
                return False, f"Subgoal '{sg.text}' is not completed", sg

        # 2. Candidate table check for comparison tasks
        if len(self.candidates) > 0:
            cand_sources = {
                _extract_domain(u)
                for c in self.candidates
                for u in (c.source_urls or ([c.source_url] if c.source_url else []))
                if _extract_domain(u)
            }
            if len(self.candidates) < self.budget.sources and len(cand_sources) < self.budget.sources:
                missing_sg = self.subgoals[-1] if self.subgoals else None
                return (
                    False,
                    f"Candidate table has {len(self.candidates)} items, expected at least {self.budget.sources}",
                    missing_sg,
                )

        # 3. Distinct sources budget check
        sources: set[str] = set()
        for sg in self.subgoals:
            for r in sg.results:
                d = _extract_domain(r.source_url)
                if d:
                    sources.add(d)
        for c in self.candidates:
            urls_to_check = c.source_urls if c.source_urls else ([c.source_url] if c.source_url else [])
            for u in urls_to_check:
                d = _extract_domain(u)
                if d:
                    sources.add(d)
        if visited_urls:
            for u in visited_urls:
                d = _extract_domain(u)
                if d:
                    sources.add(d)

        if len(sources) < self.budget.sources:
            missing_sg = self.subgoals[-1] if self.subgoals else None
            return (
                False,
                f"Found {len(sources)} sources, budget requires {self.budget.sources}",
                missing_sg,
            )

        return True, "All completion criteria satisfied", None

    def synthesize_final_answer(self, primary_goal: str) -> str:
        """Synthesize final response citing facts, candidate comparisons, and sources."""
        lines: list[str] = [f"### Görev Sonucu: {primary_goal}\n"]
        if self.subgoals:
            lines.append("#### Doğrulanan Bulgular:")
            for sg in self.subgoals:
                lines.append(f"- **{sg.text}**:")
                if sg.results:
                    for r in sg.results:
                        src_note = f" *(Kaynak: {r.source_url})*" if r.source_url else ""
                        lines.append(f"  • {r.fact}{src_note}")
                else:
                    lines.append("  • (Bulgu kaydedilmedi)")
        if self.candidates:
            lines.append("\n#### Karşılaştırma ve Aday Tablosu:")
            for c in self.candidates:
                attrs = (
                    ", ".join(f"{k}: {v}" for k, v in c.attributes.items())
                    if c.attributes
                    else "Detay yok"
                )
                src_note = f" *(Kaynak: {c.source_url})*" if c.source_url else ""
                lines.append(f"  - **{c.item}**: {attrs}{src_note}")
        return "\n".join(lines)


def parse_task_ledger(data: dict[str, Any] | str) -> TaskLedger:
    """Parse a task ledger from JSON string or dictionary with resilient fallbacks."""
    try:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                return TaskLedger(needs_plan=False)

        if not isinstance(data, dict):
            return TaskLedger(needs_plan=False)

        needs_plan = bool(data.get("needs_plan", False))
        if not needs_plan:
            return TaskLedger(needs_plan=False)

        clarify: Clarify | None = None
        clarify_raw = data.get("clarify")
        if isinstance(clarify_raw, dict) and clarify_raw.get("question"):
            raw_opts = clarify_raw.get("options")
            opts = (
                [str(o).strip() for o in raw_opts if str(o).strip()]
                if isinstance(raw_opts, (list, tuple))
                else []
            )
            clarify = Clarify(
                question=str(clarify_raw["question"]).strip(),
                options=opts,
                default=str(clarify_raw.get("default", "")).strip(),
            )

        subgoals: list[Subgoal] = []
        raw_subgoals = data.get("subgoals") or []
        if isinstance(raw_subgoals, (list, tuple)):
            for idx, sg in enumerate(raw_subgoals):
                if not isinstance(sg, dict):
                    continue
                raw_results = sg.get("results")
                results = []
                if isinstance(raw_results, (list, tuple)):
                    for r in raw_results:
                        if isinstance(r, dict) and r.get("fact") is not None and str(r.get("fact")).strip():
                            try:
                                step_val = int(r.get("step", 0) or 0)
                            except (ValueError, TypeError):
                                step_val = 0
                            results.append(
                                SubgoalResult(
                                    fact=str(r.get("fact", "")).strip(),
                                    source_url=str(r.get("source_url", "")).strip(),
                                    step=step_val,
                                )
                            )
                subgoals.append(
                    Subgoal(
                        id=str(sg.get("id") or f"sg_{idx+1}"),
                        text=str(sg.get("text") or "").strip(),
                        done_when=str(sg.get("done_when") or "").strip(),
                        status=str(sg.get("status") or ("active" if idx == 0 else "pending")),
                        results=results,
                    )
                )

        if not subgoals:
            subgoals.append(
                Subgoal(
                    id="sg_1",
                    text=str(data.get("finish_criteria") or "Complete primary goal").strip(),
                    done_when=str(data.get("finish_criteria") or "").strip(),
                    status="active",
                )
            )

        if subgoals and not any(s.status == "active" for s in subgoals):
            first_pending = next((s for s in subgoals if s.status == "pending"), None)
            if first_pending is not None:
                first_pending.status = "active"

        candidates: list[Candidate] = []
        raw_candidates = data.get("candidates") or []
        if isinstance(raw_candidates, (list, tuple)):
            for c in raw_candidates:
                if isinstance(c, dict) and c.get("item"):
                    raw_attrs = c.get("attributes")
                    attr_dict = dict(raw_attrs) if isinstance(raw_attrs, dict) else {}
                    c_item = " ".join(str(c.get("item", "")).strip().split())
                    c_url = str(c.get("source_url", "")).strip()
                    raw_urls = c.get("source_urls")
                    c_urls = (
                        [str(u).strip() for u in raw_urls if str(u).strip()]
                        if isinstance(raw_urls, (list, tuple))
                        else []
                    )
                    if c_url and c_url not in c_urls:
                        c_urls.insert(0, c_url)
                    candidates.append(
                        Candidate(
                            item=c_item,
                            source_url=c_url,
                            attributes=attr_dict,
                            source_urls=c_urls,
                        )
                    )

        budget_raw = data.get("budget") or {}
        sources = 1
        max_steps = 30
        if isinstance(budget_raw, dict):
            try:
                sources = max(1, int(budget_raw.get("sources", 1) or 1))
            except (ValueError, TypeError):
                sources = 1
            try:
                max_steps = max(1, int(budget_raw.get("max_steps", 30) or 30))
            except (ValueError, TypeError):
                max_steps = 30
        budget = Budget(sources=sources, max_steps=max_steps)

        finish_criteria = str(data.get("finish_criteria") or "").strip()

        return TaskLedger(
            needs_plan=True,
            clarify=clarify,
            subgoals=subgoals,
            candidates=candidates,
            finish_criteria=finish_criteria,
            budget=budget,
        )
    except (ValueError, TypeError, KeyError, AttributeError, IndexError):
        return TaskLedger(needs_plan=False)
