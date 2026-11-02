"""Tests for R2: Empty page domain blocking, 12-step non-progress budget, fruitless scroll limiter, and 404 guidance.

Specification Source: artifacts/araclar/gorev_g12.md §R2 (lines 70-82, 150-151) and SCOPE.md (Features 17-22).
All tests execute completely offline using mock browser/CDP and mock supervisor in < 1s.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexSession, _is_404_error_page, _is_empty_page
from reflex.config import ReflexConfig, run_mode_flags
from reflex.context import DualTierContext, extract_domain
from reflex.supervisor import SupervisorDecision
from reflex.verifier import verify_step


def _make_mock_session(url: str = "https://example.com/start", goal: str = "Search items") -> ReflexSession:
    sess = ReflexSession.__new__(ReflexSession)
    sess.config = ReflexConfig(**run_mode_flags("s2_only"))
    sess.url = url
    sess.session_id = "test-session"
    sess.trace_path = None
    sess.trace_write_errors = 0
    sess._terminal_trace_logged = False
    sess.ctx = DualTierContext(primary_goal=goal, active_subgoal=goal)
    sess.site_memory = MagicMock()
    sess.site_memory.record_rule = MagicMock()
    sess.status = "ready"
    sess.started_at = time.perf_counter()
    sess.elapsed_ms = 0
    sess.history = []
    sess.decisions = []
    sess.escalations = []
    sess.text_calls = []
    sess.interpret_calls = []
    sess.pending_fact_key = None
    sess.last_asked_question = None
    sess.pending_confirmation = None
    sess.pending_escalation_reason = None
    sess._noop_escalation_counts = {}
    sess._done_rejected_url = None
    sess._ineffective_url = ""
    sess._ineffective_fp = ""
    sess.last_error = None
    sess.s1_errors = 0
    sess._s1_steps_since_s2 = 0
    sess._stuck_counter = 0
    sess._plan_done_rejections = 0
    sess.unsettled_count = 0
    sess.active_package = None
    sess.packages_started = 0
    sess.packages_completed = 0
    sess.package_steps_executed = 0
    sess.package_steps_skipped = 0
    sess.package_offscreen_scrolls = 0
    sess.package_aborts = Counter()
    sess.s1_pkg_step1_agreed = 0
    sess.s1_pkg_step1_total = 0
    sess.s1_pkg_stepk_agreed = 0
    sess.s1_pkg_stepk_total = 0
    sess._domain_consecutive_empty = {}
    sess._no_progress_steps = 0
    sess._stuck_finish_count = 0
    sess._fruitless_scroll_count = 0
    sess._last_scroll_direction = None
    sess._locked_scroll_direction = None
    sess._404_domains = set()
    sess.page = {"url": url, "text": "start page", "actions": []}
    sess.browser = MagicMock()
    sess.browser.observe.return_value = sess.page
    sess.browser.act.return_value = {"ok": True}
    sess.browser.fresh.return_value = True
    sess.snapshot = lambda: {"status": sess.status, "history_len": len(sess.history)}
    sess._observe_settled = lambda: sess.page
    return sess


# ==============================================================================
# Test Suite 1: Empty Page Domain Blocking Tests
# ==============================================================================


def test_empty_page_definition_and_detection() -> None:
    """Verifies empty page definition: text is empty/whitespace and 0 non-utility interactive actions."""
    # Case A: completely empty
    page_a = {"url": "https://a.com", "text": "", "actions": []}
    assert _is_empty_page(page_a) is True

    # Case B: whitespace only and utility scroll action
    page_b = {"url": "https://a.com", "text": "   \n\t  ", "actions": [{"kind": "scroll", "id": "scroll_down"}]}
    assert _is_empty_page(page_b) is True

    # Case C: non-empty visible text
    page_c = {"url": "https://a.com", "text": "Hello world", "actions": []}
    assert _is_empty_page(page_c) is False

    # Case D: empty text but has interactive button
    page_d = {"url": "https://a.com", "text": "", "actions": [{"kind": "click", "role": "button", "label": "Search"}]}
    assert _is_empty_page(page_d) is False

    # Case E: empty text but has page_controls
    page_e = {"url": "https://a.com", "text": "", "actions": [], "page_controls": [{"node": 1}]}
    assert _is_empty_page(page_e) is False

    # Edge cases: None or non-dict or missing text
    assert _is_empty_page(None) is False
    assert _is_empty_page({}) is False  # missing text key is not an empty page


def test_three_consecutive_empty_pages_blocks_domain() -> None:
    """3 consecutive empty observations on https://dead-domain.com/p1, /p2, /p3 adds domain to blocked_domains."""
    sess = _make_mock_session(url="https://dead-domain.com/p1")
    sess._observe_settled = ReflexSession._observe_settled.__get__(sess, ReflexSession)

    p1 = {"url": "https://dead-domain.com/p1", "text": "", "actions": []}
    p2 = {"url": "https://dead-domain.com/p2", "text": "", "actions": []}
    p3 = {"url": "https://dead-domain.com/p3", "text": "", "actions": []}

    sess.browser.observe.side_effect = [p1, p2, p3]

    # Observation 1
    obs1 = sess._observe_settled()
    assert obs1.get("is_empty") is True
    assert "dead-domain.com" not in sess.ctx.blocked_domains
    assert sess._domain_consecutive_empty["dead-domain.com"] == 1

    # Observation 2
    obs2 = sess._observe_settled()
    assert obs2.get("is_empty") is True
    assert "dead-domain.com" not in sess.ctx.blocked_domains
    assert sess._domain_consecutive_empty["dead-domain.com"] == 2

    # Observation 3 -> triggers domain block and pending escalation
    obs3 = sess._observe_settled()
    assert obs3.get("is_empty") is True
    assert "dead-domain.com" in sess.ctx.blocked_domains
    assert sess._domain_consecutive_empty["dead-domain.com"] == 3
    assert any("dead-domain.com" in rej for rej in sess.ctx.rejected_links)
    assert sess.pending_escalation_reason is not None
    assert sess.pending_escalation_reason[0] == "domain_blocked"

    s2_block = sess.ctx.build_supervisor_context_block([])
    assert "Blocked Domains" in s2_block
    assert "dead-domain.com" in s2_block


def test_blocked_domain_prunes_visit_queue_and_rejects_candidates() -> None:
    """Candidate links on dead-domain.com are pruned from visit_queue and newly enqueued links are rejected."""
    ctx = DualTierContext(primary_goal="Explore links")
    ctx.visit_queue = ["https://dead-domain.com/q1", "https://alive-domain.com/q2"]

    ctx.block_domain("dead-domain.com")

    # Queue should be pruned automatically upon block_domain
    assert "https://dead-domain.com/q1" not in ctx.visit_queue
    assert "https://alive-domain.com/q2" in ctx.visit_queue

    # Candidate enqueue rejected
    accepted = ctx.enqueue_urls(["https://dead-domain.com/q3"], ["https://dead-domain.com/q3"])
    assert accepted == []
    assert ctx.visit_queue_rejected_count == 1
    assert "https://dead-domain.com/q3" not in ctx.visit_queue


def test_subsequent_goto_url_to_blocked_domain_prevented() -> None:
    """Subsequent GOTO_URL to dead-domain.com is prevented and rejected without browser navigation."""
    sess = _make_mock_session()
    sess.ctx.block_domain("dead-domain.com")

    sup = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="GOTO_URL",
        target="https://dead-domain.com/article",
        reasoning="Try another page",
    )
    resolved = sess._resolve_supervisor_action(sup, {}, {})
    assert resolved is None
    assert any("dead-domain.com" in rej for rej in sess.ctx.rejected_links)

    # Click rejection check
    assert sess._is_rejected_target({"kind": "click", "href": "https://dead-domain.com/article"}, "https://example.com") is True
    assert sess._is_rejected_target({"kind": "click", "href": "https://alive-domain.com/article"}, "https://example.com") is False


def test_non_empty_page_resets_empty_page_counter() -> None:
    """A non-empty page observation on the same domain resets consecutive empty counter, avoiding blocking."""
    sess = _make_mock_session(url="https://slow.com/p1")
    sess._observe_settled = ReflexSession._observe_settled.__get__(sess, ReflexSession)

    p1 = {"url": "https://slow.com/p1", "text": "", "actions": []}
    p2 = {"url": "https://slow.com/p2", "text": "", "actions": []}
    p3 = {"url": "https://slow.com/p3", "text": "Loaded catalog contents here", "actions": []}
    p4 = {"url": "https://slow.com/p4", "text": "", "actions": []}

    sess.browser.observe.side_effect = [p1, p2, p3, p4]

    sess._observe_settled()  # empty 1
    sess._observe_settled()  # empty 2
    assert sess._domain_consecutive_empty["slow.com"] == 2

    sess._observe_settled()  # non-empty -> resets to 0
    assert sess._domain_consecutive_empty["slow.com"] == 0
    assert "slow.com" not in sess.ctx.blocked_domains

    sess._observe_settled()  # empty 1 again
    assert sess._domain_consecutive_empty["slow.com"] == 1
    assert "slow.com" not in sess.ctx.blocked_domains


def test_interleaved_domains_do_not_falsely_block() -> None:
    """Empty observations across different domains do not accumulate into a false positive block."""
    sess = _make_mock_session(url="https://dom-a.com/p1")
    sess._observe_settled = ReflexSession._observe_settled.__get__(sess, ReflexSession)

    p1 = {"url": "https://dom-a.com/p1", "text": "", "actions": []}
    p2 = {"url": "https://dom-b.com/p1", "text": "", "actions": []}
    p3 = {"url": "https://dom-a.com/p2", "text": "", "actions": []}

    sess.browser.observe.side_effect = [p1, p2, p3]

    sess._observe_settled()
    sess._observe_settled()
    sess._observe_settled()

    assert sess._domain_consecutive_empty["dom-a.com"] == 1
    assert sess._domain_consecutive_empty["dom-b.com"] == 0
    assert "dom-a.com" not in sess.ctx.blocked_domains
    assert "dom-b.com" not in sess.ctx.blocked_domains


# ==============================================================================
# Test Suite 2: 12-Step Non-Progress Budget Tests
# ==============================================================================


def test_twelve_consecutive_no_progress_steps_triggers_stuck_finish_escalation() -> None:
    """12 consecutive steps with no new URL, no verifier progress, and no notebook entry triggers stuck_finish."""
    sess = _make_mock_session()
    static_page = {
        "url": "https://example.com/stuck",
        "fingerprint": "fp1",
        "text": "Static constant text",
        "actions": [{"id": "click_1", "kind": "click", "node": 1, "label": "Submit"}],
    }
    sess.page = static_page
    sess._observe_settled = lambda: static_page

    for i in range(12):
        sess._execute_and_verify(
            action={"kind": "click", "id": "click_1", "node": 1, "label": "Submit"},
            actor="S1",
            operation="CLICK",
            target="1",
            confidence=0.8,
            target_confidence=None,
            joint_confidence=0.8,
            target_margin=0.8,
            latency_ms=10,
            model_name="mock-model",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert sess._no_progress_steps == 12
    assert sess._stuck_finish_count == 1
    assert sess.pending_escalation_reason is not None
    assert sess.pending_escalation_reason[0] == "stuck_finish"
    assert "12 consecutive steps without progress" in sess.pending_escalation_reason[1]


def test_new_url_resets_non_progress_budget() -> None:
    """Reaching a new URL resets the 12-step non-progress budget counter to 0."""
    sess = _make_mock_session()
    sess._no_progress_steps = 11

    p_before = {"url": "https://example.com/start", "fingerprint": "fp1", "text": "start", "actions": []}
    p_after = {"url": "https://example.com/new-page", "fingerprint": "fp2", "text": "new page", "actions": []}
    sess.page = p_before
    sess._observe_settled = lambda: p_after

    sess._execute_and_verify(
        action={"kind": "goto_url", "id": "goto_url", "url": "https://example.com/new-page", "label": "Navigate"},
        actor="S1",
        operation="GOTO_URL",
        target=None,
        confidence=1.0,
        target_confidence=None,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert sess._no_progress_steps == 0
    assert sess.pending_escalation_reason is None


def test_verifier_effective_progress_resets_non_progress_budget() -> None:
    """Effective progress declared by verifier resets the 12-step non-progress budget counter to 0."""
    sess = _make_mock_session()
    sess._no_progress_steps = 11

    p_before = {"url": "https://example.com/form", "fingerprint": "fp1", "text": "Form empty", "actions": [{"id": "t1", "kind": "fill", "node": 1}]}
    p_after = {"url": "https://example.com/form", "fingerprint": "fp2", "text": "Form submitted successfully", "actions": []}
    sess.page = p_before
    sess._observe_settled = lambda: p_after

    sess._execute_and_verify(
        action={"kind": "fill", "id": "t1", "node": 1, "text": "Alice", "label": "Name"},
        actor="S1",
        operation="TYPE_TEXT",
        target="1",
        confidence=0.9,
        target_confidence=None,
        joint_confidence=0.9,
        target_margin=0.9,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text="Alice",
        escalation_note=None,
    )

    assert sess._no_progress_steps == 0
    assert sess.pending_escalation_reason is None


def test_notebook_finding_resets_non_progress_budget() -> None:
    """Adding a new finding to notebook/plan or verified facts resets non-progress budget counter to 0."""
    sess = _make_mock_session()
    sess._no_progress_steps = 11

    static_page = {"url": "https://example.com/page", "fingerprint": "fp1", "text": "Same text", "actions": []}
    sess.page = static_page
    sess._observe_settled = lambda: static_page

    def act_with_notebook_finding(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sess.ctx.verified_dom_facts.append({"fact": "Price is 99 USD"})
        return {"ok": True}

    sess.browser.act.side_effect = act_with_notebook_finding

    sess._execute_and_verify(
        action={"kind": "wait", "id": "wait", "label": "Wait"},
        actor="S1",
        operation="WAIT",
        target=None,
        confidence=0.9,
        target_confidence=None,
        joint_confidence=0.9,
        target_margin=0.9,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert sess._no_progress_steps == 0


def test_first_stuck_finish_allows_s2_recovery_attempt() -> None:
    """First stuck_finish escalation allows S2 recovery attempt without terminating the session."""
    sess = _make_mock_session()
    sess._no_progress_steps = 12

    recovery_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="SCROLL_DOWN",
        reasoning="Try scrolling down to see if hidden elements appear.",
        latency_ms=50,
    )

    with patch("reflex.agent.resolve_escalation", return_value=recovery_decision):
        sess._handle_escalation(
            trigger_type="stuck_finish",
            reason="12 steps without progress",
            s1_decision=None,
            step_t0=time.perf_counter(),
            s1_ms=0,
            pre_obs_ms=0,
        )

    assert sess.status == "ready"
    assert sess._stuck_finish_count == 1


def test_second_consecutive_stuck_finish_terminates_session_honestly() -> None:
    """Second consecutive stuck_finish without intervening progress causes honest session termination with status='done'."""
    sess = _make_mock_session()
    sess._stuck_finish_count = 1
    sess._no_progress_steps = 24

    sess._handle_escalation(
        trigger_type="stuck_finish",
        reason="Second consecutive stuck_finish trigger",
        s1_decision=None,
        step_t0=time.perf_counter(),
        s1_ms=0,
        pre_obs_ms=0,
    )

    assert sess.status == "done"
    assert sess.last_error is not None
    assert "İlerleme sağlanamadı" in sess.last_error or "progress" in sess.last_error.lower()


def test_intervening_progress_clears_consecutive_stuck_finish_count() -> None:
    """Progress after first stuck_finish clears consecutive stuck count, giving future stuck situations a fresh recovery attempt."""
    sess = _make_mock_session()
    sess._stuck_finish_count = 1
    sess._no_progress_steps = 12

    # Step executes and makes progress via new URL
    p_before = {"url": "https://example.com/p1", "fingerprint": "fp1", "text": "Page 1", "actions": []}
    p_after = {"url": "https://example.com/p2", "fingerprint": "fp2", "text": "Page 2", "actions": []}
    sess.page = p_before
    sess._observe_settled = lambda: p_after

    sess._execute_and_verify(
        action={"kind": "goto_url", "id": "goto_url", "url": "https://example.com/p2", "label": "Navigate"},
        actor="S1",
        operation="GOTO_URL",
        target=None,
        confidence=1.0,
        target_confidence=None,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert sess._no_progress_steps == 0
    assert sess._stuck_finish_count == 0


# ==============================================================================
# Test Suite 3: Fruitless Scroll & Verifier Tests
# ==============================================================================


def test_verifier_scroll_without_new_content_returns_no_effective_progress() -> None:
    """Verifier requires new text or elements to declare effective_progress=True on scroll; identical content returns False."""
    before_page = {
        "url": "https://a.com",
        "text": "Same static content",
        "actions": [{"id": "a1", "label": "Button 1", "node": 1}],
        "scroll": {"y": 0},
    }
    after_page = {
        "url": "https://a.com",
        "text": "Same static content",
        "actions": [{"id": "a1", "label": "Button 1", "node": 1}],
        "scroll": {"y": 500},
    }
    action = {"kind": "scroll", "label": "Scroll down", "delta": 500}

    res = verify_step(action=action, before_page=before_page, after_page=after_page)

    assert res.details.get("scroll_moved") is True
    assert res.details.get("scroll_revealed_content") is False
    assert res.effective_progress is False
    assert res.element_state_changed is False
    assert res.page_changed is False


def test_verifier_scroll_with_new_text_or_elements_returns_effective_progress() -> None:
    """Verifier returns effective_progress=True when scroll reveals new text or new interactive elements."""
    before_page = {
        "url": "https://a.com",
        "text": "Header and introductory paragraph",
        "actions": [{"id": "a1", "label": "Button 1", "node": 1}],
        "scroll": {"y": 0},
    }

    # Case A: Reveals new text
    after_page_text = {
        "url": "https://a.com",
        "text": "Header and introductory paragraph\nSecond paragraph revealed with extra details",
        "actions": [{"id": "a1", "label": "Button 1", "node": 1}],
        "scroll": {"y": 500},
    }
    res_text = verify_step(
        action={"kind": "scroll", "label": "Scroll down", "delta": 500},
        before_page=before_page,
        after_page=after_page_text,
    )
    assert res_text.details.get("scroll_revealed_content") is True
    assert res_text.effective_progress is True

    # Case B: Reveals new interactive element
    after_page_elem = {
        "url": "https://a.com",
        "text": "Header and introductory paragraph",
        "actions": [
            {"id": "a1", "label": "Button 1", "node": 1},
            {"id": "a2", "label": "Button 2", "node": 2},
        ],
        "scroll": {"y": 500},
    }
    res_elem = verify_step(
        action={"kind": "scroll", "label": "Scroll down", "delta": 500},
        before_page=before_page,
        after_page=after_page_elem,
    )
    assert res_elem.details.get("scroll_revealed_content") is True
    assert res_elem.effective_progress is True


def test_four_consecutive_fruitless_scrolls_warns_s2_and_locks_direction() -> None:
    """4 consecutive fruitless scrolls trigger warning to S2 and lock same-direction scrolling until next S2 decision."""
    sess = _make_mock_session()
    static_scroll_page = {
        "url": "https://example.com/page",
        "fingerprint": "fp-scroll",
        "text": "Unchanging short page text",
        "actions": [{"id": "scroll_down", "kind": "scroll", "node": None}],
        "scroll": {"y": 0},
    }
    sess.page = static_scroll_page
    # Simulates scrolling where y moves but text/elements don't change
    sess._observe_settled = lambda: {
        "url": "https://example.com/page",
        "fingerprint": "fp-scroll",
        "text": "Unchanging short page text",
        "actions": [{"id": "scroll_down", "kind": "scroll", "node": None}],
        "scroll": {"y": 500},
    }

    for _ in range(4):
        sess._execute_and_verify(
            action={"kind": "scroll", "id": "scroll_down", "delta": 560, "label": "Scroll down"},
            actor="S1",
            operation="SCROLL_DOWN",
            target=None,
            confidence=0.9,
            target_confidence=None,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=10,
            model_name="mock-model",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    assert sess._fruitless_scroll_count == 4
    assert sess._locked_scroll_direction == "down"
    assert "4 kaydırmada yeni bir şey çıkmadı" in sess.ctx.scroll_warning
    assert sess.pending_escalation_reason is not None
    assert sess.pending_escalation_reason[0] == "fruitless_scroll"

    # Context block for S2 contains the warning
    s2_block = sess.ctx.build_supervisor_context_block([])
    assert "Scroll Warning:" in s2_block
    assert "4 kaydırmada yeni bir şey çıkmadı" in s2_block


def test_opposite_direction_scroll_resets_fruitless_scroll_counter() -> None:
    """Opposite-direction scroll (scroll up after scroll down) resets the fruitless scroll streak."""
    sess = _make_mock_session()
    static_page = {
        "url": "https://example.com/page",
        "fingerprint": "fp",
        "text": "Text",
        "actions": [],
        "scroll": {"y": 0},
    }
    sess.page = static_page
    sess._observe_settled = lambda: static_page

    # 3 fruitless scrolls down
    for _ in range(3):
        sess._execute_and_verify(
            action={"kind": "scroll", "id": "scroll_down", "delta": 560, "label": "Scroll down"},
            actor="S1",
            operation="SCROLL_DOWN",
            target=None,
            confidence=0.9,
            target_confidence=None,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=10,
            model_name="mock-model",
            usage={},
            preset_text=None,
            escalation_note=None,
        )
    assert sess._fruitless_scroll_count == 3

    # 1 scroll up
    sess._execute_and_verify(
        action={"kind": "scroll", "id": "scroll_up", "delta": -560, "label": "Scroll up"},
        actor="S1",
        operation="SCROLL_UP",
        target=None,
        confidence=0.9,
        target_confidence=None,
        joint_confidence=0.9,
        target_margin=0.9,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    # Fruitless counter resets and records 1 for the new direction, lock is not triggered
    assert sess._fruitless_scroll_count == 1
    assert sess._locked_scroll_direction is None


def test_fruitful_scroll_resets_fruitless_scroll_counter() -> None:
    """Scroll that reveals new content (effective_progress=True) resets fruitless scroll counter."""
    sess = _make_mock_session()
    static_page = {"url": "https://example.com/page", "fingerprint": "fp", "text": "Initial text", "actions": [], "scroll": {"y": 0}}
    sess.page = static_page
    sess._observe_settled = lambda: static_page

    for _ in range(3):
        sess._execute_and_verify(
            action={"kind": "scroll", "id": "scroll_down", "delta": 560, "label": "Scroll down"},
            actor="S1",
            operation="SCROLL_DOWN",
            target=None,
            confidence=0.9,
            target_confidence=None,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=10,
            model_name="mock-model",
            usage={},
            preset_text=None,
            escalation_note=None,
        )
    assert sess._fruitless_scroll_count == 3

    # 4th scroll reveals new elements and text
    revealed_page = {
        "url": "https://example.com/page",
        "fingerprint": "fp2",
        "text": "Initial text\nRevealed new items 1 2 3 4 5",
        "actions": [{"id": "item_1", "kind": "click", "node": 1}],
        "scroll": {"y": 500},
    }
    sess._observe_settled = lambda: revealed_page

    sess._execute_and_verify(
        action={"kind": "scroll", "id": "scroll_down", "delta": 560, "label": "Scroll down"},
        actor="S1",
        operation="SCROLL_DOWN",
        target=None,
        confidence=0.9,
        target_confidence=None,
        joint_confidence=0.9,
        target_margin=0.9,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert sess._fruitless_scroll_count == 0
    assert sess._locked_scroll_direction is None


def test_s2_decision_clears_scroll_direction_lock() -> None:
    """An S2 decision/intervention clears the same-direction scroll lock."""
    sess = _make_mock_session()
    sess._locked_scroll_direction = "down"
    sess.page = {"url": "https://example.com", "actions": [{"id": "scroll_down", "kind": "scroll"}]}

    decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="GOTO_URL",
        target="https://example.com/other",
        reasoning="Try navigating elsewhere",
        latency_ms=50,
    )

    with patch("reflex.agent.resolve_escalation", return_value=decision):
        sess._handle_escalation(
            trigger_type="fruitless_scroll",
            reason="4 fruitless scrolls",
            s1_decision=None,
            step_t0=time.perf_counter(),
            s1_ms=0,
            pre_obs_ms=0,
        )

    assert sess._locked_scroll_direction is None


# ==============================================================================
# Test Suite 4: 404 URL Guessing Guidance Tests
# ==============================================================================


def test_s2_goto_url_404_captured_and_added_to_rejected_links() -> None:
    """S2 GOTO_URL resulting in 404 page is captured and added to rejected_links and domain guidance."""
    dead_url = "https://store.example/products/guessed-sku-999"
    sess = _make_mock_session(url="https://store.example/home")
    sess.ctx.visit_queue = [dead_url]

    p_404 = {
        "url": dead_url,
        "status": 404,
        "title": "404 Not Found",
        "text": "The page you requested could not be found.",
        "actions": [],
    }
    assert _is_404_error_page(p_404) is True

    sess._observe_settled = lambda: p_404

    sess._execute_and_verify(
        action={"kind": "goto_url", "url": dead_url, "label": f"Navigate to {dead_url}"},
        actor="S2",
        operation="GOTO_URL",
        target=None,
        confidence=1.0,
        target_confidence=None,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=10,
        model_name="mock-model",
        usage={},
        preset_text=dead_url,
        escalation_note=None,
    )

    assert any("404 Not Found" in rej and dead_url in rej for rej in sess.ctx.rejected_links)
    assert dead_url not in sess.ctx.visit_queue
    assert "store.example" in sess._404_domains
    assert "store.example" in sess.ctx.domain_guidance
    assert "use on-page links or search" in sess.ctx.domain_guidance["store.example"]


def test_404_guidance_injected_into_s2_context() -> None:
    """Guidance injected into S2 context steering towards on-page links or search after 404."""
    ctx = DualTierContext(primary_goal="Find product")
    ctx.domain_guidance["catalog.example"] = "Before guessing new deep URLs on this domain, use on-page links or search."

    block = ctx.build_supervisor_context_block([])
    assert "Domain Guidance" in block
    assert "catalog.example" in block
    assert "use on-page links or search" in block


def test_goto_url_remains_permitted_for_valid_targets_after_404() -> None:
    """GOTO_URL is not globally banned after 404; remains permitted for valid targets and search."""
    sess = _make_mock_session(url="https://store.example/home")
    sess._404_domains.add("store.example")
    sess.ctx.domain_guidance["store.example"] = "Before guessing new deep URLs on this domain, use on-page links or search."

    # Ungrounded deep path on 404 domain is rejected
    sup_ungrounded = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="GOTO_URL",
        target="https://store.example/categories/shoes/item42",
        reasoning="Guess another product URL",
    )
    res_ungrounded = sess._resolve_supervisor_action(sup_ungrounded, {}, {})
    assert res_ungrounded is None
    assert any("ungrounded deep URL on 404 domain" in rej for rej in sess.ctx.rejected_links)

    # Search URL on 404 domain is permitted
    sup_search = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="GOTO_URL",
        target="https://store.example/search?q=shoes",
        reasoning="Search using site search URL",
    )
    res_search = sess._resolve_supervisor_action(sup_search, {}, {})
    assert res_search is not None
    assert res_search.get("url") == "https://store.example/search?q=shoes"

    # URL on completely different domain is permitted
    sup_other = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="GOTO_URL",
        target="https://other-domain.example/catalog",
        reasoning="Navigate to other site",
    )
    res_other = sess._resolve_supervisor_action(sup_other, {}, {})
    assert res_other is not None
    assert res_other.get("url") == "https://other-domain.example/catalog"


# ==============================================================================
# Test Suite 5: Isolated Mock Architecture Tests
# ==============================================================================


def test_mock_architecture_runs_offline_under_5_seconds() -> None:
    """Ensures test suite executes completely offline without real CDP or network in < 5 seconds."""
    t0 = time.perf_counter()
    sess = _make_mock_session()

    page = {
        "url": "https://example.com/step",
        "fingerprint": "fp",
        "text": "Some text",
        "actions": [{"id": "btn", "kind": "click", "node": 1}],
    }
    sess.page = page
    sess._observe_settled = lambda: page

    for _ in range(20):
        sess._execute_and_verify(
            action={"kind": "click", "id": "btn", "node": 1, "label": "Click"},
            actor="S1",
            operation="CLICK",
            target="1",
            confidence=0.9,
            target_confidence=None,
            joint_confidence=0.9,
            target_margin=0.9,
            latency_ms=1,
            model_name="mock-model",
            usage={},
            preset_text=None,
            escalation_note=None,
        )

    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"Expected < 1.0s runtime, took {elapsed:.3f}s"
