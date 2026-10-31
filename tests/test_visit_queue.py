"""Tests for System 2 visit queue (grounded URL queue, pruning, caps, and S1 isolation)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflex.agent import ReflexSession
from reflex.browser.cdp import NavigationTimeout
from reflex.config import ReflexConfig, run_mode_flags
from reflex.context import MAX_VISIT_QUEUE_URL_CHARS, MAX_VISIT_QUEUE_URLS, DualTierContext
from reflex.planner import Subgoal, TaskLedger
from reflex.supervisor import SupervisorDecision
from reflex.system1 import build_system1_request


def test_grounding_accepts_3_page_hrefs_and_rejects_1_ungrounded_url() -> None:
    """3 hrefs present on the page enter the queue; 1 URL not on the page is rejected and counted."""
    ctx = DualTierContext(primary_goal="Inspect listed items")
    elements = [
        {"index": "1", "role": "link", "label": "Item 1", "href": "https://directory.example/item/1"},
        {"index": "2", "role": "link", "label": "Item 2", "href": "https://directory.example/item/2"},
        {"index": "3", "role": "link", "label": "Item 3", "href": "https://directory.example/item/3"},
    ]
    proposed = [
        "https://directory.example/item/1",
        "https://directory.example/item/2/",  # canonical_url match with trailing slash
        "https://directory.example/item/3?utm_source=list",  # canonical_url match with tracking param
        "https://invented.example/not-on-page",  # ungrounded hallucination
    ]

    accepted = ctx.enqueue_urls(proposed, elements)

    assert len(accepted) == 3
    assert len(ctx.visit_queue) == 3
    assert "https://invented.example/not-on-page" not in ctx.visit_queue
    assert ctx.visit_queue_rejected_count == 1

    # Duplicate additions with canonical_url equivalence are deduplicated without increasing rejected count
    dup_accepted = ctx.enqueue_urls(["https://directory.example/item/1/"], elements)
    assert dup_accepted == []
    assert len(ctx.visit_queue) == 3
    assert ctx.visit_queue_rejected_count == 1


def test_visited_url_drops_from_queue() -> None:
    """URLs added to visited_urls (or navigated to) automatically drop from the queue."""
    ctx = DualTierContext(primary_goal="Open each item")
    hrefs = [
        "https://catalog.example/entry/1",
        "https://catalog.example/entry/2",
        "https://catalog.example/entry/3",
    ]
    ctx.enqueue_urls(hrefs, hrefs)
    assert ctx.visit_queue == hrefs

    # Visiting entry 1 (with trailing slash canonical equivalence) drops it immediately
    ctx.visited_urls.append("https://catalog.example/entry/1/")
    assert ctx.visit_queue == [
        "https://catalog.example/entry/2",
        "https://catalog.example/entry/3",
    ]
    s2_block = ctx.build_supervisor_context_block([])
    queued_section = s2_block.split("Queued URLs")[1]
    assert "https://catalog.example/entry/1" not in queued_section
    assert "https://catalog.example/entry/2" in queued_section
    assert "https://catalog.example/entry/3" in queued_section


class _TimeoutBrowser:
    def __init__(self, page: dict[str, Any], dead_url: str) -> None:
        self.page = page
        self.dead_url = dead_url

    def act(self, action: dict[str, Any], page: dict[str, Any], text: str | None = None) -> dict[str, Any]:
        raise NavigationTimeout(f"{self.dead_url} did not respond")

    def observe(self, screenshot: bool = False) -> dict[str, Any]:
        return self.page


def test_navigation_timeout_drops_url_from_queue() -> None:
    """A queued URL that fails with NavigationTimeout enters rejected_links and drops from visit_queue."""
    dead_url = "http://unreachable-item.example/"
    alive_url = "https://reachable-item.example/"
    page = {
        "url": "https://directory.example/list",
        "actions": [
            {"id": "a1", "node": 1, "kind": "click", "role": "link", "label": "Dead", "href": dead_url},
            {"id": "a2", "node": 2, "kind": "click", "role": "link", "label": "Alive", "href": alive_url},
        ],
        "fingerprint": "fp-list",
    }

    sess = ReflexSession.__new__(ReflexSession)
    sess.config = ReflexConfig(**run_mode_flags("s2_only"))
    sess.ctx = DualTierContext(primary_goal="Visit items", active_subgoal="Visit items")
    sess.page = page
    sess.history = []
    sess.browser = _TimeoutBrowser(page, dead_url)  # type: ignore[assignment]
    sess.status = "ready"
    sess.last_error = None
    sess.pending_escalation_reason = None
    sess._observe_settled = lambda: page  # type: ignore[method-assign]
    sess.snapshot = lambda: {"status": sess.status}  # type: ignore[method-assign]

    sess.ctx.enqueue_urls([dead_url, alive_url], page["actions"])
    assert sess.ctx.visit_queue == [dead_url, alive_url]

    sess._execute_and_verify(
        action={"kind": "goto_url", "url": dead_url, "label": f"GOTO_URL {dead_url}"},
        actor="S2",
        operation="GOTO_URL",
        target=None,
        confidence=1.0,
        target_confidence=None,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=0,
        model_name="m",
        usage={},
        preset_text=dead_url,
        escalation_note=None,
    )

    assert sess.pending_escalation_reason is not None
    assert sess.pending_escalation_reason[0] == "navigation_timeout"
    assert sess.ctx.visit_queue == [alive_url]
    s2_block = sess.ctx.build_supervisor_context_block([])
    queued_part = s2_block.split("Queued URLs")[1]
    assert dead_url not in queued_part
    assert alive_url in queued_part


def test_visit_queue_capped_at_30_and_200_chars_per_url() -> None:
    """Queue accepts at most 30 URLs when 35 are provided, and truncates each URL to at most 200 characters."""
    ctx = DualTierContext(primary_goal="Collect 30 items")
    long_suffix = "x" * 210
    hrefs = [
        f"https://directory.example/item/{i:02d}?pad={long_suffix}" if i == 1 else f"https://directory.example/item/{i:02d}"
        for i in range(1, 36)
    ]

    ctx.enqueue_urls(hrefs, hrefs)

    assert len(ctx.visit_queue) == MAX_VISIT_QUEUE_URLS == 30
    assert all(len(u) <= MAX_VISIT_QUEUE_URL_CHARS == 200 for u in ctx.visit_queue)
    assert len(ctx.visit_queue[0]) == 200
    assert "https://directory.example/item/30" in ctx.visit_queue
    assert "https://directory.example/item/31" not in ctx.visit_queue
    assert ctx.visit_queue_rejected_count == 0


def test_queue_shown_in_s2_context_and_excluded_from_system1_goal_and_request() -> None:
    """Queued URLs appear in S2 context block but never in compile_system1_goal or System 1 request."""
    ctx = DualTierContext(
        primary_goal="Inspect listed entries",
        active_subgoal="Open first entry",
    )
    urls = ["https://directory.example/entry-a", "https://directory.example/entry-b"]
    ctx.enqueue_urls(urls, urls)

    s2_block = ctx.build_supervisor_context_block([])
    assert "Queued URLs (collected earlier; open with GOTO_URL directly, no need to return to the source page):" in s2_block
    assert "  - https://directory.example/entry-a" in s2_block
    assert "  - https://directory.example/entry-b" in s2_block

    s1_goal = ctx.compile_system1_goal()
    assert "Queued URLs" not in s1_goal
    assert "entry-a" not in s1_goal
    assert "entry-b" not in s1_goal

    page = {
        "url": "https://directory.example/",
        "title": "Directory",
        "text": "Entries",
        "actions": [{"id": "c1", "node": 1, "kind": "click", "role": "link", "label": "Entry A", "href": urls[0]}],
    }
    s1_body, _, _ = build_system1_request(page, s1_goal, [])
    serialized_s1 = str(s1_body)
    assert "Queued URLs" not in serialized_s1
    assert "entry-b" not in serialized_s1


def test_visit_queue_works_both_with_and_without_plan() -> None:
    """Supervisor queue_urls populates visit_queue and renders in S2 context both with and without an active plan."""
    list_page = {
        "url": "https://directory.example/members",
        "title": "Members",
        "text": "Member list",
        "fingerprint": "fp-members",
        "actions": [
            {"id": "a1", "node": 1, "kind": "click", "role": "link", "label": "Alpha", "href": "https://alpha.example/"},
            {"id": "a2", "node": 2, "kind": "click", "role": "link", "label": "Beta", "href": "https://beta.example/"},
        ],
        "page_links": [
            {"role": "link", "label": "Gamma", "href": "https://gamma.example/"},
        ],
    }
    detail_page = {
        "url": "https://alpha.example/",
        "title": "Alpha",
        "text": "Alpha details",
        "fingerprint": "fp-alpha",
        "actions": [],
    }

    for with_plan in (False, True):
        with patch("reflex.agent.Browser") as mock_browser_cls, patch("reflex.agent.interpret_user_message") as mock_interp:
            mock_browser = MagicMock()
            mock_browser.observe.side_effect = [list_page, detail_page]
            mock_browser.act.return_value = {"executed": "goto_url"}
            mock_browser_cls.return_value = mock_browser

            plan_obj = (
                TaskLedger(
                    needs_plan=True,
                    subgoals=[Subgoal(id="sg_1", text="Inspect members", status="active")],
                )
                if with_plan
                else None
            )
            mock_interp.return_value = MagicMock(
                allow_arbitrary_choices=False,
                constraints=[],
                plan=plan_obj,
                model_dump=lambda: {},
            )

            session = ReflexSession(url="https://directory.example/members", goal="Inspect 3 members")
            if not with_plan:
                session.ctx.plan = None

            sup = SupervisorDecision(
                action_type="OVERRIDE_ACTION",
                operation="GOTO_URL",
                url="https://alpha.example/",
                queue_urls=[
                    "https://alpha.example/",
                    "https://beta.example/",
                    "https://gamma.example/",
                    "https://nonexistent.example/",
                ],
                memory="Queued member URLs and opening Alpha",
                reasoning="Queue links before leaving list page",
            )
            with patch("reflex.agent.resolve_escalation", return_value=sup):
                session._handle_escalation(trigger_type="s2_every_step", reason="step", s1_decision=None)

            # Alpha was opened on this step so it drops from the queue; Beta and Gamma remain queued; nonexistent was rejected
            assert session.ctx.visit_queue == ["https://beta.example/", "https://gamma.example/"]
            assert session.ctx.visit_queue_rejected_count == 1
            s2_block = session.ctx.build_supervisor_context_block(session.history)
            assert "Queued URLs (collected earlier; open with GOTO_URL directly, no need to return to the source page):" in s2_block
            assert "https://beta.example/" in s2_block
            assert "https://gamma.example/" in s2_block
            if with_plan:
                assert session.ctx.plan is not None
                assert "Plan Status: Active" in session.ctx.plan.format_for_s2()
                assert "Queued URLs" not in session.ctx.plan.format_for_s2()


def test_resolve_escalation_parses_queue_urls_from_json_response() -> None:
    """resolve_escalation parses queue_urls list from the Supervisor JSON completion."""
    from reflex.supervisor import CompletionResult, resolve_escalation

    ctx = DualTierContext(primary_goal="Inspect items")
    page = {"url": "https://directory.example/", "title": "List", "text": "Items", "actions": []}
    elements = [
        {"index": "1", "role": "link", "label": "One", "href": "https://one.example/", "operations": ["CLICK"]},
        {"index": "2", "role": "link", "label": "Two", "href": "https://two.example/", "operations": ["CLICK"]},
    ]
    fake_comp = CompletionResult(
        {
            "evaluation_previous_goal": "Starting on directory",
            "memory": "Queueing 2 links",
            "next_goal": "Open first item",
            "action_type": "OVERRIDE_ACTION",
            "reasoning": "Queue listed items and open first",
            "operation": "GOTO_URL",
            "url": "https://one.example/",
            "queue_urls": ["https://one.example/", "https://two.example/", "  ", 123],
        },
        120,
        0.001,
        "test-model",
    )
    with patch("reflex.supervisor._chat_completion", return_value=fake_comp):
        dec = resolve_escalation(
            ctx=ctx,
            page=page,
            elements=elements,
            history=[],
            escalation_reason="step",
            trigger_type="s2_every_step",
            s1_decision=None,
            supervisor_model="test-model",
        )

    assert dec.queue_urls == ["https://one.example/", "https://two.example/"]


def test_new_goal_clears_visit_queue_and_rejected_count() -> None:
    """Switching to a NEW_GOAL clears any leftover visit_queue and rejected count."""
    from reflex.supervisor import UserMessageInterpretation

    with patch("reflex.agent.Browser") as mock_browser_cls, patch("reflex.agent.interpret_user_message") as mock_interp:
        mock_browser = MagicMock()
        mock_browser.observe.return_value = {"url": "https://directory.example/", "actions": []}
        mock_browser_cls.return_value = mock_browser
        mock_interp.return_value = UserMessageInterpretation(intent="CONTINUE_OR_STEER")

        session = ReflexSession(url="https://directory.example/", goal="Initial list task")
        session.ctx.enqueue_urls(
            ["https://a.example/", "https://fake.example/"],
            ["https://a.example/"],
        )
        assert session.ctx.visit_queue == ["https://a.example/"]
        assert session.ctx.visit_queue_rejected_count == 1

        mock_interp.return_value = UserMessageInterpretation(
            intent="NEW_GOAL",
            new_goal="Completely different task",
        )
        session.reply_user("Switch to a new task")

        assert session.ctx.visit_queue == []
        assert session.ctx.visit_queue_rejected_count == 0

