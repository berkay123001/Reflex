"""Personal-data fields get only values the user gave; nothing is made up (offline).

Online-Mind2Web full300 (2026-11-04, landwatch.com): with no user present, System 2 filled a seller
contact form with "John Buyer" / "john.buyer@example.com" and sent it; after the send button started
asking for approval it still typed the made-up details. The prompt rule alone did not stop it.
"""

from __future__ import annotations

import time

from reflex.agent import _is_personal_field, _user_gave
from reflex.context import DualTierContext
from tests.test_stuck_finish import _make_mock_session


def _fill(label: str) -> dict:
    return {"id": "e1", "kind": "fill", "role": "textbox", "label": label, "node": 7}


def _type(sess, action: dict, text: str) -> None:
    sess._execute_and_verify(action=action, actor="S2", operation="TYPE_TEXT", target="1", confidence=1.0,
                             target_confidence=1.0, joint_confidence=1.0, target_margin=1.0, latency_ms=0,
                             model_name="test", usage={}, preset_text=text, escalation_note=None,
                             step_t0=time.perf_counter())


def test_personal_fields_are_recognised() -> None:
    for label in ("Anthony Napodano → Name", "Email Address", "E-mail*", "Phone Number", "Mobile phone",
                  "Full Name or Company Name*", "Password", "Card number", "Street address", "Address line 1"):
        assert _is_personal_field(_fill(label)), label
    for label in ("Search by country, city, zip", "Location", "ZIP code", "Household Income", "Keywords",
                  "Search by agent name", "City or ZIP", "Message", "Weight*"):
        assert not _is_personal_field(_fill(label)), label


def test_only_values_from_the_user_pass() -> None:
    ctx = DualTierContext(primary_goal="My name is Mike. Send Christene (christenson@gmail.com) a $100 eGift Card.", active_subgoal="g")
    assert _user_gave(ctx, "Mike")
    assert _user_gave(ctx, "christenson@gmail.com")
    assert not _user_gave(ctx, "John Buyer")
    assert not _user_gave(ctx, "john.buyer@example.com")
    ctx.add_user_message("my phone is (575) 570-8186")
    assert _user_gave(ctx, "5755708186")
    ctx.user_facts["email"] = "me@site.org"
    assert _user_gave(ctx, "me@site.org")


def test_made_up_personal_value_is_not_typed() -> None:
    sess = _make_mock_session(goal="Contact the cheapest land seller in Luna County")
    action = _fill("Anthony Napodano → Email Address")
    sess.page["actions"] = [action]
    calls = []
    sess.browser.act = lambda *a, **k: calls.append(a) or {}
    _type(sess, action, "john.buyer@example.com")
    assert calls == []
    trigger, reason = sess.pending_escalation_reason
    assert trigger == "personal_data"
    assert "did not give" in reason and "Email Address" in reason
    assert sess.status == "ready"


def test_value_from_the_goal_is_typed() -> None:
    sess = _make_mock_session(goal="Track order 12345 with email 12345@gmail.com. My last name is Johnson.")
    action = _fill("Email")
    sess.page["actions"] = [action]
    calls = []
    sess.browser.act = lambda *a, **k: calls.append(a) or {}
    _type(sess, action, "12345@gmail.com")
    assert calls, "a value the user gave must reach the browser"
