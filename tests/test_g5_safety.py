"""Unit tests for Task G5: Comprehensive Irreversible Action Safety Guard."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexSession
from reflex.config import ReflexConfig
from reflex.safety import (
    ACCOUNT_PATTERNS,
    DELETION_PATTERNS,
    IRREVERSIBLE_PATTERNS,
    PAYMENT_PATTERNS,
    check_action_safety,
    extract_origin,
)
from reflex.supervisor import UserMessageInterpretation


@pytest.fixture(autouse=True)
def _mock_session_init_supervisor():
    with patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()):
        yield


def _make_fake_browser(page: dict[str, Any] | None = None) -> MagicMock:
    if page is None:
        page = {
            "url": "https://shop.example.com/checkout",
            "title": "Checkout",
            "text": "Complete your purchase",
            "fingerprint": "fp_checkout_1",
            "actions": [
                {"id": "pay_btn", "node": 10, "kind": "click", "role": "button", "label": "Öde"},
                {"id": "cancel_btn", "node": 11, "kind": "click", "role": "button", "label": "Cancel"},
            ],
            "elements": [],
        }
    fake_b = MagicMock()
    fake_b.observe.return_value = page
    fake_b.fresh.return_value = True
    fake_b.act.return_value = {
        "url": page["url"],
        "title": page["title"],
        "text": page["text"],
        "actions": page["actions"],
        "fingerprint": "fp_checkout_2",
    }
    fake_b.evaluate.return_value = None
    return fake_b


# ---------------------------------------------------------------------------
# 1. extract_origin Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected_origin"),
    [
        ("https://example.com/checkout/pay", "https://example.com"),
        ("https://example.com:8443/api/delete", "https://example.com:8443"),
        ("http://127.0.0.1:8080/dashboard", "http://127.0.0.1:8080"),
        ("http://localhost:3000/", "http://localhost:3000"),
        ("https://sub.domain.co.uk/path?q=1#hash", "https://sub.domain.co.uk"),
        ("/relative/path", ""),
        ("", ""),
        (None, ""),
        ("invalid_url", ""),
    ],
)
def test_extract_origin(url: str | None, expected_origin: str) -> None:
    assert extract_origin(url) == expected_origin


# ---------------------------------------------------------------------------
# 2. Parametric Tests for Required Irreversible Action Keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "Satın Al",
        "Sil",
        "Delete",
        "Checkout",
        "Öde",
        "Pay",
        "Hesabı Kapat",
    ],
)
def test_irreversible_patterns_required_keywords(label: str) -> None:
    """Task G5 requirement: Parametric tests for 'Satın Al', 'Sil', 'Delete',

    'Checkout', 'Öde', 'Pay', 'Hesabı Kapat'.
    """
    action = {"kind": "click", "label": label}
    result = check_action_safety(action)
    assert result.requires_confirmation is True
    assert label in result.reason
    assert result.category in {"payment", "deletion", "account"}


@pytest.mark.parametrize(
    ("label", "expected_category"),
    [
        ("Ödeyin", "payment"),
        ("Ödeme Yap", "payment"),
        ("Ödemeyi Tamamla", "payment"),
        ("Silin", "deletion"),
        ("Kalıcı olarak sil", "deletion"),
        ("Satın Alma", "payment"),
        ("Purchase", "payment"),
        ("Buy Now", "payment"),
        ("Hesabı Sil", "account"),
        ("Close Account", "account"),
        ("Delete Account", "account"),
        ("Revoke", "account"),
        ("Remove Account", "deletion"),
        ("Çöpe At", "deletion"),
        ("Destroy", "deletion"),
        ("Erase", "deletion"),
        ("Siparişi Ver", "payment"),
        ("Siparişi Tamamla", "payment"),
        ("Place Order", "payment"),
        ("Complete Purchase", "payment"),
        ("Aboneliği İptal Et", "account"),
        ("Üyeliğimi Kapat", "account"),
    ],
)
def test_irreversible_patterns_extended_bilingual(label: str, expected_category: str) -> None:
    """Comprehensive bilingual variations (Turkish & English) must be flagged with correct category."""
    action = {"kind": "click", "label": label}
    result = check_action_safety(action)
    assert result.requires_confirmation is True
    assert result.category == expected_category


@pytest.mark.parametrize(
    "label",
    [
        "Devam Et",
        "Cancel",
        "Arama Yap",
        "Search",
        "Sepete Ekle",
        "Open Details",
        "Geri",
        "Kaydet",
        "Save",
        "Düzenle",
        "Edit",
        "İptal",
        "Filtrele",
        "Sonraki Sayfa",
        "Temizle",
        "Kaldır",
        "Kaldırın",
        "Remove",
        "Çöp",
        "Ödeme",
        "Sipariş",
        "Filtreyi kaldır",
        "Remove filter",
    ],
)
def test_safe_actions_pass_through(label: str) -> None:
    """Safe actions (standard navigation, cart additions, search, cancel dialogs, benign terms) must NOT be blocked."""
    action = {"kind": "click", "label": label}
    result = check_action_safety(action)
    assert result.requires_confirmation is False


# ---------------------------------------------------------------------------
# 3. Destructive Action Kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["click", "select", "press_enter", "goto_url"])
def test_destructive_kinds_trigger_confirmation(kind: str) -> None:
    action = {"kind": kind, "label": "Sil", "url": "https://example.com/delete"}
    result = check_action_safety(action)
    assert result.requires_confirmation is True


@pytest.mark.parametrize("kind", ["scroll", "wait", "inspect"])
def test_safe_kinds_never_trigger_confirmation(kind: str) -> None:
    action = {"kind": kind, "label": "Öde"}
    result = check_action_safety(action)
    assert result.requires_confirmation is False


# ---------------------------------------------------------------------------
# 4. Origin-Scoped Signatures & Cross-Origin Isolation
# ---------------------------------------------------------------------------


def test_cross_origin_isolation() -> None:
    """Approval signature on origin A must NEVER authorize an action on origin B."""
    action_a = {
        "kind": "click",
        "label": "Pay Now",
        "origin": "https://checkout.example.com",
    }
    action_b = {
        "kind": "click",
        "label": "Pay Now",
        "origin": "https://phishing.example.com",
    }

    approved = {"click:Pay Now:https://checkout.example.com"}

    # Attempt on origin B (phishing site) with approval from origin A
    res_b = check_action_safety(action_b, approved_signatures=approved)
    assert res_b.requires_confirmation is True
    assert "click:Pay Now:https://checkout.example.com" in approved

    # Attempt on origin A (legitimate site) with approval from origin A
    res_a = check_action_safety(action_a, approved_signatures=approved)
    assert res_a.requires_confirmation is False
    # Signature must be consumed
    assert "click:Pay Now:https://checkout.example.com" not in approved


# ---------------------------------------------------------------------------
# 5. Single-Use Signature Consumption
# ---------------------------------------------------------------------------


def test_single_use_signature_consumption() -> None:
    """Approved signature must be consumed on first execution and require confirmation on second attempt."""
    action = {
        "kind": "click",
        "label": "Delete Account",
        "origin": "https://account.example.com",
    }
    approved = {"click:Delete Account:https://account.example.com"}

    # First attempt: approved
    first_res = check_action_safety(action, approved_signatures=approved)
    assert first_res.requires_confirmation is False
    assert len(approved) == 0

    # Second attempt: requires confirmation again
    second_res = check_action_safety(action, approved_signatures=approved)
    assert second_res.requires_confirmation is True


# ---------------------------------------------------------------------------
# 6. Backward Compatibility for Legacy 2-Part Signatures
# ---------------------------------------------------------------------------


def test_legacy_2_part_signature_backward_compatibility() -> None:
    """Legacy signature format f'{kind}:{label}' must match and be consumed cleanly."""
    action = {"kind": "click", "label": "Pay Now $420.00"}
    approved = {"click:Pay Now $420.00"}

    first_res = check_action_safety(action, approved_signatures=approved)
    assert first_res.requires_confirmation is False
    assert len(approved) == 0

    second_res = check_action_safety(action, approved_signatures=approved)
    assert second_res.requires_confirmation is True


# ---------------------------------------------------------------------------
# 7. press_enter on Destructive Forms
# ---------------------------------------------------------------------------


def test_press_enter_inspects_focused_form_destructive() -> None:
    """press_enter must inspect focused form's submit button label and trigger confirmation if destructive."""
    fake_b = _make_fake_browser()
    # Mock evaluate returning destructive submit button label "Öde"
    fake_b.evaluate.return_value = "Öde"
    cfg = ReflexConfig(require_confirmation_for_destructive=True)

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com/checkout", goal="Complete purchase", config=cfg)

    # Action is press_enter
    action = {"id": "press_enter", "kind": "press_enter", "label": "Press Enter"}

    snap = session._execute_and_verify(
        action=action,
        actor="S2",
        operation="PRESS_ENTER",
        target=None,
        confidence=1.0,
        target_confidence=1.0,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=10,
        model_name="test_model",
        usage={},
        preset_text=None,
        escalation_note="Form submission",
    )

    assert snap["status"] == "waiting_confirmation"
    assert session.status == "waiting_confirmation"
    assert session.pending_confirmation is not None
    assert "Öde" in session.pending_confirmation["summary"]
    assert "https://shop.example.com" in session.pending_confirmation["signature"]


def test_press_enter_safe_form_no_confirmation() -> None:
    """press_enter on a safe form (e.g. submit button labeled 'Search') must proceed without confirmation."""
    fake_b = _make_fake_browser()
    fake_b.evaluate.return_value = "Search"
    cfg = ReflexConfig(require_confirmation_for_destructive=True)

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com/search", goal="Search products", config=cfg)

    action = {"id": "press_enter", "kind": "press_enter", "label": "Press Enter"}

    snap = session._execute_and_verify(
        action=action,
        actor="S2",
        operation="PRESS_ENTER",
        target=None,
        confidence=1.0,
        target_confidence=1.0,
        joint_confidence=1.0,
        target_margin=1.0,
        latency_ms=10,
        model_name="test_model",
        usage={},
        preset_text=None,
        escalation_note="Search submission",
    )

    # Must NOT wait for confirmation
    assert snap["status"] != "waiting_confirmation"
    assert session.status != "waiting_confirmation"
    assert session.pending_confirmation is None


# ---------------------------------------------------------------------------
# 8. NEW_GOAL Clears approved_actions
# ---------------------------------------------------------------------------


def test_new_goal_clears_approved_actions() -> None:
    """When a NEW_GOAL intent is received via reply_user, self.ctx.approved_actions must be cleared."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig()

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com", goal="Initial goal", config=cfg)

    # Populate approved_actions
    session.ctx.approved_actions.add("click:Pay Now:https://shop.example.com")
    session.ctx.approved_actions.add("click:Delete:https://shop.example.com")
    assert len(session.ctx.approved_actions) == 2

    mock_interp = UserMessageInterpretation(
        intent="NEW_GOAL",
        new_goal="Search for books instead",
    )

    with patch("reflex.agent.interpret_user_message", return_value=mock_interp):
        session.reply_user("New task: search books")

    assert len(session.ctx.approved_actions) == 0


# ---------------------------------------------------------------------------
# 9. End-to-End confirm_action Workflow (Approve & Single-Use, Reject)
# ---------------------------------------------------------------------------


def test_confirm_action_approve_executes_and_consumes_signature() -> None:
    """Approving an action executes it once, discards the signature, and requires confirmation again on retry."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig(require_confirmation_for_destructive=True)

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com/checkout", goal="Buy item", config=cfg)

    pay_action = {"id": "pay_btn", "kind": "click", "label": "Satın Al"}

    # Trigger destructive check
    snap = session._execute_and_verify(
        action=pay_action,
        actor="S1",
        operation="CLICK",
        target="pay_btn",
        confidence=0.95,
        target_confidence=0.95,
        joint_confidence=0.90,
        target_margin=0.5,
        latency_ms=20,
        model_name="turnike",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert session.status == "waiting_confirmation"
    assert session.pending_confirmation is not None
    sig = session.pending_confirmation["signature"]
    assert "https://shop.example.com" in sig

    # Approve action
    approved_snap = session.confirm_action(approved=True)
    assert session.pending_confirmation is None
    # Signature was consumed
    assert sig not in session.ctx.approved_actions

    # Repeating the exact same destructive action requires confirmation again!
    repeat_snap = session._execute_and_verify(
        action=pay_action,
        actor="S1",
        operation="CLICK",
        target="pay_btn",
        confidence=0.95,
        target_confidence=0.95,
        joint_confidence=0.90,
        target_margin=0.5,
        latency_ms=20,
        model_name="turnike",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert session.status == "waiting_confirmation"
    assert session.pending_confirmation is not None


def test_confirm_action_reject_blocks_session() -> None:
    """Declining an action sets status to blocked and does not execute."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig(require_confirmation_for_destructive=True)

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com/checkout", goal="Buy item", config=cfg)

    pay_action = {"id": "pay_btn", "kind": "click", "label": "Öde"}
    session._execute_and_verify(
        action=pay_action,
        actor="S1",
        operation="CLICK",
        target="pay_btn",
        confidence=0.95,
        target_confidence=0.95,
        joint_confidence=0.90,
        target_margin=0.5,
        latency_ms=20,
        model_name="turnike",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    assert session.status == "waiting_confirmation"

    # Reject
    snap = session.confirm_action(approved=False)
    assert session.status == "blocked"
    assert session.pending_confirmation is None
    assert "declined" in (session.last_error or "").lower()


def test_confirm_action_reject_with_alternative_instruction() -> None:
    """Declining with an alternative instruction sets tactical hint and readies session."""
    fake_b = _make_fake_browser()
    cfg = ReflexConfig(require_confirmation_for_destructive=True)

    with patch("reflex.agent.Browser", return_value=fake_b):
        session = ReflexSession(url="https://shop.example.com/checkout", goal="Buy item", config=cfg)

    pay_action = {"id": "pay_btn", "kind": "click", "label": "Öde"}
    session._execute_and_verify(
        action=pay_action,
        actor="S1",
        operation="CLICK",
        target="pay_btn",
        confidence=0.95,
        target_confidence=0.95,
        joint_confidence=0.90,
        target_margin=0.5,
        latency_ms=20,
        model_name="turnike",
        usage={},
        preset_text=None,
        escalation_note=None,
    )

    snap = session.confirm_action(approved=False, alternative_instruction="Go back to cart")
    assert session.status == "ready"
    assert "Go back to cart" in session.ctx.tactical_hint


# ---------------------------------------------------------------------------
# 10. goto_url Destructive Path Tests
# ---------------------------------------------------------------------------


def test_goto_url_destructive_path() -> None:
    dest_action = {
        "id": "goto_url",
        "kind": "goto_url",
        "label": "Navigate to delete account",
        "url": "https://example.com/settings/delete-account",
    }
    result = check_action_safety(dest_action)
    assert result.requires_confirmation is True

    safe_action = {
        "id": "goto_url",
        "kind": "goto_url",
        "label": "Navigate to products",
        "url": "https://example.com/products/all",
    }
    safe_result = check_action_safety(safe_action)
    assert safe_result.requires_confirmation is False


# ---------------------------------------------------------------------------
# 11. Milestone M4 (R4/Ö1 fix(G5)) Safety Guard Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "description"),
    [
        ({"kind": "click", "role": "button", "label": "Temizle"}, "Temizle button"),
        ({"kind": "click", "label": "Temizle"}, "Temizle bare action"),
        ({"kind": "click", "role": "button", "label": "Filtreyi kaldır"}, "Filtreyi kaldır button"),
        ({"kind": "click", "label": "Filtreyi kaldır"}, "Filtreyi kaldır bare action"),
        ({"kind": "click", "role": "button", "label": "Remove filter"}, "Remove filter button"),
        ({"kind": "click", "label": "Remove filter"}, "Remove filter bare action"),
        ({"kind": "click", "role": "link", "label": "Ödeme Yöntemleri"}, "Ödeme Yöntemleri link"),
        ({"kind": "click", "role": "link", "label": "Sipariş Takibi"}, "Sipariş Takibi link"),
        ({"kind": "click", "role": "link", "label": "Çöp Kutusu"}, "Çöp Kutusu link"),
        ({"kind": "click", "role": "link", "label": "Checkout guide"}, "Checkout guide link"),
    ],
)
def test_negative_confirmation_exempted_m4(action: dict[str, Any], description: str) -> None:
    """R4 / Ö1 fix(G5): Benign terms and role=link navigations must NOT require confirmation."""
    result = check_action_safety(action)
    assert result.requires_confirmation is False, f"{description} should NOT require confirmation"


@pytest.mark.parametrize(
    "label",
    [
        "Satın Al",
        "Sil",
        "Delete",
        "Ödemeyi Tamamla",
        "Place Order",
        "Siparişi Ver",
    ],
)
def test_positive_confirmation_required_m4(label: str) -> None:
    """R4 / Ö1 fix(G5): Dangerous button actions MUST require confirmation."""
    btn_action = {"kind": "click", "role": "button", "label": label}
    result = check_action_safety(btn_action)
    assert result.requires_confirmation is True, f"Button '{label}' MUST require confirmation"
    assert result.category in {"payment", "deletion", "account"}

    bare_action = {"kind": "click", "label": label}
    result_bare = check_action_safety(bare_action)
    assert result_bare.requires_confirmation is True, f"Bare action '{label}' MUST require confirmation"


def test_link_role_exempts_even_destructive_labels() -> None:
    """role='link' navigations are reversible GET requests and completely exempt from confirmation."""
    for label in ["Satın Al", "Sil", "Delete", "Öde", "Place Order", "Delete Account", "Checkout"]:
        link_action = {"kind": "click", "role": "link", "label": label, "url": "https://example.com/nav"}
        result = check_action_safety(link_action)
        assert result.requires_confirmation is False, f"Link with label '{label}' must be exempt"


def test_goto_url_does_not_inspect_url_text() -> None:
    """goto_url must not trigger confirmation based on URL text alone."""
    actions = [
        {"kind": "goto_url", "url": "https://example.com/checkout/guide"},
        {"kind": "goto_url", "url": "https://example.com/cöp-kutusu/odeme-yontemleri"},
        {"kind": "goto_url", "url": "https://example.com/admin/delete-item", "label": "View page"},
    ]
    for act in actions:
        result = check_action_safety(act)
        assert result.requires_confirmation is False, f"goto_url {act.get('url')} should not trigger on URL text"


def test_non_button_interactive_roles_do_not_trigger_confirmation() -> None:
    """Clicks on non-button elements (tabs, checkboxes, etc.) must not trigger confirmation guard."""
    for role in ["tab", "checkbox", "radio", "combobox"]:
        action = {"kind": "click", "role": role, "label": "Delete"}
        result = check_action_safety(action)
        assert result.requires_confirmation is False, f"Role '{role}' should not trigger confirmation"


def test_patterns_exclude_standalone_benign_terms() -> None:
    """Milestone M4 (Ö1 fix(G5)): Standalone benign nouns and reversible verbs are removed."""
    forbidden_exact = {
        r"\bödeme\b",
        r"\bsipari[şs]\b",
        r"\btemizle(?:yin|yiniz)?\b",
        r"\bkald[ıi]r(?:[ıi]n|[ıi]n[ıi]z)?\b",
        r"\bremove\b",
        r"\bçöp\b",
    }
    for pat in forbidden_exact:
        assert pat not in PAYMENT_PATTERNS
        assert pat not in DELETION_PATTERNS
        assert pat not in IRREVERSIBLE_PATTERNS
    assert len(ACCOUNT_PATTERNS) > 0


# ---------------------------------------------------------------------------
# 12. F3 fix(G5): Pseudo-Button Links (href="#", empty, or "javascript:")
# ---------------------------------------------------------------------------


def test_pseudo_button_links_require_confirmation() -> None:
    """F3 fix(G5): Links with href='#', empty, or starting with 'javascript:' act as buttons and require confirmation."""
    # Exact required test case from F3 specification (without explicit 'kind')
    res_js = check_action_safety({"role": "link", "href": "javascript:void(0)", "label": "Satın Al"})
    assert res_js.requires_confirmation is True
    assert res_js.category == "payment"

    # Hash href="#"
    res_hash = check_action_safety({"kind": "click", "role": "link", "href": "#", "label": "Satın Al"})
    assert res_hash.requires_confirmation is True
    assert res_hash.category == "payment"

    # Empty href=""
    res_empty = check_action_safety({"kind": "click", "role": "link", "href": "", "label": "Hesabı Sil"})
    assert res_empty.requires_confirmation is True
    assert res_empty.category == "account"

    # Missing href and url
    res_missing = check_action_safety({"role": "link", "label": "Delete"})
    assert res_missing.requires_confirmation is True
    assert res_missing.category == "deletion"

    # Genuine navigation link remains exempt
    res_nav = check_action_safety({"role": "link", "href": "https://example.com/shop/buy", "label": "Satın Al"})
    assert res_nav.requires_confirmation is False



