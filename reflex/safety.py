"""Safety guards for irreversible browser actions and untrusted DOM isolation."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel


def extract_origin(url: str | None) -> str:
    """Extract scheme + netloc (e.g. 'https://example.com' or 'http://localhost:3000') from a URL.

    Returns an empty string if url is missing, invalid, or relative.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(str(url).strip())
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}".lower()
        return ""
    except Exception:
        return ""


# Patterns that represent irreversible financial, destructive, or external-transmission actions
PAYMENT_PATTERNS = [
    r"\bpay\b",
    r"\bpay\s+now\b",
    r"\bcheckout\b(?!\s+guide\b)",
    r"\bcheck\s*out\b(?!\s+guide\b)",
    r"\bbuy\b",
    r"\bbuy\s+now\b",
    r"\bpurchase\b",
    r"\bcomplete\s+(?:purchase|order|payment)\b",
    r"\bplace\s+order\b",
    r"\bsend\s+(?:money|payment|transfer|wire)\b",
    r"\bauthorize\s+payment\b",
    r"\bconfirm\s+(?:payment|purchase|order|transfer)\b",
    r"\böde(?:yin|yiniz)?\b",
    r"\bödeme(?:yi)?\s+(?:yap|onayla|tamamla|et)\b",
    r"\bsat[ıi]n\s+al(?:[ıi]n|[ıi]n[ıi]z)?\b",
    r"\bsat[ıi]n\s+alma\b",
    r"\bhemen\s+sat[ıi]n\s+al\b",
    r"\bsipari[şs](?:i|ini)?\s+(?:ver|tamamla|onayla|olu[şs]tur)\b",
    r"\bsipari[şs]\s*ver\b",
]

DELETION_PATTERNS = [
    r"\bdelete\b",
    r"\bdelete\s+(?:account|permanently|repository|project|all)\b",
    r"\bremove\s+account\b",
    r"\bdestroy\b",
    r"\berase\b",
    r"\bpurge\b",
    r"\bsil(?:in|iniz)?\b",
    r"\bkal[ıi]c[ıi]\s+olarak\s+sil\b",
    r"\bçöpe?\s*at\b",
]

ACCOUNT_PATTERNS = [
    r"\bclose\s+account\b",
    r"\bdelete\s+account\b",
    r"\bcancel\s+subscription\b",
    r"\brevoke\b",
    r"\brevoke\s+access\b",
    r"\bterminate\s+account\b",
    r"\bhesab[ıi](?:m[ıi]|n[ıi])?\s+(?:kapat|sil|sonland[ıi]r)\b",
    r"\büyeli[ğg](?:i|imi|ini)?\s+(?:iptal\s+et|kapat|sil)\b",
    r"\baboneli[ğg](?:i|imi|ini)?\s+(?:iptal\s+et|sonland[ıi]r)\b",
]

IRREVERSIBLE_PATTERNS = PAYMENT_PATTERNS + DELETION_PATTERNS + ACCOUNT_PATTERNS

_COMPILED_CATEGORIZED_PATTERNS: list[tuple[re.Pattern, str]] = (
    [(re.compile(pat, re.IGNORECASE), "account") for pat in ACCOUNT_PATTERNS]
    + [(re.compile(pat, re.IGNORECASE), "payment") for pat in PAYMENT_PATTERNS]
    + [(re.compile(pat, re.IGNORECASE), "deletion") for pat in DELETION_PATTERNS]
)
_COMPILED_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in IRREVERSIBLE_PATTERNS]

DESTRUCTIVE_KINDS = {"click", "select", "press_enter", "goto_url"}


class SafetyCheckResult(BaseModel):
    """Result of checking whether an action requires explicit human confirmation."""

    requires_confirmation: bool = False
    reason: str = ""
    matched_pattern: str | None = None
    action_summary: str = ""
    action_signature: str = ""
    category: str | None = None


def check_action_safety(
    action: dict[str, Any],
    *,
    origin: str | None = None,
    approved_signatures: set[str] | None = None,
    enabled: bool = True,
) -> SafetyCheckResult:
    """Return whether a candidate browser action is irreversible and requires user confirmation."""
    if not enabled:
        return SafetyCheckResult(requires_confirmation=False)

    raw_kind = action.get("kind")
    kind = str(raw_kind if raw_kind is not None and str(raw_kind).strip() != "" else "click").strip().lower()
    label = str(action.get("label", "")).strip()
    submit_label = str(action.get("submit_label", "")).strip()
    effective_label = label or submit_label

    role = str(action.get("role", "")).strip().lower() if action.get("role") is not None else None
    raw_href_val = action.get("href") if "href" in action else action.get("url", "")
    href = str(raw_href_val or "").strip()
    is_button_like_link = (
        role == "link"
        and (href == "" or href == "#" or href.lower().startswith("javascript:"))
    )

    # Determine origin: explicit arg -> action["origin"] -> action["url"]
    action_origin = origin
    if action_origin is None:
        if action.get("origin"):
            action_origin = str(action["origin"]).strip()
        elif action.get("url"):
            action_origin = extract_origin(action["url"])

    norm_origin = extract_origin(action_origin) if action_origin else ""

    # Signatures
    sig_origin = f"{kind}:{effective_label}:{norm_origin}" if norm_origin else f"{kind}:{effective_label}"
    sig_legacy = f"{kind}:{effective_label}"

    # Check for approval (single-use consumption)
    if approved_signatures is not None:
        if norm_origin and sig_origin in approved_signatures:
            if hasattr(approved_signatures, "discard"):
                approved_signatures.discard(sig_origin)
            elif hasattr(approved_signatures, "remove"):
                approved_signatures.remove(sig_origin)
            return SafetyCheckResult(
                requires_confirmation=False,
                action_summary=sig_origin,
                action_signature=sig_origin,
            )
        if sig_legacy in approved_signatures:
            if hasattr(approved_signatures, "discard"):
                approved_signatures.discard(sig_legacy)
            elif hasattr(approved_signatures, "remove"):
                approved_signatures.remove(sig_legacy)
            return SafetyCheckResult(
                requires_confirmation=False,
                action_summary=sig_legacy,
                action_signature=sig_legacy,
            )

    # 1. Genuine navigation links are GET navigations and inherently reversible;
    # links with empty href, '#', or 'javascript:' act like buttons and must be checked.
    if role == "link" and not is_button_like_link:
        return SafetyCheckResult(
            requires_confirmation=False,
            action_summary=sig_origin,
            action_signature=sig_origin,
        )

    if kind not in DESTRUCTIVE_KINDS:
        return SafetyCheckResult(
            requires_confirmation=False,
            action_summary=sig_origin,
            action_signature=sig_origin,
        )

    # 2. Confirmation guard applies only to button/submit elements (and button-like links).
    # For clicks, non-button elements (e.g. tabs, checkboxes, comboboxes) do not trigger confirmation.
    # Note: role is None or empty is allowed for backward compatibility with manual/test action dicts.
    if kind == "click":
        is_button_or_submit = (
            role in ("button", "submit")
            or is_button_like_link
            or bool(action.get("submits_form"))
            or str(action.get("type", "")).strip().lower() in ("button", "submit")
            or str(action.get("tag", "")).strip().lower() == "button"
            or role is None
            or role == ""
        )
        if not is_button_or_submit:
            return SafetyCheckResult(
                requires_confirmation=False,
                action_summary=sig_origin,
                action_signature=sig_origin,
            )

    texts_to_check: list[str] = []
    if label:
        texts_to_check.append(label)
    if submit_label and submit_label != label:
        texts_to_check.append(submit_label)

    for pat, cat in _COMPILED_CATEGORIZED_PATTERNS:
        for txt in texts_to_check:
            if pat.search(txt):
                return SafetyCheckResult(
                    requires_confirmation=True,
                    reason=f"Irreversible or high-impact {cat} action detected on '{txt}'",
                    matched_pattern=pat.pattern,
                    action_summary=f"{kind.upper()} '{txt}'",
                    action_signature=sig_origin,
                    category=cat,
                )

    return SafetyCheckResult(
        requires_confirmation=False,
        action_summary=sig_origin,
        action_signature=sig_origin,
    )


def sanitize_untrusted_text(raw: str, max_chars: int = 4000) -> str:
    """Neutralize case-insensitive tag-escape attempts inside untrusted webpage strings."""
    cleaned = str(raw or "")
    cleaned = re.sub(r"</?\s*untrusted_page_data[^>]*>", "[untrusted_page_data]", cleaned, flags=re.IGNORECASE)
    return cleaned[:max_chars]


def wrap_untrusted_dom(page: dict[str, Any], elements: list[dict[str, Any]], max_text_chars: int = 3000) -> str:
    """Format observed DOM elements and page text inside explicit untrusted delimiters."""
    url = sanitize_untrusted_text(str(page.get("url", "")), 500)
    title = sanitize_untrusted_text(str(page.get("title", "")), 300)
    text = sanitize_untrusted_text(str(page.get("text", "")), max_text_chars)
    raw_full = str(page.get("full_text", "") or "")
    if len(raw_full) > 7500:
        raw_full = raw_full[:4200] + "\n...[middle section omitted]...\n" + raw_full[-3000:]
    full_text = sanitize_untrusted_text(raw_full, 7800)
    raw_headings = page.get("headings") or []
    headings_str = " | ".join(sanitize_untrusted_text(str(h), 120) for h in raw_headings[:80]) if raw_headings else ""

    scroll = page.get("scroll") or {}
    scroll_y = int(scroll.get("y", 0) or 0)
    scroll_h = int(scroll.get("height", 0) or 0)
    vp_h = int(page.get("h", 780) or 780)
    at_top = scroll_y <= 4
    at_bottom = (scroll_y + vp_h) >= (scroll_h - 8) if scroll_h > 0 else True
    scroll_line = f"y={scroll_y}px / total_height={scroll_h}px (at_top={at_top}, at_bottom={at_bottom})"

    elem_lines: list[str] = []
    for el in elements[:60]:
        idx = el.get("index")
        role = el.get("role", "")
        lbl = sanitize_untrusted_text(str(el.get("label", "")), 140)
        ops = ",".join(el.get("operations", []))
        extras = []
        for k in ("value", "checked", "selected", "expanded", "href"):
            if el.get(k) not in (None, ""):
                max_len = 180 if k == "href" else 80
                extras.append(f"{k}={sanitize_untrusted_text(str(el[k]), max_len)}")
        extra_str = (" | " + ", ".join(extras)) if extras else ""
        elem_lines.append(f"[{idx}] ({role}) {lbl}{extra_str} | ops: {ops}")
        for opt in el.get("options", [])[:15]:
            opt_lbl = sanitize_untrusted_text(str(opt.get("label", "")), 100)
            elem_lines.append(f"    option [{opt.get('index')}] {opt_lbl}")

    elements_block = "\n".join(elem_lines) if elem_lines else "(no interactive elements observed)"
    extra_full_block = ""
    if full_text and len(full_text) > len(text) + 40:
        extra_full_block = f"\n\nFull Page Text (Entire Document From Top to Bottom):\n{full_text}"
    headings_block = f"\nAll Page Headings (h1-h4): {headings_str}" if headings_str else ""

    return (
        "<untrusted_page_data>\n"
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Scroll Position: {scroll_line}{headings_block}\n"
        f"Interactive Elements:\n{elements_block}\n\n"
        f"Visible Viewport Text:\n{text}{extra_full_block}\n"
        "</untrusted_page_data>"
    )
