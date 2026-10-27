"""Safety guards for irreversible browser actions and untrusted DOM isolation."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

# Patterns that represent irreversible financial, destructive, or external-transmission actions
IRREVERSIBLE_PATTERNS = [
    r"\bpay\s+now\b",
    r"\bcomplete\s+(?:purchase|order|payment)\b",
    r"\bplace\s+order\b",
    r"\bbuy\s+now\b",
    r"\bconfirm\s+(?:payment|purchase|order|transfer|deletion)\b",
    r"\bdelete\s+(?:account|permanently|repository|project|all)\b",
    r"\bremove\s+account\b",
    r"\bsend\s+(?:money|payment|transfer|wire)\b",
    r"\bauthorize\s+payment\b",
    r"\bsiparişi\s+(?:tamamla|onayla)\b",
    r"\bödeme(?:yi)?\s+(?:yap|onayla|tamamla)\b",
    r"\bhemen\s+satın\s+al\b",
    r"\bkalıcı\s+olarak\s+sil\b",
    r"\bhesabı\s+sil\b",
]

_COMPILED_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in IRREVERSIBLE_PATTERNS]


class SafetyCheckResult(BaseModel):
    """Result of checking whether an action requires explicit human confirmation."""

    requires_confirmation: bool = False
    reason: str = ""
    matched_pattern: str | None = None
    action_summary: str = ""


def check_action_safety(
    action: dict[str, Any],
    *,
    approved_signatures: set[str] | None = None,
    enabled: bool = True,
) -> SafetyCheckResult:
    """Return whether a candidate browser action is irreversible and requires user confirmation."""
    if not enabled:
        return SafetyCheckResult(requires_confirmation=False)

    kind = str(action.get("kind", ""))
    label = str(action.get("label", "")).strip()
    signature = f"{kind}:{label}"

    if approved_signatures and signature in approved_signatures:
        return SafetyCheckResult(requires_confirmation=False, action_summary=signature)

    if kind not in {"click", "select"}:
        return SafetyCheckResult(requires_confirmation=False, action_summary=signature)

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(label):
            return SafetyCheckResult(
                requires_confirmation=True,
                reason=f"Irreversible or high-impact action detected on '{label}'",
                matched_pattern=pattern.pattern,
                action_summary=f"{kind.upper()} '{label}'",
            )

    return SafetyCheckResult(requires_confirmation=False, action_summary=signature)


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
