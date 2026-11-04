"""Safety guards for irreversible browser actions and untrusted DOM isolation."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel

from .config import MAX_SUPERVISOR_ELEMENTS

MAX_PAGE_CONTROLS = 150
MAX_CONTROL_LABEL_LENGTH = 140
MAX_OFFSCREEN_FIELDS = 40


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

_URL_QUERY_RE = re.compile(r"(https?://[^\s?#]+)[?#]\S*", re.IGNORECASE)
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
    if kind == "goto_url":
        # Query strings carry search parameters ("&checkout=2026-11-07" on hotel sites), not actions.
        texts_to_check = [_URL_QUERY_RE.sub(r"\1", t) for t in texts_to_check]

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


def _normalize_turkish_fold(s: str) -> str:
    """Casefold string with Turkish dotted/dotless I normalization."""
    return s.replace("I", "ı").replace("İ", "i").casefold().replace("\u0307", "")


def _extract_focus_prefixes(focus: str) -> list[str]:
    """Extract distinct prefix stems (first 5 letters, or 4 for 4-letter words) from focus string.

    Tokens with length < 4 are ignored.
    """
    norm = _normalize_turkish_fold(focus)
    tokens = re.findall(r"\w+", norm)
    prefixes: list[str] = []
    for t in tokens:
        if len(t) >= 4:
            prefixes.append(t[:5])
    return list(dict.fromkeys(prefixes))


def _chunk_paragraph_text(text: str, chunk_size: int = 600) -> list[str]:
    """Split text into ~600 character paragraph chunks respecting natural boundaries."""
    if not text:
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_paras = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    if not raw_paras:
        raw_paras = [p.strip() for p in normalized.split("\n") if p.strip()]
    if not raw_paras:
        raw_paras = [normalized.strip()] if normalized.strip() else []

    chunks: list[str] = []
    curr: list[str] = []
    curr_len = 0

    def flush_curr() -> None:
        nonlocal curr, curr_len
        if curr:
            chunks.append("\n\n".join(curr))
            curr = []
            curr_len = 0

    for para in raw_paras:
        if len(para) <= chunk_size:
            if curr_len + len(para) + 2 <= chunk_size:
                curr.append(para)
                curr_len += len(para) + (2 if len(curr) > 1 else 0)
            else:
                flush_curr()
                curr.append(para)
                curr_len = len(para)
        else:
            flush_curr()
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            for line in lines:
                if len(line) <= chunk_size:
                    if curr_len + len(line) + 1 <= chunk_size:
                        curr.append(line)
                        curr_len += len(line) + (1 if len(curr) > 1 else 0)
                    else:
                        flush_curr()
                        curr.append(line)
                        curr_len = len(line)
                else:
                    flush_curr()
                    words = line.split(" ")
                    w_curr: list[str] = []
                    w_len = 0
                    for w in words:
                        if not w:
                            continue
                        if len(w) > chunk_size:
                            if w_curr:
                                chunks.append(" ".join(w_curr))
                                w_curr = []
                                w_len = 0
                            for i in range(0, len(w), chunk_size):
                                chunks.append(w[i : i + chunk_size])
                        elif w_len + len(w) + 1 <= chunk_size:
                            w_curr.append(w)
                            w_len += len(w) + 1
                        else:
                            chunks.append(" ".join(w_curr))
                            w_curr = [w]
                            w_len = len(w)
                    if w_curr:
                        chunks.append(" ".join(w_curr))
            flush_curr()
    flush_curr()
    return chunks


def _score_chunk(chunk: str, prefixes: list[str]) -> tuple[int, int]:
    """Score chunk returning (distinct_terms_count, total_matches_count).

    distinct_terms_count: number of distinct focus prefixes matched in this chunk.
    total_matches_count: total number of word-prefix matches in this chunk.
    """
    if not prefixes or not chunk:
        return (0, 0)
    norm_chunk = _normalize_turkish_fold(chunk)
    words = [w for w in re.findall(r"\w+", norm_chunk) if len(w) >= 4]
    if not words:
        return (0, 0)
    matched_prefixes: set[str] = set()
    total_matches = 0
    for w in words:
        for p in prefixes:
            if w.startswith(p) or p.startswith(w):
                matched_prefixes.add(p)
                total_matches += 1
    return (len(matched_prefixes), total_matches)


def _filter_prefixes_by_df(chunks: list[str], prefixes: list[str]) -> list[str]:
    """Filter out focus prefixes that appear in > 50% of candidate chunks (DF > len(chunks) / 2).

    Language-agnostic dynamic document frequency filter (strictly no fixed stopwords).
    If len(chunks) <= 1, no prefixes are excluded.
    """
    if not prefixes or len(chunks) <= 1:
        return list(prefixes)

    threshold = len(chunks) / 2
    tokenized_chunks = [
        [w for w in re.findall(r"\w+", _normalize_turkish_fold(c)) if len(w) >= 4]
        for c in chunks
    ]
    active_prefixes: list[str] = []
    for p in prefixes:
        df = sum(1 for words in tokenized_chunks if any(w.startswith(p) or p.startswith(w) for w in words))
        if df <= threshold:
            active_prefixes.append(p)
    return active_prefixes


def extract_focused_full_text(raw_full: str, focus: str = "", max_chars: int = 7500) -> str:
    """Extract focused middle paragraphs when full_text exceeds max_chars.

    If focus is empty/whitespace or yields no valid terms, legacy truncation
    (4200 head + 3000 tail) is used.
    """
    if len(raw_full) <= max_chars:
        return raw_full

    clean_focus = str(focus or "").strip()
    if not clean_focus:
        return raw_full[:4200] + "\n...[middle section omitted]...\n" + raw_full[-3000:]

    prefixes = _extract_focus_prefixes(clean_focus)
    if not prefixes:
        return raw_full[:4200] + "\n...[middle section omitted]...\n" + raw_full[-3000:]

    head = raw_full[:2000]
    tail = raw_full[-1000:]
    middle = raw_full[2000:-1000]

    chunks = _chunk_paragraph_text(middle, chunk_size=600)
    if not chunks:
        return head + "\n[... omitted ...]\n" + tail

    active_prefixes = _filter_prefixes_by_df(chunks, prefixes)
    if not active_prefixes:
        return head + "\n[... omitted ...]\n" + tail

    scored_chunks = []
    for idx, c in enumerate(chunks):
        distinct_cnt, total_cnt = _score_chunk(c, active_prefixes)
        if distinct_cnt > 0:
            scored_chunks.append((idx, distinct_cnt, total_cnt, c))

    # Multi-tier ranking:
    # 1. Primary: distinct matched terms count descending (-x[1])
    # 2. Secondary: total match count descending (-x[2])
    # 3. Tertiary: original chunk order ascending (x[0])
    ranked = sorted(scored_chunks, key=lambda x: (-x[1], -x[2], x[0]))

    if not ranked:
        return head + "\n[... omitted ...]\n" + tail

    selected_indices: set[int] = set()
    total_len = 0
    for idx, distinct_cnt, total_cnt, c in ranked:
        if total_len + len(c) <= 4000:
            selected_indices.add(idx)
            total_len += len(c)
        elif not selected_indices:
            max_c = max(100, max_chars - len(head) - len(tail) - 100)
            c_trimmed = c[:max_c]
            chunks[idx] = c_trimmed
            selected_indices.add(idx)
            total_len += len(c_trimmed)
            break

    if not selected_indices:
        return head + "\n[... omitted ...]\n" + tail

    sorted_idx = sorted(selected_indices)
    parts: list[str] = [head]

    if sorted_idx[0] > 0:
        parts.append("\n[... omitted ...]\n")
    else:
        parts.append("\n\n")

    for i, idx in enumerate(sorted_idx):
        if i > 0:
            prev_idx = sorted_idx[i - 1]
            if idx > prev_idx + 1:
                parts.append("\n[... omitted ...]\n")
            else:
                parts.append("\n\n")
        parts.append(chunks[idx])

    if sorted_idx[-1] < len(chunks) - 1:
        parts.append("\n[... omitted ...]\n")
    else:
        parts.append("\n\n")

    parts.append(tail)

    result = "".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars]
    return result


def _offscreen_fields_block(controls: list[dict[str, Any]]) -> str:
    """Read-only state of form fields scrolled out of view: field values are not page text,
    so without this System 2 cannot tell an off-screen field is already filled."""
    lines: list[str] = []
    offscreen = [c for c in controls if c.get("where") in {"above", "below"}]
    for ctrl in offscreen[:MAX_OFFSCREEN_FIELDS]:
        role = sanitize_untrusted_text(str(ctrl.get("role", "")), 40)
        lbl = sanitize_untrusted_text(str(ctrl.get("label", "")), MAX_CONTROL_LABEL_LENGTH)
        if ctrl.get("checked") not in (None, ""):
            state = f"checked={sanitize_untrusted_text(str(ctrl['checked']), 20)}"
        elif ctrl.get("value") not in (None, ""):
            state = f"value={sanitize_untrusted_text(str(ctrl['value']), 80)}"
        else:
            state = "empty"
        lines.append(f"- ({role}) {lbl} | {state} | {ctrl['where']} the viewport")
    if not lines:
        return ""
    if len(offscreen) > MAX_OFFSCREEN_FIELDS:
        lines.append(f"+{len(offscreen) - MAX_OFFSCREEN_FIELDS} more form fields outside the viewport")
    return (
        "\n\nForm fields outside the viewport (current state only; scroll to reach one before acting on it):\n"
        + "\n".join(lines)
    )


def wrap_untrusted_dom(
    page: dict[str, Any],
    elements: list[dict[str, Any]],
    max_text_chars: int = 3000,
    focus: str = "",
    max_elements: int = MAX_SUPERVISOR_ELEMENTS,
    total_elements: int | None = None,
    *,
    package_mode: bool = False,
) -> str:
    """Format observed DOM elements and page text inside explicit untrusted delimiters."""
    url = sanitize_untrusted_text(str(page.get("url", "")), 500)
    title = sanitize_untrusted_text(str(page.get("title", "")), 300)
    text = sanitize_untrusted_text(str(page.get("text", "")), max_text_chars)
    raw_full = str(page.get("full_text", "") or "")
    if len(raw_full) > 7500:
        if focus and focus.strip():
            raw_full = extract_focused_full_text(raw_full, focus=focus, max_chars=7500)
        else:
            raw_full = raw_full[:4200] + "\n...[middle section omitted]...\n" + raw_full[-3000:]
    full_text = sanitize_untrusted_text(raw_full, 7500 if (focus and focus.strip()) else 7800)
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
    for el in elements[:max_elements]:
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

    raw_actions = [
        a
        for a in (page.get("actions") or [])
        if isinstance(a, dict) and a.get("kind") in {"click", "fill", "select"}
    ]
    interactive_nodes = {a["node"] for a in raw_actions if a.get("node") is not None}
    action_element_count = len(interactive_nodes) if interactive_nodes else len(raw_actions)

    total_candidates = [len(elements), action_element_count]
    if total_elements is not None:
        total_candidates.append(total_elements)
    omitted = max(0, max(total_candidates) - max_elements)
    if omitted > 0:
        elem_lines.append(f"+{omitted} more interactive elements omitted (scroll or use GOTO_URL)")

    elements_block = "\n".join(elem_lines) if elem_lines else "(no interactive elements observed)"

    page_controls_block = ""
    raw_page_controls = page.get("page_controls") or []
    if package_mode and raw_page_controls:
        ctrl_lines: list[str] = []
        for i, ctrl in enumerate(raw_page_controls[:MAX_PAGE_CONTROLS]):
            idx = f"f{i + 1}"
            role = ctrl.get("role", "")
            lbl = sanitize_untrusted_text(str(ctrl.get("label", "")), MAX_CONTROL_LABEL_LENGTH)
            kind = ctrl.get("kind", "click")
            op = "TYPE_TEXT" if kind == "fill" else ("SELECT" if kind == "select" else "CLICK")
            extras = []
            for k in ("value", "checked", "selected"):
                if ctrl.get(k) not in (None, ""):
                    extras.append(f"{k}={sanitize_untrusted_text(str(ctrl[k]), 80)}")
            extra_str = (" | " + ", ".join(extras)) if extras else ""
            ctrl_lines.append(f"[{idx}] ({role}) {lbl}{extra_str} | ops: {op}")
        if ctrl_lines:
            page_controls_block = "\n\nPage-wide form controls (also off-screen):\n" + "\n".join(ctrl_lines)
    elif raw_page_controls:
        page_controls_block = _offscreen_fields_block(raw_page_controls)

    extra_full_block = ""
    if full_text and len(full_text) > len(text) + 40:
        extra_full_block = f"\n\nFull Page Text (Entire Document From Top to Bottom):\n{full_text}"
    headings_block = f"\nAll Page Headings (h1-h4): {headings_str}" if headings_str else ""
    dialog = sanitize_untrusted_text(str(page.get("blocking_dialog") or ""), 90)
    if dialog:
        headings_block += (
            f"\nOpen dialog over the page: {dialog} (elements behind it cannot be used while it is open; "
            "use it if it is part of the task, otherwise close it with its close button or PRESS_ESCAPE)"
        )

    return (
        "<untrusted_page_data>\n"
        f"URL: {url}\n"
        f"Title: {title}\n"
        f"Scroll Position: {scroll_line}{headings_block}\n"
        f"Interactive Elements:\n{elements_block}"
        f"{page_controls_block}\n\n"
        f"Visible Viewport Text:\n{text}{extra_full_block}\n"
        "</untrusted_page_data>"
    )
