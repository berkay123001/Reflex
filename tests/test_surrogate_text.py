"""Unit tests for R5: Unicode surrogate sanitization in browser observation."""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx

import reflex.browser
from reflex.browser import Browser, clean_surrogates
from reflex.safety import wrap_untrusted_dom


def _bare_browser() -> Browser:
    """Create a minimal Browser instance without launching a real Chrome process."""
    b = Browser.__new__(Browser)
    b.session = "s1"
    b.target = "t1"
    b._cdp_lock = threading.Lock()
    b._native_dialogs = []
    b.last_action_rect = None
    b.last_jpeg_bytes = None
    b.after_input = None
    b.evaluate = lambda expr: {}  # type: ignore[method-assign]
    b._dismiss_js_dialog = lambda: None  # type: ignore[method-assign]
    return b


def test_clean_surrogates_lone_high_surrogate() -> None:
    """Lone UTF-16 high surrogates (e.g. \\ud83e) are stripped."""
    assert clean_surrogates("\ud83e") == ""
    assert clean_surrogates("Prefix \ud83e Suffix") == "Prefix  Suffix"


def test_clean_surrogates_lone_low_surrogate() -> None:
    """Lone UTF-16 low surrogates (e.g. \\udd20) are stripped."""
    assert clean_surrogates("\udd20") == ""
    assert clean_surrogates("Left \udd20 Right") == "Left  Right"


def test_clean_surrogates_paired_surrogates_recombined() -> None:
    """Valid surrogate pairs are properly recombined into full Unicode characters."""
    assert clean_surrogates("\ud83e\udd20") == "🤠"
    assert clean_surrogates("Hello \ud83e\udd20 World") == "Hello 🤠 World"


def test_clean_surrogates_mixed_paired_and_unpaired() -> None:
    """Paired surrogates are recombined while lone surrogates are dropped."""
    assert clean_surrogates("\ud83e\udd20\ud83e") == "🤠"
    assert clean_surrogates("\ud83d\udc4d and \ud83e then \ud83d\udc4e") == "👍 and  then 👎"


def test_clean_surrogates_valid_text_untouched() -> None:
    """Regular UTF-8 characters and Turkish text remain untouched."""
    text = "Türkçe karakterler: çığıöşü ÇİĞÖŞÜ - Özel simgeler: !@#$%^&*() - Emojiler: 🚀✨🎉"
    assert clean_surrogates(text) == text


def test_clean_surrogates_primitives_untouched() -> None:
    """Non-string primitives pass through untouched."""
    assert clean_surrogates(123) == 123
    assert clean_surrogates(45.67) == 45.67
    assert clean_surrogates(True) is True
    assert clean_surrogates(False) is False
    assert clean_surrogates(None) is None


def test_clean_surrogates_deeply_nested_structures() -> None:
    """Sanitizes strings in nested dictionaries (keys and values), lists, and tuples."""
    nested: dict[str, Any] = {
        "key_\ud83e": "val_\ud83e",
        "list": ["item_\ud83e", {"inner_\ud83e": "inner_val_\ud83e"}],
        "tuple": ("t1_\ud83e", 42, ("t2_\ud83e\udd20", None)),
        "number": 100,
        "clean": "regular text",
    }
    cleaned = clean_surrogates(nested)
    assert cleaned == {
        "key_": "val_",
        "list": ["item_", {"inner_": "inner_val_"}],
        "tuple": ("t1_", 42, ("t2_🤠", None)),
        "number": 100,
        "clean": "regular text",
    }


def test_browser_observe_sanitizes_unpaired_surrogates() -> None:
    """Browser.observe() sanitizes unpaired surrogates from evaluate(READ_STATE)."""
    b = _bare_browser()
    raw_state: dict[str, Any] = {
        "url": "https://example.com/search?q=\ud83e",
        "title": "Title \ud83e",
        "text": "Body \ud83e text",
        "full_text": "Full body \ud83e text",
        "actions": [
            {
                "label": "Btn \ud83e",
                "node": "1",
                "role": "button",
                "href": "https://example.com/\ud83e",
            }
        ],
    }
    b.evaluate = lambda expr: dict(raw_state)  # type: ignore[method-assign]
    obs = b.observe(screenshot=False)

    assert obs["url"] == "https://example.com/search?q="
    assert obs["title"] == "Title "
    assert obs["text"] == "Body  text"
    assert obs["full_text"] == "Full body  text"
    assert obs["actions"][0]["label"] == "Btn "
    assert obs["actions"][0]["href"] == "https://example.com/"
    assert "fingerprint" in obs
    assert isinstance(obs["fingerprint"], str)
    assert len(obs["fingerprint"]) == 64


def test_s2_request_body_encodes_cleanly_with_surrogates() -> None:
    """Page observation with half-cut emojis wrapped in wrap_untrusted_dom serializes cleanly."""
    # Mimics Trendyol kulaklik_platform observation that previously crashed S2
    raw_page: dict[str, Any] = {
        "url": "https://www.trendyol.com/sr?q=kablolu+",
        "title": "Kablolu Kulaklık Modelleri ve Fiyatları \ud83e",
        "text": "Kablolu Kulaklık \ud83e indirimde",
        "full_text": "Tüm ürünler ve fırsatlar \ud83e detaylar",
        "actions": [
            {
                "label": "Kablolu Kulaklık \ud83e Sepete Ekle",
                "node": "1",
                "role": "button",
                "rect": [10, 10, 50, 20],
            },
            {
                "label": "Filtrele \ud83e\udd20",
                "node": "2",
                "role": "button",
                "rect": [10, 40, 50, 20],
            },
        ],
    }

    clean_page = clean_surrogates(raw_page)
    prompt = wrap_untrusted_dom(clean_page, clean_page["actions"])

    payload = {
        "model": "muse-spark-1.3-contributor",
        "messages": [
            {"role": "system", "content": "You are a web navigation supervisor."},
            {"role": "user", "content": prompt},
        ],
    }

    # Verify json.dumps with ensure_ascii=False encodes to UTF-8 without UnicodeEncodeError
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert b"\\ud83e" not in encoded
    assert "🤠".encode() in encoded

    # Verify httpx.Request (used by S2 supervisor) serializes JSON without error
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload)
    assert req.content is not None
    assert "🤠".encode() in req.content


def test_clean_surrogates_exported_in_reflex_browser() -> None:
    """clean_surrogates is exported in reflex.browser and reflex.browser.__all__."""
    assert hasattr(reflex.browser, "clean_surrogates")
    assert "clean_surrogates" in reflex.browser.__all__
    assert reflex.browser.clean_surrogates is clean_surrogates
