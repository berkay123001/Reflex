"""System 2 is told when a dialog covers the page (snapshot + prompt).

Online-Mind2Web full300 (2026-11-04, BabyCenter height predictor): a sign-up dialog covered the
calculator, so its fields were not actionable; System 2 only saw page text, never closed the dialog,
and scrolled up and down for 20 steps looking for the inputs.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser
from reflex.safety import wrap_untrusted_dom

FORM = """<label for="h">Child's height</label><input id="h"><button>Calculate</button>"""
MODAL = """<div class="signup-modal" role="dialog" aria-modal="true" style="position:fixed;inset:80px;background:white;z-index:9">
  <h2>Join BabyCenter</h2><input placeholder="Email"><button>Join now</button></div>"""
OVERLAY = """<div class="newsletter-overlay" style="position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:9">
  <div style="margin:200px auto;width:400px;background:white"><h3>Get our newsletter</h3><button>No thanks</button></div></div>"""
PICKER = """<div role="dialog" aria-label="Choose date" style="position:fixed;left:400px;top:250px;width:300px;height:300px;background:white">
  <button>1</button><button>2</button></div>"""
BANNER = """<div role="dialog" style="position:fixed;left:0;right:0;bottom:0;height:80px;background:white">We use cookies
  <button>Accept</button></div>"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def _observe(body: str) -> dict:
    with managed_isolated_browser(port=0):
        browser = Browser(f"data:text/html,<!DOCTYPE html><html><body>{FORM}{body}</body></html>")
        try:
            return browser.observe()
        finally:
            browser.close()


def test_snapshot_reports_dialogs_over_the_middle_of_the_screen() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    assert _observe(MODAL).get("blocking_dialog") == "Join BabyCenter"
    assert _observe(OVERLAY).get("blocking_dialog") == "Get our newsletter"
    assert not _observe(BANNER).get("blocking_dialog")
    assert not _observe(PICKER).get("blocking_dialog")  # a date picker the agent opened itself
    assert not _observe("").get("blocking_dialog")


def test_prompt_tells_system2_to_close_the_dialog_first() -> None:
    page = {"url": "https://example.test/", "title": "t", "text": "x", "blocking_dialog": "Join BabyCenter"}
    out = wrap_untrusted_dom(page, [{"index": "1", "role": "button", "label": "Join now", "operations": ["CLICK"]}])
    assert "Open dialog over the page: Join BabyCenter" in out
    assert "PRESS_ESCAPE" in out
    assert out.index("Open dialog") < out.index("</untrusted_page_data>")
    plain = wrap_untrusted_dom(dict(page, blocking_dialog=""), [])
    assert "Open dialog" not in plain
