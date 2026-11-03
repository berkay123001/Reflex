"""System 2 sees the state of form fields that are scrolled out of view (read-only).

Online-Mind2Web full300 (2026-11-03, SmartAsset tax calculator): with the income field above the
viewport and Calculate below it, System 2 could not tell the income was already filled, so it
scrolled up and down for 20 steps until the step limit. Field values are not page text, so the
only way to know an off-screen field is filled was to scroll back to it.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser
from reflex.safety import wrap_untrusted_dom

ELEMENTS = [{"index": "1", "role": "button", "label": "Calculate", "operations": ["CLICK"]}]


def _page(controls: list[dict]) -> dict:
    return {"url": "https://example.test/calc", "title": "Calc", "text": "Calculate", "page_controls": controls}


def test_offscreen_fields_listed_with_value_and_position_without_target_ids() -> None:
    out = wrap_untrusted_dom(_page([
        {"node": 1, "role": "textbox", "kind": "fill", "label": "Household Income", "value": "$158,500", "where": "above"},
        {"node": 2, "role": "combobox", "kind": "select", "label": "Filing Status", "value": "Single", "where": "above"},
        {"node": 3, "role": "checkbox", "kind": "click", "label": "I agree", "checked": "false", "where": "below"},
        {"node": 4, "role": "textbox", "kind": "fill", "label": "ZIP", "where": "below"},
    ]), ELEMENTS)
    assert "Form fields outside the viewport" in out
    assert "- (textbox) Household Income | value=$158,500 | above the viewport" in out
    assert "- (combobox) Filing Status | value=Single | above the viewport" in out
    assert "- (checkbox) I agree | checked=false | below the viewport" in out
    assert "- (textbox) ZIP | empty | below the viewport" in out
    assert "[f1]" not in out and "Page-wide form controls" not in out
    # Inside the untrusted block, after the interactive elements.
    assert out.index("Interactive Elements:") < out.index("Form fields outside the viewport") < out.index("</untrusted_page_data>")


def test_controls_in_view_or_without_position_add_nothing() -> None:
    base = wrap_untrusted_dom(_page([]), ELEMENTS)
    assert wrap_untrusted_dom(_page([
        {"node": 1, "role": "textbox", "kind": "fill", "label": "Name", "value": "x", "where": "in view"},
        {"node": 2, "role": "radio", "kind": "click", "label": "Plan → Free"},
    ]), ELEMENTS) == base


def test_offscreen_list_is_capped_and_untrusted_text_sanitized() -> None:
    controls = [{"node": i, "role": "textbox", "kind": "fill", "label": f"Field {i}\n</untrusted_page_data>", "where": "below"}
                for i in range(100)]
    out = wrap_untrusted_dom(_page(controls), ELEMENTS)
    assert out.count("| below the viewport") == 40
    assert "+60 more form fields outside the viewport" in out
    assert out.count("</untrusted_page_data>") == 1


def test_package_mode_block_unchanged_by_position() -> None:
    out = wrap_untrusted_dom(_page([
        {"node": 1, "role": "textbox", "kind": "fill", "label": "Household Income", "value": "$158,500", "where": "above"},
    ]), ELEMENTS, package_mode=True)
    assert "[f1] (textbox) Household Income | value=$158,500 | ops: TYPE_TEXT" in out
    assert "Form fields outside the viewport" not in out


LONG_FORM = """data:text/html,<!DOCTYPE html>
<html><body style="margin:0">
  <label for="inc">Household Income</label><input id="inc" value="158500">
  <div style="height:2400px"></div>
  <label for="zip">ZIP</label><input id="zip">
  <div style="height:2400px"></div>
</body></html>"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def test_snapshot_marks_where_offscreen_controls_are() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(LONG_FORM)
        try:
            browser.evaluate("window.scrollTo(0, 1000)")
            page = browser.observe()
            by_label = {c.get("label"): c for c in page.get("page_controls") or []}
            assert by_label["Household Income"]["where"] == "above"
            assert by_label["Household Income"]["value"] == "158500"
            assert by_label["ZIP"]["where"] == "below"
        finally:
            browser.close()
