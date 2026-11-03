"""Form fields that share a label get their own field caption and placeholder.

Online-Mind2Web full300 (2026-11-04, craigslist): the bedroom and bathroom min/max boxes all had the
label "whole number, no letters or symbols (apartments / housing for rent)", so System 2 typed the
bathroom minimum into the bedroom maximum box.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser

PAGE = """data:text/html,<!DOCTYPE html>
<html><body>
  <h1>apartments / housing for rent</h1>
  <form>
    <div class="filter"><div class="filter-title">BEDROOMS</div>
      <input title="whole number, no letters or symbols" placeholder="min">
      <input title="whole number, no letters or symbols" placeholder="max">
    </div>
    <div class="filter"><div class="filter-title">BATHROOMS</div>
      <input title="whole number, no letters or symbols" placeholder="min">
      <input title="whole number, no letters or symbols" placeholder="max">
    </div>
    <label for="q">Search</label><input id="q">
  </form>
</body></html>"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def test_repeated_min_max_boxes_get_distinct_labels() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PAGE)
        try:
            page = browser.observe()
            labels = [a["label"] for a in page["actions"] if a.get("kind") == "fill"]
            boxes = [lbl for lbl in labels if "whole number" in lbl]
            assert len(boxes) == 4 and len(set(boxes)) == 4, boxes
            assert "BEDROOMS" in boxes[0] and "min" in boxes[0]
            assert "BEDROOMS" in boxes[1] and "max" in boxes[1]
            assert "BATHROOMS" in boxes[2] and "min" in boxes[2]
            assert "BATHROOMS" in boxes[3] and "max" in boxes[3]
            assert "Search" in labels and all("BEDROOMS" not in lbl for lbl in labels if lbl.startswith("Search"))
        finally:
            browser.close()
