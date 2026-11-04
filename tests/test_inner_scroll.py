"""Pages that scroll an inner container (app layouts) can be scrolled.

Online-Mind2Web full300 (2026-11-04, Google Finance): the window itself does not scroll there, so the
snapshot offered no "Scroll down" action; every SCROLL_DOWN from System 2 resolved to nothing, and
after six such decisions the idle guard ended the task.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser

PAGE = """data:text/html,<!DOCTYPE html>
<html><body style="margin:0;height:100vh;overflow:hidden">
  <header style="height:60px">Finance</header>
  <main id="m" style="position:absolute;top:60px;bottom:0;left:0;right:0;overflow-y:auto">
    <div style="height:2000px">Quote</div>
    <h2>Top news</h2><a href="https://example.com/n1">Microsoft beats estimates</a>
    <div style="height:400px"></div>
  </main>
</body></html>"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def test_inner_scroller_offers_and_performs_scrolling() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PAGE)
        try:
            page = browser.observe()
            ids = {a["id"] for a in page["actions"]}
            assert "scroll_down" in ids and "scroll_up" not in ids
            for _ in range(6):
                down = next((a for a in page["actions"] if a["id"] == "scroll_down"), None)
                if down is None:
                    break
                browser.act(down, page=page)
                page = browser.observe()
            assert browser.evaluate("document.getElementById('m').scrollTop") > 1000
            after = browser.observe()
            assert any(a["id"] == "scroll_up" for a in after["actions"])
            assert any("Microsoft beats estimates" in str(a.get("label")) for a in after["actions"])
        finally:
            browser.close()


def test_window_scrolling_pages_unchanged() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser("data:text/html,<body><div style='height:3000px'>long</div></body>")
        try:
            ids = [a["id"] for a in browser.observe()["actions"]]
            assert ids.count("scroll_down") == 1 and "scroll_up" not in ids
        finally:
            browser.close()


PANEL_PAGE = """data:text/html,<!DOCTYPE html>
<html><body style="margin:0">
  <div style="height:3000px">Map and listings</div>
  <div id="p" role="dialog" aria-label="Filters" style="position:fixed;top:0;right:0;width:420px;height:100vh;overflow-y:auto;background:white">
    <h2>Filters</h2><div style="height:1500px">Price, beds</div>
    <label><input type="checkbox"> Pool</label>
  </div>
</body></html>"""


def test_scrolling_moves_an_open_side_panel_not_the_page_behind() -> None:
    """full300 2026-11-04 (compass.com): the filters panel scrolls on its own, off the wheel point."""
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PANEL_PAGE)
        try:
            page = browser.observe()
            assert any(a["id"] == "scroll_down" for a in page["actions"])
            for _ in range(5):
                down = next((a for a in page["actions"] if a["id"] == "scroll_down"), None)
                if down is None:
                    break
                browser.act(down, page=page)
                page = browser.observe()
            assert browser.evaluate("document.getElementById('p').scrollTop") > 700
            assert browser.evaluate("window.scrollY") == 0
            assert any("Pool" in str(a.get("label")) for a in page["actions"])
        finally:
            browser.close()


SPLIT_PAGE = """data:text/html,<!DOCTYPE html>
<html><body style="margin:0;height:100vh;overflow:hidden">
  <div style="position:absolute;left:0;top:0;bottom:0;width:62%;overflow:hidden;background:silver">Map</div>
  <div id="list" style="position:absolute;right:0;top:0;bottom:0;width:38%;overflow-y:scroll">
    <div style="height:2000px">Results</div><a href="https://example.com/h">Home with pool</a>
  </div>
</body></html>"""


def test_split_layout_scrolls_the_result_list_beside_a_map() -> None:
    """full300 2026-11-04 (compass.com): a map fills the middle of the screen, the results scroll on the right."""
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(SPLIT_PAGE)
        try:
            page = browser.observe()
            assert any(a["id"] == "scroll_down" for a in page["actions"])
            for _ in range(6):
                down = next((a for a in page["actions"] if a["id"] == "scroll_down"), None)
                if down is None:
                    break
                browser.act(down, page=page)
                page = browser.observe()
            assert browser.evaluate("document.getElementById('list').scrollTop") > 1000
            assert any("Home with pool" in str(a.get("label")) for a in page["actions"])
        finally:
            browser.close()
