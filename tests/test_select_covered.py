"""A dropdown hidden behind a panel is a failed step the agent can recover from, not a crashed task.

Online-Mind2Web full300 (2026-11-03): a filters panel opened over a <select>; act() raised a bare
RuntimeError, which no step handler catches, so the whole task ended as a runner error.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser, TargetCoveredError
from reflex.browser.isolated import managed_isolated_browser

PAGE = """data:text/html,<!DOCTYPE html>
<html>
<body style="margin:0">
  <label for="sort">Sort By</label>
  <select id="sort" style="margin:40px">
    <option value="date">Date</option>
    <option value="pay">Pay</option>
  </select>
</body>
</html>"""

OPEN_PANEL = """(() => {
  const p = document.createElement('div');
  p.setAttribute('role', 'dialog');
  p.textContent = 'Filters';
  p.style.cssText = 'position:fixed;inset:0;background:#fff;z-index:9';
  document.body.appendChild(p);
})()"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def test_select_behind_panel_raises_target_covered() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PAGE)
        try:
            page = browser.observe()
            pay = next(a for a in page["actions"] if a.get("kind") == "select" and a.get("value") == "pay")
            browser.evaluate(OPEN_PANEL)
            with pytest.raises(TargetCoveredError, match="Sort By"):
                browser.act(pay, page=page)
            assert browser.evaluate("document.getElementById('sort').value") == "date"
        finally:
            browser.close()


def test_select_with_missing_option_raises_target_covered() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PAGE)
        try:
            page = browser.observe()
            pay = next(a for a in page["actions"] if a.get("kind") == "select" and a.get("value") == "pay")
            browser.evaluate("document.querySelector('option[value=pay]').remove()")
            with pytest.raises(TargetCoveredError):
                browser.act(pay, page=page)
        finally:
            browser.close()
