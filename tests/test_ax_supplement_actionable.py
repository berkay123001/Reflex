"""The accessibility-tree supplement only adds elements that can be acted on.

Online-Mind2Web full300 (2026-11-04, booking.com): the children count is a hidden
<input type="range"> behind visible +/- buttons. The AX supplement offered it as "Children" (slider);
its staleness guard is always null, so every action on it failed with StalePage and the idle guard
ended the task after six decisions.
"""

from __future__ import annotations

import shutil

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser

PAGE = """data:text/html,<!DOCTYPE html><html><body>
  <div>Children
    <input type="range" id="group_children" aria-label="Children" min="0" max="10" value="0"
           style="opacity:0;position:absolute;left:10px;top:40px;width:120px;height:20px">
    <button aria-label="Increase number of Children">+</button>
  </div>
</body></html>"""


@pytest.fixture(autouse=True)
def _own_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another test may leave BU_CDP_URL pointing at the reserved shared port; these tests start their own browser."""
    monkeypatch.delenv("BU_CDP_URL", raising=False)


def _chromium() -> bool:
    return bool(shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome"))


def test_hidden_range_is_not_offered_but_its_button_is() -> None:
    if not _chromium():
        pytest.skip("Chromium required for isolated browser test")
    with managed_isolated_browser(port=0):
        browser = Browser(PAGE, ax_supplement=True)
        try:
            page = browser.observe()
            roles = [(a.get("role"), a.get("label")) for a in page["actions"] if a.get("node") is not None]
            assert ("slider", "Children") not in roles, roles
            assert any("Increase number of Children" in str(lbl) for _, lbl in roles), roles
            for a in page["actions"]:
                if a.get("is_ax_supplement"):
                    assert (page.get("guards") or {}).get(str(a["node"])), a
        finally:
            browser.close()
