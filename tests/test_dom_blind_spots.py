"""Tests for Milestone 1 / Requirement R1b: snapshot.js DOM Blind Spots.

Verifies:
1. Input types (date, time, datetime-local, month, week, range) recognized in snapshot.
2. Value setting for date, time, datetime-local, month, week, range in act() via native property setter descriptor.
3. Extended ARIA roles (slider, listbox, menuitemcheckbox, treeitem) recognized.
4. Open shadowRoot interactive elements and text inspected.
5. Closed shadowRoot remains uninspected.
6. Reproduction of pointerRoot swallowing bug and verification of the fix.
7. Neutralization of target="_blank" on form submissions to remain in the same tab.
"""

from __future__ import annotations

import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from reflex.browser.cdp import Browser
from reflex.browser.isolated import managed_isolated_browser


def is_chromium_available() -> bool:
    """Check if Chromium or Chrome binary is present in PATH."""
    return bool(
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )


def test_input_types_recognized_in_snapshot() -> None:
    """Verify that date, time, datetime-local, month, week, and range inputs are discovered with correct roles."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>Input Types Test</title></head>
<body>
  <h1>HTML5 Input Types</h1>
  <input id="i_date" type="date" value="2026-11-02" aria-label="Appointment Date" />
  <input id="i_time" type="time" value="14:30" aria-label="Meeting Time" />
  <input id="i_dt" type="datetime-local" value="2026-11-02T14:30" aria-label="Flight Departure" />
  <input id="i_month" type="month" value="2026-11" aria-label="Billing Month" />
  <input id="i_week" type="week" value="2026-W45" aria-label="Release Week" />
  <input id="i_range" type="range" min="0" max="100" value="65" aria-label="Volume Level" />
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # date, time, datetime-local, month, week -> role='textbox'
            date_act = next((a for a in actions if a.get("label") == "Appointment Date"), None)
            assert date_act is not None, "Date input must be listed"
            assert date_act["role"] == "textbox"
            assert date_act.get("value") == "2026-11-02"

            time_act = next((a for a in actions if a.get("label") == "Meeting Time"), None)
            assert time_act is not None, "Time input must be listed"
            assert time_act["role"] == "textbox"
            assert time_act.get("value") == "14:30"

            dt_act = next((a for a in actions if a.get("label") == "Flight Departure"), None)
            assert dt_act is not None, "datetime-local input must be listed"
            assert dt_act["role"] == "textbox"
            assert dt_act.get("value") == "2026-11-02T14:30"

            month_act = next((a for a in actions if a.get("label") == "Billing Month"), None)
            assert month_act is not None, "Month input must be listed"
            assert month_act["role"] == "textbox"
            assert month_act.get("value") == "2026-11"

            week_act = next((a for a in actions if a.get("label") == "Release Week"), None)
            assert week_act is not None, "Week input must be listed"
            assert week_act["role"] == "textbox"
            assert week_act.get("value") == "2026-W45"

            # range -> role='slider'
            range_act = next((a for a in actions if a.get("label") == "Volume Level"), None)
            assert range_act is not None, "Range input must be listed"
            assert range_act["role"] == "slider"
            assert range_act.get("value") == "65"
        finally:
            browser.close()


def test_act_fill_date_time_range_inputs() -> None:
    """Verify that act() sets values for date, time, and range inputs using the native setter."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <input id="d" type="date" value="2026-01-01" aria-label="Target Date"
         onchange="document.getElementById('d_out').textContent=this.value" />
  <div id="d_out">idle</div>

  <input id="t" type="time" value="08:00" aria-label="Target Time"
         onchange="document.getElementById('t_out').textContent=this.value" />
  <div id="t_out">idle</div>

  <input id="r" type="range" min="0" max="100" value="20" aria-label="Target Range"
         onchange="document.getElementById('r_out').textContent=this.value" />
  <div id="r_out">idle</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()

            # 1. Fill date input
            d_act = next(a for a in page["actions"] if a.get("label") == "Target Date")
            browser.act(d_act, page=page, text="2027-05-20")
            d_val = browser.evaluate("document.getElementById('d').value")
            assert d_val == "2027-05-20", f"Expected date '2027-05-20', got '{d_val}'"
            d_out = browser.evaluate("document.getElementById('d_out').textContent")
            assert d_out == "2027-05-20", "onchange event must fire for date input"

            # 2. Fill time input
            t_act = next(a for a in page["actions"] if a.get("label") == "Target Time")
            browser.act(t_act, page=page, text="16:45")
            t_val = browser.evaluate("document.getElementById('t').value")
            assert t_val == "16:45", f"Expected time '16:45', got '{t_val}'"
            t_out = browser.evaluate("document.getElementById('t_out').textContent")
            assert t_out == "16:45", "onchange event must fire for time input"

            # 3. Fill range input
            r_act = next(a for a in page["actions"] if a.get("label") == "Target Range")
            browser.act(r_act, page=page, text="85")
            r_val = browser.evaluate("document.getElementById('r').value")
            assert r_val == "85", f"Expected range '85', got '{r_val}'"
            r_out = browser.evaluate("document.getElementById('r_out').textContent")
            assert r_out == "85", "onchange event must fire for range input"
        finally:
            browser.close()


def test_aria_roles_in_snapshot() -> None:
    """Verify that slider, listbox, menuitemcheckbox, and treeitem roles are captured in snapshot.js."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <h1>ARIA Roles</h1>
  <div role="slider" aria-valuenow="50" aria-valuemin="0" aria-valuemax="100" aria-label="Equalizer Bass" style="display:block;width:100px;height:20px;"></div>

  <div role="listbox" aria-label="Select Country">
    <div role="option" aria-selected="true">Turkey</div>
  </div>

  <div role="menu">
    <div role="menuitemcheckbox" aria-checked="true" aria-label="Enable Notifications">Enable Notifications</div>
  </div>

  <div role="tree">
    <div role="treeitem" aria-label="Source Folder">Source Folder</div>
  </div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            slider = next((a for a in actions if a.get("label") == "Equalizer Bass"), None)
            assert slider is not None, "role='slider' must be listed"
            assert slider["role"] == "slider"

            listbox = next((a for a in actions if a.get("label") == "Select Country"), None)
            assert listbox is not None, "role='listbox' must be listed"
            assert listbox["role"] == "listbox"

            menu_chk = next((a for a in actions if a.get("label") == "Enable Notifications"), None)
            assert menu_chk is not None, "role='menuitemcheckbox' must be listed"
            assert menu_chk["role"] == "menuitemcheckbox"
            assert menu_chk.get("checked") == "true"

            tree_item = next((a for a in actions if a.get("label") == "Source Folder"), None)
            assert tree_item is not None, "role='treeitem' must be listed"
            assert tree_item["role"] == "treeitem"
        finally:
            browser.close()


def test_open_shadow_root_observation() -> None:
    """Verify that interactive elements and text inside open shadow roots are inspected."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <h1>Outer Document</h1>
  <div id="host_el"></div>
  <script>
    const host = document.getElementById("host_el");
    const shadow = host.attachShadow({mode: "open"});
    shadow.innerHTML = `
      <p>Shadow Paragraph Content</p>
      <button id="shadow_btn" onclick="this.getRootNode().getElementById('shadow_out').textContent='shadow_clicked'">
        Shadow Action Button
      </button>
      <input id="shadow_inp" type="text" value="shadow_val" aria-label="Shadow Input" />
      <div id="shadow_out">shadow_idle</div>
    `;
  </script>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()

            # 1. Verify text extraction includes shadow DOM text
            assert "Shadow Paragraph Content" in page.get("text", "")

            # 2. Verify interactive button in shadow DOM is listed
            btn_act = next((a for a in page["actions"] if a.get("label") == "Shadow Action Button"), None)
            assert btn_act is not None, "Button inside open shadowRoot must be listed in actions"

            # 3. Verify input in shadow DOM is listed
            inp_act = next((a for a in page["actions"] if a.get("value") == "shadow_val"), None)
            assert inp_act is not None, "Input inside open shadowRoot must be listed in actions"

            # 4. Verify act() can click the shadow button
            browser.act(btn_act, page=page)
            time.sleep(0.2)
            out_text = browser.evaluate(
                "document.getElementById('host_el').shadowRoot.getElementById('shadow_out').textContent"
            )
            assert out_text == "shadow_clicked", "Click on shadow button must execute its click handler"
        finally:
            browser.close()


def test_closed_shadow_root_remains_uninspected() -> None:
    """Verify that elements inside closed shadow roots are not inspected."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <div id="closed_host"></div>
  <script>
    const host = document.getElementById("closed_host");
    const shadow = host.attachShadow({mode: "closed"});
    shadow.innerHTML = `<button id="closed_btn">Secret Closed Button</button>`;
  </script>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            labels = [a.get("label") for a in page["actions"]]
            assert "Secret Closed Button" not in labels, "Closed shadow roots must not be inspected"
        finally:
            browser.close()


def test_pointer_root_swallowing_reproduction_and_fix() -> None:
    """Reproduction and verification test for the pointerRoot swallowing bug.

    When an outer card element has cursor:pointer and descriptive text > 120 chars,
    it is rejected. Child clickable divs with cursor:pointer must NOT be swallowed.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    # Card with cursor:pointer and text length ~ 190 characters (> 120)
    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <div id="card" style="cursor: pointer; padding: 20px; width: 450px; border: 1px solid gray;">
    <p>
      This is a comprehensive overview of service terms and general policy guidelines that easily exceeds
      the one hundred and twenty character limit enforced by the control label length check in snapshot.js.
    </p>
    <div id="btn_alpha" style="cursor: pointer; background: rgb(238,238,238); margin: 10px; padding: 10px;"
         onclick="document.getElementById('res').textContent='alpha_chosen'">
      Accept Alpha Plan
    </div>
    <div id="btn_beta" style="cursor: pointer; background: rgb(238,238,238); margin: 10px; padding: 10px;"
         onclick="document.getElementById('res').textContent='beta_chosen'">
      Accept Beta Plan
    </div>
  </div>
  <div id="res">idle</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # The child options must NOT be swallowed
            alpha_act = next((a for a in actions if a.get("label") == "Accept Alpha Plan"), None)
            assert alpha_act is not None, "Child clickable option 'Accept Alpha Plan' must not be swallowed"
            assert alpha_act["role"] == "button"

            beta_act = next((a for a in actions if a.get("label") == "Accept Beta Plan"), None)
            assert beta_act is not None, "Child clickable option 'Accept Beta Plan' must not be swallowed"
            assert beta_act["role"] == "button"

            # Execute click on child option to verify actionability
            browser.act(alpha_act, page=page)
            time.sleep(0.2)
            res_text = browser.evaluate("document.getElementById('res').textContent")
            assert res_text == "alpha_chosen", "Clicking child pointer option must trigger click handler"
        finally:
            browser.close()


def test_form_target_blank_neutralization() -> None:
    """Verify that form submission target='_blank' and button formtarget='_blank' are stripped in act()."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    class FormHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if "/submitted" in self.path:
                self.wfile.write(b"<!DOCTYPE html><html><body><h1>Form Submitted Same Tab</h1></body></html>")
            else:
                self.wfile.write(b"""<!DOCTYPE html>
<html>
<body>
  <form id="f1" action="/submitted" method="GET" target="_blank">
    <input type="text" name="q" value="sample_query" />
    <button id="sub_btn" type="submit">Submit Blank Form</button>
  </form>
</body>
</html>""")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), FormHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        with managed_isolated_browser(port=0) as cdp_url:
            browser = Browser(f"http://127.0.0.1:{server_port}/")
            try:
                page = browser.observe()
                sub_act = next(a for a in page["actions"] if a.get("label") == "Submit Blank Form")

                # Store existing target count
                init_targets = set(browser._owned_targets)

                # Act on submit button
                browser.act(sub_act, page=page)
                time.sleep(0.5)

                # Verification:
                # 1. Form submission must NOT create a new tab/target
                current_targets = set(browser._owned_targets)
                assert len(current_targets) == len(init_targets), "No new browser targets should be spawned"

                # 2. Navigation must happen in the current browsing context
                after_page = browser.observe()
                assert "Form Submitted Same Tab" in after_page.get("text", "")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
