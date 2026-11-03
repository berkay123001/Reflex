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


def test_nested_shadow_roots_multi_level_stress() -> None:
    """Stress test: 3-level deeply nested open shadowRoot hierarchy and nested closed shadowRoot.

    Verifies:
    1. Text extraction traverses multi-level nested open shadowRoots.
    2. Elements inside closed shadowRoot inside open shadowRoot remain uninspected.
    3. Deeply nested shadowRoot buttons/inputs are listed in actions.
    4. act() can successfully click buttons and fill inputs within multi-level nested shadowRoots.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <h1>Outer Document Heading</h1>
  <div id="host_l1"></div>
  <script>
    const h1 = document.getElementById("host_l1");
    const s1 = h1.attachShadow({mode: "open"});
    s1.innerHTML = `
      <h2>Shadow Level 1 Heading</h2>
      <p>Shadow Level 1 Text</p>
      <div id="host_l2"></div>
    `;

    const h2 = s1.getElementById("host_l2");
    const s2 = h2.attachShadow({mode: "open"});
    s2.innerHTML = `
      <h3>Shadow Level 2 Heading</h3>
      <p>Shadow Level 2 Text</p>
      <div id="host_l2_closed"></div>
      <div id="host_l3"></div>
    `;

    // Closed shadowRoot nested inside open shadowRoot level 2
    const hClosed = s2.getElementById("host_l2_closed");
    const sClosed = hClosed.attachShadow({mode: "closed"});
    sClosed.innerHTML = `<button id="secret_nested_closed_btn">Secret Nested Closed Button</button>`;

    // Level 3 open shadowRoot
    const h3 = s2.getElementById("host_l3");
    const s3 = h3.attachShadow({mode: "open"});
    s3.innerHTML = `
      <h4>Shadow Level 3 Heading</h4>
      <p>Shadow Level 3 Deep Content</p>
      <button id="deep_action_btn" onclick="this.getRootNode().getElementById('deep_status').textContent='deep_success'">
        Deep Action Button
      </button>
      <input id="deep_text_inp" type="text" value="deep_init" aria-label="Deep Input Box" />
      <div id="deep_status">deep_idle</div>
    `;
  </script>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()

            # 1. Verify text from open shadow levels 1, 2, 3 is present
            page_text = page.get("text", "")
            assert "Shadow Level 1 Text" in page_text
            assert "Shadow Level 2 Text" in page_text
            assert "Shadow Level 3 Deep Content" in page_text

            # 2. Closed shadow root content must NOT be present
            assert "Secret Nested Closed Button" not in page_text
            labels = [a.get("label") for a in page["actions"]]
            assert "Secret Nested Closed Button" not in labels

            # 3. Deep level 3 interactive elements must be in actions
            deep_btn = next((a for a in page["actions"] if a.get("label") == "Deep Action Button"), None)
            assert deep_btn is not None, "Button in 3-level nested open shadowRoot must be listed"

            deep_inp = next((a for a in page["actions"] if a.get("label") == "Deep Input Box"), None)
            assert deep_inp is not None, "Input in 3-level nested open shadowRoot must be listed"
            assert deep_inp.get("value") == "deep_init"

            # 4. act() click on deep level 3 button
            browser.act(deep_btn, page=page)
            time.sleep(0.3)
            status_text = browser.evaluate(
                "document.getElementById('host_l1').shadowRoot.getElementById('host_l2').shadowRoot.getElementById('host_l3').shadowRoot.getElementById('deep_status').textContent"
            )
            assert status_text == "deep_success", "Clicking deeply nested shadow button must trigger its click handler"

            # 5. act() fill on deep level 3 input
            browser.act(deep_inp, page=page, text="deep_modified")
            inp_val = browser.evaluate(
                "document.getElementById('host_l1').shadowRoot.getElementById('host_l2').shadowRoot.getElementById('host_l3').shadowRoot.getElementById('deep_text_inp').value"
            )
            assert inp_val == "deep_modified", "Filling deeply nested shadow input must set its value"
        finally:
            browser.close()


def test_pointer_root_multi_level_and_selector_exclusion_stress() -> None:
    """Stress test: Multi-level pointerRoot hierarchy with selector exclusion.

    Verifies:
    1. Multi-level ancestors with pointer and label > 120 chars are omitted without swallowing children.
    2. Child clickable divs (<= 120 chars) are recognized as custom buttons.
    3. Child container with pointer that wraps an element matching selector is excluded in favor of the inner selector control.
    4. Child custom button actions are clickable via act().
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <!-- Outer card: pointer, length > 120 -->
  <div id="grandparent_card" style="cursor: pointer; padding: 20px; width: 500px; border: 2px solid blue;">
    <p>
      Grandparent outer card text that provides broad context regarding the overall module and easily surpasses the one hundred and twenty character limit enforced by snapshot.js.
    </p>
    <!-- Middle wrapper: pointer, length > 120 -->
    <div id="parent_wrapper" style="cursor: pointer; padding: 15px; margin: 10px; border: 1px dashed green;">
      <p>
        Parent container text that elaborates on sub-options and also deliberately exceeds the one hundred and twenty character threshold so it cannot become a single custom button.
      </p>
      <!-- Child 1: pointer, short label -->
      <div id="choice_alpha" style="cursor: pointer; background: rgb(240, 240, 240); padding: 10px; margin: 5px;"
           onclick="document.getElementById('stress_res').textContent='alpha_selected'">
        Stress Choice Alpha
      </div>
      <!-- Child 2: pointer, short label -->
      <div id="choice_beta" style="cursor: pointer; background: rgb(240, 240, 240); padding: 10px; margin: 5px;"
           onclick="document.getElementById('stress_res').textContent='beta_selected'">
        Stress Choice Beta
      </div>
      <!-- Child 3: pointer wrapper containing a native button matching selector -->
      <div id="wrapper_with_native" style="cursor: pointer; background: rgb(240, 240, 240); padding: 10px; margin: 5px;">
        <button id="native_action_btn" onclick="document.getElementById('stress_res').textContent='native_clicked'">
          Native Inside Pointer Wrapper
        </button>
      </div>
    </div>
  </div>
  <div id="stress_res">idle</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # 1. Verify grandparent and parent are not in actions
            action_labels = [a.get("label") for a in actions]
            assert not any("Grandparent outer card" in lbl for lbl in action_labels if lbl)
            assert not any("Parent container text" in lbl for lbl in action_labels if lbl)

            # 2. Child 1 and Child 2 must be listed as custom buttons
            alpha_act = next((a for a in actions if a.get("label") == "Stress Choice Alpha"), None)
            assert alpha_act is not None, "Child Choice Alpha must be listed"
            assert alpha_act["role"] == "button"
            assert alpha_act.get("custom") is True

            beta_act = next((a for a in actions if a.get("label") == "Stress Choice Beta"), None)
            assert beta_act is not None, "Child Choice Beta must be listed"
            assert beta_act["role"] == "button"
            assert beta_act.get("custom") is True

            # 3. wrapper_with_native must NOT be a custom button, but the native button inside must be listed
            native_act = next((a for a in actions if a.get("label") == "Native Inside Pointer Wrapper"), None)
            assert native_act is not None, "Native button must be listed"
            assert native_act["role"] == "button"
            assert native_act.get("custom") is not True

            # 4. Verify act() on custom button
            browser.act(alpha_act, page=page)
            time.sleep(0.2)
            res1 = browser.evaluate("document.getElementById('stress_res').textContent")
            assert res1 == "alpha_selected", "Clicking custom pointer option must trigger handler"

            # 5. Verify act() on native button
            browser.act(native_act, page=page)
            time.sleep(0.2)
            res2 = browser.evaluate("document.getElementById('stress_res').textContent")
            assert res2 == "native_clicked", "Clicking native button inside pointer wrapper must trigger handler"
        finally:
            browser.close()


def test_act_fill_datetime_local_month_week_stress() -> None:
    """Stress test: act() filling datetime-local, month, and week inputs with event firing.

    Verifies:
    1. HTML5 datetime-local, month, and week inputs are recognized with role='textbox'.
    2. Values are correctly assigned via act().
    3. Input and change events are fired.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <input id="dt_inp" type="datetime-local" value="2026-01-01T10:00" aria-label="Conference Schedule"
         onchange="document.getElementById('dt_out').textContent=this.value" />
  <div id="dt_out">idle</div>

  <input id="m_inp" type="month" value="2026-03" aria-label="Subscription Period"
         onchange="document.getElementById('m_out').textContent=this.value" />
  <div id="m_out">idle</div>

  <input id="w_inp" type="week" value="2026-W12" aria-label="Sprint Target"
         onchange="document.getElementById('w_out').textContent=this.value" />
  <div id="w_out">idle</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()

            # 1. Verify datetime-local
            dt_act = next(a for a in page["actions"] if a.get("label") == "Conference Schedule")
            assert dt_act["role"] == "textbox"
            assert dt_act.get("value") == "2026-01-01T10:00"
            browser.act(dt_act, page=page, text="2027-12-31T23:59")
            assert browser.evaluate("document.getElementById('dt_inp').value") == "2027-12-31T23:59"
            assert browser.evaluate("document.getElementById('dt_out').textContent") == "2027-12-31T23:59"

            # 2. Verify month
            m_act = next(a for a in page["actions"] if a.get("label") == "Subscription Period")
            assert m_act["role"] == "textbox"
            assert m_act.get("value") == "2026-03"
            browser.act(m_act, page=page, text="2028-09")
            assert browser.evaluate("document.getElementById('m_inp').value") == "2028-09"
            assert browser.evaluate("document.getElementById('m_out').textContent") == "2028-09"

            # 3. Verify week
            w_act = next(a for a in page["actions"] if a.get("label") == "Sprint Target")
            assert w_act["role"] == "textbox"
            assert w_act.get("value") == "2026-W12"
            browser.act(w_act, page=page, text="2028-W42")
            assert browser.evaluate("document.getElementById('w_inp').value") == "2028-W42"
            assert browser.evaluate("document.getElementById('w_out').textContent") == "2028-W42"
        finally:
            browser.close()


def test_aria_slider_and_menuitemcheckbox_act_stress() -> None:
    """Stress test: role='slider' (custom element) and role='menuitemcheckbox' with act() interaction.

    Verifies:
    1. Custom element with role='slider' has kind='fill' and aria-valuenow reflected.
    2. act() on custom slider sets aria-valuenow and dispatches events.
    3. role='menuitemcheckbox' has kind='click' and reflects checked state.
    4. act() clicking role='menuitemcheckbox' triggers click handler and updates state.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<body>
  <div id="c_slider" role="slider" tabindex="0" aria-valuenow="40" aria-valuemin="0" aria-valuemax="100"
       aria-label="Display Brightness" style="display:block;width:150px;height:25px;"
       onkeydown="let v=Number(this.getAttribute('aria-valuenow')); if(event.key==='ArrowRight') v=Math.min(100,v+1); else if(event.key==='ArrowLeft') v=Math.max(0,v-1); this.setAttribute('aria-valuenow',String(v));"
       onchange="document.getElementById('slider_out').textContent=this.getAttribute('aria-valuenow')">
  </div>
  <div id="slider_out">idle</div>

  <div role="menu">
    <div id="c_chk" role="menuitemcheckbox" aria-checked="false" aria-label="Auto Dark Mode"
         onclick="const n=this.getAttribute('aria-checked')==='true'?'false':'true'; this.setAttribute('aria-checked',n); document.getElementById('chk_out').textContent=n;">
      Auto Dark Mode
    </div>
  </div>
  <div id="chk_out">idle</div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            # 1. Custom slider
            slider_act = next((a for a in actions if a.get("label") == "Display Brightness"), None)
            assert slider_act is not None, "Custom slider must be listed"
            assert slider_act["role"] == "slider"
            assert slider_act.get("value") == "40"
            assert slider_act.get("kind") == "fill"

            browser.act(slider_act, page=page, text="92")
            slider_now = browser.evaluate("document.getElementById('c_slider').getAttribute('aria-valuenow')")
            assert slider_now == "92", f"Expected aria-valuenow '92', got '{slider_now}'"
            slider_out = browser.evaluate("document.getElementById('slider_out').textContent")
            assert slider_out == "92", "change event must fire for custom slider"

            # 2. Menuitemcheckbox
            chk_act = next((a for a in actions if a.get("label") == "Auto Dark Mode"), None)
            assert chk_act is not None, "role='menuitemcheckbox' must be listed"
            assert chk_act["role"] == "menuitemcheckbox"
            assert chk_act.get("checked") == "false"
            assert chk_act.get("kind") == "click"

            browser.act(chk_act, page=page)
            time.sleep(0.2)
            chk_out = browser.evaluate("document.getElementById('chk_out').textContent")
            assert chk_out == "true", "Clicking menuitemcheckbox must toggle checked state to true"
        finally:
            browser.close()


def test_form_target_blank_neutralization_button_formtarget_stress() -> None:
    """Stress test: form submission neutralization when submit button uses formtarget='_blank'.

    Verifies:
    1. Submit button with formtarget='_blank' has formtarget stripped upon click in act().
    2. Form submission stays in current tab without spawning any new targets.
    3. Navigation succeeds in the same browsing context.
    """
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    class FormtargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if "/navigated" in self.path:
                self.wfile.write(b"<!DOCTYPE html><html><body><h1>Formtarget Neutralized Same Tab</h1></body></html>")
            else:
                self.wfile.write(b"""<!DOCTYPE html>
<html>
<body>
  <form id="f_test" action="/navigated" method="GET">
    <input type="text" name="term" value="stress_query" />
    <button id="btn_ft" type="submit" formtarget="_blank">Submit via Formtarget Button</button>
  </form>
</body>
</html>""")

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), FormtargetHandler)
    server_port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        with managed_isolated_browser(port=0) as cdp_url:
            browser = Browser(f"http://127.0.0.1:{server_port}/")
            try:
                page = browser.observe()
                ft_act = next(a for a in page["actions"] if a.get("label") == "Submit via Formtarget Button")

                init_targets = set(browser._owned_targets)
                browser.act(ft_act, page=page)
                time.sleep(0.5)

                current_targets = set(browser._owned_targets)
                assert len(current_targets) == len(init_targets), "No new browser targets should be spawned"

                after_page = browser.observe()
                assert "Formtarget Neutralized Same Tab" in after_page.get("text", "")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_slider_aria_label_and_labelledby() -> None:
    """Verify that role="slider" elements with aria-label or aria-labelledby resolve correct descriptive labels."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>ARIA Slider Labels</title></head>
<body>
  <h1>Retirement Planning Sliders</h1>
  <span id="lbl_retire">Retirement Age</span>
  <div id="slider_retire" role="slider" aria-labelledby="lbl_retire" aria-valuenow="65" aria-valuemin="18" aria-valuemax="100" tabindex="0" style="width:200px; height:20px;"></div>

  <div id="slider_current" role="slider" aria-label="Current Age" aria-valuenow="30" aria-valuemin="18" aria-valuemax="100" tabindex="0" style="width:200px; height:20px;"></div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            retire_slider = next((a for a in actions if a.get("role") == "slider" and "Retirement Age" in a.get("label", "")), None)
            assert retire_slider is not None, "Slider with aria-labelledby must have label 'Retirement Age'"
            assert retire_slider.get("value") == "65"
            assert retire_slider.get("label") != "slider", "Must not fall back to generic 'slider'"

            current_slider = next((a for a in actions if a.get("role") == "slider" and "Current Age" in a.get("label", "")), None)
            assert current_slider is not None, "Slider with aria-label must have label 'Current Age'"
            assert current_slider.get("value") == "30"
            assert current_slider.get("label") != "slider", "Must not fall back to generic 'slider'"
        finally:
            browser.close()


def test_input_range_labels_explicit_and_wrapped() -> None:
    """Verify that <input type="range"> extracts labels from <label for="..."> and wrapping <label>."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>Range Input Labels</title></head>
<body>
  <h1>Financial Inputs</h1>
  <label for="balance_input">Starting Balance</label>
  <input id="balance_input" type="range" min="0" max="100000" value="30000" style="width:200px; height:20px;" />

  <label>
    Annual Return Rate
    <input id="return_input" type="range" min="0" max="20" value="3" style="width:200px; height:20px;" />
  </label>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            balance_act = next((a for a in actions if a.get("role") == "slider" and "Starting Balance" in a.get("label", "")), None)
            assert balance_act is not None, "Range input with <label for> must have label 'Starting Balance'"
            assert balance_act.get("value") == "30000"

            return_act = next((a for a in actions if a.get("role") == "slider" and "Annual Return Rate" in a.get("label", "")), None)
            assert return_act is not None, "Range input inside <label> must have label 'Annual Return Rate'"
            assert return_act.get("value") == "3"
        finally:
            browser.close()


def test_slider_container_and_sibling_labels() -> None:
    """Verify that custom UI sliders (div[role=slider] without aria-label) extract labels from container/sibling elements."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>Container Sliders</title></head>
<body>
  <h1>Tax Rate Configuration</h1>
  <!-- Container Pattern 1: Heading and sibling span inside form group -->
  <div class="form-group" style="padding: 10px;">
    <span class="field-title" style="font-weight: bold;">Current Tax Rate</span>
    <div class="slider-track" style="margin-top: 5px;">
      <div id="slider_curr_tax" role="slider" tabindex="0" aria-valuenow="13" aria-valuemin="0" aria-valuemax="50" style="width:200px; height:20px;"></div>
    </div>
    <span class="field-display">13%</span>
  </div>

  <!-- Container Pattern 2: Fieldset with legend and range input -->
  <fieldset style="margin-top: 20px;">
    <legend>Retirement Tax Rate</legend>
    <div class="range-wrapper">
      <input id="slider_ret_tax" type="range" min="0" max="50" value="24" style="width:200px; height:20px;" />
    </div>
  </fieldset>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            curr_tax = next((a for a in actions if a.get("role") == "slider" and "Current Tax Rate" in a.get("label", "")), None)
            assert curr_tax is not None, "Slider in form-group must extract 'Current Tax Rate' from container sibling"
            assert curr_tax.get("value") == "13"
            assert curr_tax.get("label") != "slider", "Must not fall back to generic 'slider'"

            ret_tax = next((a for a in actions if a.get("role") == "slider" and "Retirement Tax Rate" in a.get("label", "")), None)
            assert ret_tax is not None, "Slider in fieldset must extract 'Retirement Tax Rate' from legend"
            assert ret_tax.get("value") == "24"
            assert ret_tax.get("label") != "slider", "Must not fall back to generic 'slider'"
        finally:
            browser.close()


def test_slider_range_boilerplate_fallback_label() -> None:
    """Verify that sliders with range boilerplate aria-labels fall back to container labels."""
    if not is_chromium_available():
        pytest.skip("Chromium required for isolated browser test")

    html = """data:text/html,<!DOCTYPE html>
<html>
<head><title>Slider Range Boilerplate Test</title></head>
<body>
  <h1>Retirement Calculator</h1>

  <!-- Pattern 1: Boilerplate Slider range minimum ... with container field-title -->
  <div class="form-group" style="margin-bottom: 20px;">
    <span class="field-title" style="font-weight: bold;">Current Age</span>
    <div class="slider-track" style="margin-top: 5px;">
      <div id="sl_age" role="slider" tabindex="0"
           aria-label="Slider range minimum 0, maximum 100000, increments of 1"
           aria-valuenow="35" style="width:200px; height:20px;"></div>
    </div>
  </div>

  <!-- Pattern 2: Dual thumb leading/trailing range boilerplate -->
  <div class="form-group" style="margin-bottom: 20px;">
    <span class="field-title" style="font-weight: bold;">Target Retirement Age</span>
    <div class="slider-track" style="margin-top: 5px;">
      <div id="sl_retire" role="slider" tabindex="0"
           aria-label="leading, 25 to 75 25 to 75"
           aria-valuenow="65" style="width:200px; height:20px;"></div>
    </div>
  </div>
</body>
</html>"""

    with managed_isolated_browser(port=0) as cdp_url:
        browser = Browser(html)
        try:
            page = browser.observe()
            actions = page["actions"]

            age_act = next((a for a in actions if a.get("role") == "slider" and "Current Age" in a.get("label", "")), None)
            assert age_act is not None, f"Slider for Current Age must be present in actions: {[a.get('label') for a in actions]}"
            assert age_act.get("label") == "Current Age", (
                f"Expected label 'Current Age', got {age_act.get('label')!r}"
            )
            assert age_act.get("value") == "35"

            retire_act = next((a for a in actions if a.get("role") == "slider" and "Target Retirement Age" in a.get("label", "")), None)
            assert retire_act is not None, f"Slider for Target Retirement Age must be present in actions: {[a.get('label') for a in actions]}"
            assert age_act.get("label") != "Slider range minimum 0, maximum 100000, increments of 1"
            assert retire_act.get("label") == "Target Retirement Age", (
                f"Expected label 'Target Retirement Age', got {retire_act.get('label')!r}"
            )
            assert retire_act.get("value") == "65"
        finally:
            browser.close()



