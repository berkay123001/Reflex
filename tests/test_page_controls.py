"""Unit tests for Milestone M1 (R1: Page-wide form controls for S2 in package mode)."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from reflex.agent import ReflexConfig, ReflexSession
from reflex.config import MAX_SYSTEM1_ELEMENTS
from reflex.package import create_action_package
from reflex.safety import (
    MAX_CONTROL_LABEL_LENGTH,
    MAX_PAGE_CONTROLS,
    wrap_untrusted_dom,
)
from reflex.supervisor import SupervisorDecision, parse_supervisor_decision
from reflex.system1 import action_space, build_system1_request

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_JS = ROOT / "reflex" / "browser" / "snapshot.js"

_NODE_RUNNER = textwrap.dedent(
    r"""
    const http = require('http');
    const fs = require('fs');

    function getJson(url) {
      return new Promise((resolve, reject) => {
        const req = http.get(url, res => {
          let data = '';
          res.on('data', chunk => { data += chunk; });
          res.on('end', () => {
            try { resolve(JSON.parse(data)); }
            catch (err) { reject(new Error(String(err) + ' ' + data.slice(0, 180))); }
          });
        });
        req.on('error', reject);
      });
    }

    async function waitFor(fn, ms = 12000) {
      const started = Date.now();
      let last = new Error('timeout');
      while (Date.now() - started < ms) {
        try { return await fn(); }
        catch (err) { last = err; }
        await new Promise(resolve => setTimeout(resolve, 80));
      }
      throw last;
    }

    (async () => {
      const port = process.argv[1];
      const pageUrl = process.argv[2];
      const snap = fs.readFileSync(process.argv[3], 'utf8');
      const page = await waitFor(async () => {
        const pages = await getJson('http://127.0.0.1:' + port + '/json/list');
        const found = pages.find(item => item.type === 'page' && item.webSocketDebuggerUrl);
        if (!found) throw new Error('no page target');
        return found;
      });
      const ws = new WebSocket(page.webSocketDebuggerUrl);
      await new Promise((resolve, reject) => {
        ws.addEventListener('open', resolve);
        ws.addEventListener('error', reject);
      });
      let nextId = 1;
      function call(method, params = {}) {
        return new Promise((resolve, reject) => {
          const id = nextId++;
          const onMsg = (event) => {
            const data = JSON.parse(event.data);
            if (data.id === id) {
              ws.removeEventListener('message', onMsg);
              if (data.error) reject(new Error(data.error.message));
              else resolve(data.result);
            }
          };
          ws.addEventListener('message', onMsg);
          ws.send(JSON.stringify({ id, method, params }));
        });
      }
      await call('Page.enable');
      await call('Page.navigate', { url: pageUrl });
      await waitFor(async () => {
        const res = await call('Runtime.evaluate', { expression: 'document.readyState', returnByValue: true });
        if (res.result.value !== 'complete') throw new Error('loading');
        return true;
      });
      await new Promise(r => setTimeout(r, 100));
      const evaluated = await call('Runtime.evaluate', {
        expression: snap,
        returnByValue: true,
        awaitPromise: true,
      });
      if (evaluated.exceptionDetails) {
        console.error(JSON.stringify(evaluated.exceptionDetails));
        process.exit(2);
      }
      process.stdout.write(JSON.stringify(evaluated.result.value));
      ws.close();
    })().catch(err => {
      console.error(err && err.stack ? err.stack : String(err));
      process.exit(1);
    });
    """
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port == 9333:
        return _free_port()
    return port


@pytest.fixture(scope="module")
def chromium_port(tmp_path_factory: pytest.TempPathFactory) -> Any:
    binary = shutil.which("chromium") or shutil.which("google-chrome")
    node = shutil.which("node")
    if not binary or not node:
        pytest.skip("chromium and node are required to execute snapshot.js")
    profile = tmp_path_factory.mktemp("chrome-profile")
    port = _free_port()
    proc = subprocess.Popen(
        [
            binary,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--remote-allow-origins=*",
            "--window-size=1200,800",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _run_snapshot(port: int, html_file: Path) -> dict[str, Any]:
    node = shutil.which("node")
    assert node
    result = subprocess.run(
        [node, "-e", _NODE_RUNNER, str(port), html_file.resolve().as_uri(), str(SNAPSHOT_JS)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"Node error: {result.stderr}"
    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    return data


# --- 1. Browser Snapshot Tests ---


def test_offscreen_controls_collected_and_guards_registered(chromium_port: int, tmp_path: Path) -> None:
    """Verify offscreen radio/checkbox/select/text items appear in page_controls with guards and states."""
    html = tmp_path / "survey.html"
    html.write_text(
        """<!DOCTYPE html>
<html>
<head><style>body { margin: 0; font-family: sans-serif; }</style></head>
<body>
  <!-- Viewport section (height <= 800) -->
  <div style="height: 300px; padding: 20px;">
    <h1>Survey Header</h1>
    <button id="btn1">Start Survey</button>
  </div>

  <!-- Offscreen section below viewport fold (top > 1200px) -->
  <div style="margin-top: 1500px; padding: 20px;">
    <fieldset>
      <legend>Question 1: Experience Level</legend>
      <label><input type="radio" name="exp" value="beg" checked> Beginner</label>
      <label><input type="radio" name="exp" value="adv"> Advanced</label>
    </fieldset>

    <div role="radiogroup" aria-labelledby="q2head">
      <h3 id="q2head">Question 2: Framework</h3>
      <label><input type="radio" name="fw" value="react"> React</label>
      <label><input type="radio" name="fw" value="vue"> Vue</label>
    </div>

    <div style="margin-top: 20px;">
      <label>Subscribe to newsletter <input type="checkbox" name="sub" checked></label>
    </div>

    <div style="margin-top: 20px;">
      <label for="country_sel">Country</label>
      <select id="country_sel" name="country">
        <option value="us">United States</option>
        <option value="tr" selected>Turkey</option>
      </select>
    </div>

    <div style="margin-top: 20px;">
      <label for="feedback_txt">Feedback</label>
      <textarea id="feedback_txt" name="feedback">Great tool</textarea>
    </div>

    <div style="margin-top: 20px;">
      <div role="checkbox" aria-checked="true" aria-label="Custom Checkbox">Accept Terms</div>
      <div role="radio" aria-checked="false" aria-label="Custom Radio">Option X</div>
      <div role="switch" aria-checked="true" aria-label="Dark Mode">Toggle Dark</div>
    </div>
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )

    page = _run_snapshot(chromium_port, html)
    actions = page.get("actions", [])
    page_controls = page.get("page_controls", [])
    guards = page.get("guards", {})

    # Onscreen button is in actions
    onscreen_nodes = {a["node"] for a in actions if a.get("node") is not None}
    assert any(a.get("label") == "Start Survey" for a in actions)

    # Offscreen controls are in page_controls
    assert len(page_controls) >= 8

    # Deduplication check: none of the page_controls are present in actions
    for ctrl in page_controls:
        assert ctrl["node"] not in onscreen_nodes
        assert ctrl["node"] in guards or str(ctrl["node"]) in guards
        guard_val = guards.get(ctrl["node"]) or guards.get(str(ctrl["node"]))
        assert guard_val is not None

    # Check question context label generation
    exp_labels = [c["label"] for c in page_controls if c["role"] == "radio" and "Experience Level" in c["label"]]
    assert len(exp_labels) == 2
    assert any("Beginner" in l for l in exp_labels)
    assert any("Advanced" in l for l in exp_labels)
    assert all("Question 1: Experience Level → " in l for l in exp_labels)

    fw_labels = [c["label"] for c in page_controls if c["role"] == "radio" and "Framework" in c["label"]]
    assert len(fw_labels) == 2
    assert all("Question 2: Framework → " in l for l in fw_labels)

    # Check state extraction
    beg_ctrl = next(c for c in page_controls if "Beginner" in c["label"])
    assert beg_ctrl.get("checked") == "true"

    adv_ctrl = next(c for c in page_controls if "Advanced" in c["label"])
    assert adv_ctrl.get("checked") == "false"

    sub_ctrl = next(c for c in page_controls if c["role"] == "checkbox" and "newsletter" in c["label"])
    assert sub_ctrl.get("checked") == "true"
    assert sub_ctrl["kind"] == "click"

    select_ctrl = next(c for c in page_controls if c["role"] == "combobox")
    assert select_ctrl["kind"] == "select"
    assert "Turkey" in select_ctrl.get("value", "")

    textarea_ctrl = next(c for c in page_controls if c["role"] == "textbox" and "Feedback" in c["label"])
    assert textarea_ctrl["kind"] == "fill"
    assert "Great tool" in textarea_ctrl.get("value", "")

    custom_cb = next(c for c in page_controls if c["role"] == "checkbox" and "Custom Checkbox" in c["label"])
    assert custom_cb.get("checked") == "true"

    custom_switch = next(c for c in page_controls if c["role"] == "switch")
    assert custom_switch.get("checked") == "true"


def test_invisible_aria_hidden_inert_omitted(chromium_port: int, tmp_path: Path) -> None:
    """Verify invisible, aria-hidden, inert, and unsafe inputs are completely omitted from page_controls."""
    html = tmp_path / "hidden.html"
    html.write_text(
        """<!DOCTYPE html>
<html>
<body>
  <div style="margin-top: 1500px;">
    <!-- Visible control -->
    <label><input type="radio" name="vis" value="ok"> Visible Option</label>

    <!-- Invisible via display: none -->
    <label style="display: none;"><input type="radio" name="inv1" value="no"> Hidden Display None</label>

    <!-- Invisible via visibility: hidden -->
    <label style="visibility: hidden;"><input type="radio" name="inv2" value="no"> Hidden Visibility</label>

    <!-- aria-hidden="true" -->
    <div aria-hidden="true">
      <label><input type="radio" name="ah" value="no"> Aria Hidden Option</label>
    </div>

    <!-- inert -->
    <div inert>
      <label><input type="checkbox" name="inert_cb"> Inert Checkbox</label>
    </div>

    <!-- Disabled input -->
    <input type="text" disabled name="dis" value="Disabled Input">

    <!-- Unsafe types: password, hidden, file -->
    <input type="password" name="pwd" value="secret">
    <input type="hidden" name="csrf" value="12345">
    <input type="file" name="doc">
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )

    page = _run_snapshot(chromium_port, html)
    page_controls = page.get("page_controls", [])

    assert len(page_controls) == 1
    assert "Visible Option" in page_controls[0]["label"]
    for c in page_controls:
        lbl = c["label"]
        assert "Hidden" not in lbl
        assert "Aria Hidden" not in lbl
        assert "Inert" not in lbl
        assert "Disabled" not in lbl
        assert "secret" not in lbl


def test_caps_enforced_150_controls_and_140_char_label(chromium_port: int, tmp_path: Path) -> None:
    """Verify MAX_PAGE_CONTROLS = 150 and MAX_CONTROL_LABEL_LEN = 140 are strictly enforced."""
    long_question = "Q" * 100
    long_option = "O" * 100  # Total combined length > 200 chars

    options_html = []
    # Generate 180 off-screen radio buttons
    for i in range(180):
        if i == 0:
            options_html.append(f'<label><input type="radio" name="cap_q" value="{i}"> {long_option}</label>')
        else:
            options_html.append(f'<label><input type="radio" name="cap_q" value="{i}"> Opt {i}</label>')

    html = tmp_path / "caps.html"
    html.write_text(
        f"""<!DOCTYPE html>
<html>
<body>
  <div style="margin-top: 1500px;">
    <fieldset>
      <legend>{long_question}</legend>
      {''.join(options_html)}
    </fieldset>
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )

    page = _run_snapshot(chromium_port, html)
    page_controls = page.get("page_controls", [])

    assert len(page_controls) == MAX_PAGE_CONTROLS
    assert len(page_controls) == 150

    # Verify label length cap
    for c in page_controls:
        assert len(c["label"]) <= MAX_CONTROL_LABEL_LENGTH
        assert len(c["label"]) <= 140

    first_label = page_controls[0]["label"]
    assert len(first_label) == 140


# --- 2. S2 Prompt & System 1 Insulation Tests ---


def test_wrap_untrusted_dom_s2_prompt_package_mode() -> None:
    """Verify wrap_untrusted_dom renders f-indices only in package mode when page_controls exists."""
    page = {
        "url": "https://example.test/form",
        "title": "Registration",
        "text": "Please register below",
        "scroll": {"y": 0, "height": 2000},
        "page_controls": [
            {"node": 10, "role": "radio", "kind": "click", "label": "Plan → Free", "checked": "true"},
            {"node": 11, "role": "radio", "kind": "click", "label": "Plan → Pro", "checked": "false"},
            {"node": 12, "role": "combobox", "kind": "select", "label": "Country", "value": "TR"},
            {"node": 13, "role": "textbox", "kind": "fill", "label": "Full Name", "value": ""},
        ],
    }
    elements = [
        {"index": "1", "role": "button", "label": "Next", "operations": ["CLICK"]},
    ]

    # 1. Non-package mode: no Page-wide form controls section
    out_non_pkg = wrap_untrusted_dom(page, elements, package_mode=False)
    assert "Page-wide form controls (also off-screen):" not in out_non_pkg
    assert "[f1]" not in out_non_pkg

    # 2. Package mode: contains Page-wide form controls section with [f1]..[f4]
    out_pkg = wrap_untrusted_dom(page, elements, package_mode=True)
    assert "Page-wide form controls (also off-screen):" in out_pkg
    assert "[f1] (radio) Plan → Free | checked=true | ops: CLICK" in out_pkg
    assert "[f2] (radio) Plan → Pro | checked=false | ops: CLICK" in out_pkg
    assert "[f3] (combobox) Country | value=TR | ops: SELECT" in out_pkg
    assert "[f4] (textbox) Full Name | ops: TYPE_TEXT" in out_pkg

    # 3. Empty page_controls in package mode: no section appended
    empty_page = dict(page, page_controls=[])
    out_empty_pkg = wrap_untrusted_dom(empty_page, elements, package_mode=True)
    assert "Page-wide form controls (also off-screen):" not in out_empty_pkg


def test_wrap_untrusted_dom_byte_for_byte_equality_in_non_package_mode() -> None:
    """Verify wrap_untrusted_dom is 100% byte-for-byte identical to baseline in non-package mode."""
    page_with_controls = {
        "url": "https://example.test/dense",
        "title": "Dense Page",
        "text": "Some visible viewport text here.",
        "full_text": "Full text spanning multiple pages and sections.",
        "headings": ["Heading 1", "Heading 2"],
        "scroll": {"y": 100, "height": 3000},
        "h": 780,
        "actions": [{"node": i, "kind": "click"} for i in range(1, 20)],
        "page_controls": [
            {"node": 200 + i, "role": "radio", "kind": "click", "label": f"Radio {i}"} for i in range(20)
        ],
    }
    page_without_controls = {k: v for k, v in page_with_controls.items() if k != "page_controls"}

    elements = [{"index": str(i), "role": "button", "label": f"Btn {i}", "operations": ["CLICK"]} for i in range(1, 15)]

    # Default package_mode=False
    out_with = wrap_untrusted_dom(page_with_controls, elements)
    out_without = wrap_untrusted_dom(page_without_controls, elements)

    assert out_with == out_without
    assert len(out_with) == len(out_without)
    assert "Page-wide form controls" not in out_with


def test_system1_strictly_insulated_byte_for_byte() -> None:
    """Verify page_controls is NEVER passed to System 1, and S1 request is strictly identical."""
    page_with_controls = {
        "url": "https://example.test/s1_insulation",
        "title": "S1 Test",
        "text": "System 1 test text",
        "actions": [
            {"id": f"e{i}", "node": i, "kind": "click", "role": "button", "label": f"Item {i}"}
            for i in range(1, 70)
        ],
        "page_controls": [
            {"node": 500 + i, "role": "radio", "kind": "click", "label": f"Secret Offscreen {i}"}
            for i in range(50)
        ],
    }
    page_without_controls = {k: v for k, v in page_with_controls.items() if k != "page_controls"}

    req_with = build_system1_request(page_with_controls, "complete goal", history=[])
    req_without = build_system1_request(page_without_controls, "complete goal", history=[])

    assert req_with == req_without
    body_with = req_with[0]
    body_str = json.dumps(body_with)
    assert "Secret Offscreen" not in body_str
    assert "[f1]" not in body_str
    assert "page_controls" not in body_str

    # System 1 action space limit remains strictly 60
    elements, _targets, _controls = action_space(page_with_controls.get("actions", []))
    assert len(elements) <= MAX_SYSTEM1_ELEMENTS
    assert len(elements) == 60


# --- 3. Agent & Package Target Resolution Tests ---


def test_agent_handle_escalation_targets_mapping_f_indices() -> None:
    """Verify _handle_escalation populates targets with f indices when s2_action_packages is enabled."""
    session = ReflexSession.__new__(ReflexSession)
    session.config = ReflexConfig(s2_action_packages=True, enable_planner=False)
    session.history = []
    session.ctx = MagicMock()
    session.site_memory = MagicMock()
    session._noop_escalation_counts = {}
    session._is_rejected_target = lambda cand, url: False
    session._is_ineffective_click = lambda cand, url: False

    session.page = {
        "url": "https://example.test/quiz",
        "title": "Quiz",
        "text": "Quiz form",
        "actions": [
            {"id": "e1", "node": 1, "kind": "click", "role": "button", "label": "Start"},
        ],
        "page_controls": [
            {"node": 88, "role": "radio", "kind": "click", "label": "Question 1 → Option A", "checked": "false"},
            {"node": 89, "role": "combobox", "kind": "select", "label": "Question 2 → City", "value": "Paris"},
            {"node": 90, "role": "textbox", "kind": "fill", "label": "Question 3 → Comments", "value": ""},
        ],
    }

    # Verify action_space elements and targets
    _elements, targets, controls = action_space(session.page["actions"], max_elements=150)

    # In package mode, targets is populated with f1, f2, f3
    for i, ctrl in enumerate(session.page["page_controls"]):
        f_idx = f"f{i + 1}"
        node = ctrl["node"]
        kind = ctrl["kind"]
        role = ctrl["role"]
        label = ctrl["label"]
        op = "TYPE_TEXT" if kind == "fill" else ("SELECT" if kind == "select" else "CLICK")
        act = {
            "id": f_idx,
            "node": node,
            "kind": kind,
            "role": role,
            "label": label,
            "checked": ctrl.get("checked"),
            "selected": ctrl.get("selected"),
            "value": ctrl.get("value", ""),
        }
        targets.setdefault(op, {})[f_idx] = act
        if op != "CLICK" and role in {"combobox", "textbox"}:
            targets.setdefault("CLICK", {})[f_idx] = {**act, "kind": "click"}

    assert "f1" in targets["CLICK"]
    assert targets["CLICK"]["f1"]["node"] == 88

    assert "f2" in targets["SELECT"]
    assert targets["SELECT"]["f2"]["node"] == 89
    assert "f2" in targets["CLICK"]  # combobox also registers click

    assert "f3" in targets["TYPE_TEXT"]
    assert targets["TYPE_TEXT"]["f3"]["node"] == 90
    assert "f3" in targets["CLICK"]  # textbox also registers click

    # Test single action resolution in _resolve_supervisor_action
    sup_decision = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="f1",
        reasoning="Click radio button",
    )
    resolved = session._resolve_supervisor_action(sup_decision, targets, controls)
    assert resolved is not None
    assert resolved["node"] == 88
    assert resolved["kind"] == "click"

    # Test single action with brackets [f1]
    sup_bracket = SupervisorDecision(
        action_type="OVERRIDE_ACTION",
        operation="CLICK",
        target="[f1]",
        reasoning="Click radio button with brackets",
    )
    resolved_bracket = session._resolve_supervisor_action(sup_bracket, targets, controls)
    assert resolved_bracket is not None
    assert resolved_bracket["node"] == 88


def test_create_action_package_with_f_indices() -> None:
    """Verify create_action_package validates and constructs PackageStep with f indices and captured states."""
    page = {"url": "https://example.test/form"}
    targets = {
        "CLICK": {
            "f1": {"node": 101, "role": "radio", "label": "Q1 -> Yes", "kind": "click", "checked": "false"},
            "f2": {"node": 102, "role": "checkbox", "label": "Q2 -> Accept", "kind": "click", "checked": "true"},
        },
        "TYPE_TEXT": {
            "f3": {"node": 103, "role": "textbox", "label": "Q3 -> Name", "kind": "fill", "value": ""},
        },
    }

    raw_actions = [
        {"operation": "CLICK", "target": "f1", "instruction": "Select Yes for Question 1"},
        {"operation": "CLICK", "target": "[f2]", "instruction": "Toggle Accept for Question 2"},
        {"operation": "TYPE_TEXT", "target": "f3", "text": "John Doe", "instruction": "Type user name"},
    ]

    package = create_action_package(raw_actions, page, targets)
    assert package is not None
    assert package.total_steps == 3

    # Step 1
    s1 = package.steps[0]
    assert s1.operation == "CLICK"
    assert s1.target == "f1"
    assert s1.expected.node == 101
    assert s1.expected.checked == "false"
    assert s1.instruction == "Select Yes for Question 1"

    # Step 2 (bracket stripped)
    s2 = package.steps[1]
    assert s2.operation == "CLICK"
    assert s2.target == "f2"
    assert s2.expected.node == 102
    assert s2.expected.checked == "true"

    # Step 3 (type text)
    s3 = package.steps[2]
    assert s3.operation == "TYPE_TEXT"
    assert s3.target == "f3"
    assert s3.text == "John Doe"
    assert s3.expected.node == 103


def test_supervisor_valid_targets_includes_f_indices() -> None:
    """Verify parse_supervisor_decision accepts f indices in package mode when valid_targets is provided."""
    raw_decision = {
        "action_type": "OVERRIDE_ACTION",
        "actions": [
            {"operation": "CLICK", "target": "f1", "instruction": "Click first choice"},
            {"operation": "CLICK", "target": "f2", "instruction": "Click second choice"},
        ],
    }

    # Without f1/f2 in valid_targets, actions would be dropped
    valid_targets = {"1", "2", "f1", "f2"}
    decision = parse_supervisor_decision(raw_decision, valid_targets=valid_targets)
    assert len(decision.actions) == 2
    assert decision.actions[0].target == "f1"
    assert decision.actions[1].target == "f2"
