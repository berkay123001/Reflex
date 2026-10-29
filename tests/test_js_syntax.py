"""Tests for JavaScript syntax verification in snapshot.js and cdp.py inline JS."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from reflex.browser.cdp import (
    _STEALTH_INIT_SCRIPT,
    DEFAULT_CDP_PORT,
    MARKER,
    Browser,
    ensure_chromium_cdp,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = ROOT_DIR / "reflex" / "browser" / "snapshot.js"
CDP_PATH = ROOT_DIR / "reflex" / "browser" / "cdp.py"


@pytest.fixture
def node_executable():
    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js is not installed; skipping JS syntax validation tests")
    return node_bin


def _check_js_syntax(node_bin: str, code: str) -> None:
    res = subprocess.run([node_bin, "--check"], check=False, input=code, text=True, capture_output=True)
    assert res.returncode == 0, f"JavaScript SyntaxError:\n{res.stderr}\nCode snippet:\n{code[:300]}..."


def test_snapshot_js_syntax(node_executable):
    """snapshot.js must be valid JavaScript syntax."""
    content = SNAPSHOT_PATH.read_text(encoding="utf-8")
    _check_js_syntax(node_executable, content)


def test_cdp_marker_syntax(node_executable):
    """MARKER expression in cdp.py must be valid JavaScript."""
    _check_js_syntax(node_executable, MARKER)


def test_cdp_inline_scripts_syntax(node_executable):
    """All inline JS templates in cdp.py must have valid JS syntax."""
    cdp_code = CDP_PATH.read_text(encoding="utf-8")

    # 1. _STEALTH_INIT_SCRIPT
    _check_js_syntax(node_executable, _STEALTH_INIT_SCRIPT)

    # 2. Settle state script
    m_settle = re.search(r'state = self\.evaluate\(\s*"""(.*?)"""\s*\)', cdp_code, re.DOTALL)
    assert m_settle, "Could not find settle state script"
    _check_js_syntax(node_executable, m_settle.group(1))

    # 3. act target query script
    m_target = re.search(r'target = self\.evaluate\(\s*"""(.*?)"""\s*\+\s*json\.dumps\(action\)\s*\+\s*"\)"', cdp_code, re.DOTALL)
    assert m_target, "Could not find act target script"
    dummy_action = {"kind": "click", "node": 1}
    _check_js_syntax(node_executable, m_target.group(1) + json.dumps(dummy_action) + ")")

    # 4. act clear field script
    m_clear = re.search(r'self\.evaluate\(\s*"""\(nodeId => \{(.*?)\}\)\("""\s*\+\s*json\.dumps\(action\.get\("node"\)\)\s*\+\s*"\)"', cdp_code, re.DOTALL)
    assert m_clear, "Could not find clear field script"
    _check_js_syntax(node_executable, f"(nodeId => {{{m_clear.group(1)}}})(1)")

    # 5. act is_search script
    m_search = re.search(r'is_search = bool\(\s*self\.evaluate\(\s*"""\(nodeId => \{(.*?)\}\)\("""\s*\+\s*json\.dumps\(action\.get\("node"\)\)\s*\+\s*"\)"', cdp_code, re.DOTALL)
    assert m_search, "Could not find is_search script"
    _check_js_syntax(node_executable, f"(nodeId => {{{m_search.group(1)}}})(1)")

    # 6. fresh() script
    m_fresh = re.search(r'current = self\.evaluate\(\s*(".*?")\s*f(".*?")\s*\)', cdp_code, re.DOTALL)
    assert m_fresh, "Could not find fresh script"
    part1 = json.loads(m_fresh.group(1))
    part2 = json.loads(m_fresh.group(2).replace("}}", "}").replace("{node}", "42"))
    _check_js_syntax(node_executable, part1 + part2)


@pytest.mark.parametrize(
    "action",
    [
        {"kind": "click", "node": 42},
        {"kind": "fill", "node": 10, "text": "search query"},
        {"kind": "select", "node": 5, "value": "opt1"},
    ],
)
def test_observe_wait_syntax_with_fake_call(node_executable, action):
    """observe() generated quiescence script must be captured via fake call and pass node syntax check."""
    b = Browser.__new__(Browser)
    b.after_input = action
    captured_calls = []

    def fake_call(method, **kwargs):
        captured_calls.append((method, kwargs))
        return {}

    def fake_eval(expr):
        return {
            "url": "http://127.0.0.1/test",
            "title": "Test",
            "text": "test",
            "actions": [],
            "scroll": {},
            "marker": "test-marker",
            "guards": {},
        }

    b.call = fake_call
    b.evaluate = fake_eval
    b.last_action_rect = None
    b.last_jpeg_bytes = None
    b._try_pierce_cloudflare = lambda info: False

    b.observe(screenshot=False)

    eval_calls = [kwargs for method, kwargs in captured_calls if method == "Runtime.evaluate"]
    assert len(eval_calls) >= 1, "Runtime.evaluate was not called for after_input wait"
    expression = eval_calls[0]["expression"]

    # Verify syntax with node --check
    _check_js_syntax(node_executable, expression)


def test_broken_syntax_demonstration(node_executable):
    """Demonstrate that the previous syntax without opening paren fails node --check."""
    broken_expression = '(action => new Promise(resolve => { resolve(action); })){"kind": "click", "node": 42})'
    res = subprocess.run([node_executable, "--check"], check=False, input=broken_expression, text=True, capture_output=True)
    assert res.returncode != 0
    assert "SyntaxError" in res.stderr

    fixed_expression = '(action => new Promise(resolve => { resolve(action); }))({"kind": "click", "node": 42})'
    res_fixed = subprocess.run([node_executable, "--check"], check=False, input=fixed_expression, text=True, capture_output=True)
    assert res_fixed.returncode == 0


def test_cdp_port_respects_bu_cdp_url():
    """ensure_chromium_cdp must probe port from BU_CDP_URL and never probe 9333 when BU_CDP_URL is set."""
    probed_ports = []

    def fake_port_open(host: str, port: int) -> bool:
        probed_ports.append(port)
        return True

    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9444"}, clear=False),
        patch("reflex.browser.cdp._port_open", side_effect=fake_port_open),
    ):
        ensure_chromium_cdp()
        assert probed_ports == [9444]
        assert 9333 not in probed_ports
        assert os.environ["BU_CDP_URL"] == "http://127.0.0.1:9444"


def test_browser_init_port_respects_bu_cdp_url():
    """Browser.__init__ must pass parsed port from BU_CDP_URL to ensure_chromium_cdp."""
    passed_ports = []

    def fake_ensure(port=None):
        passed_ports.append(port)

    fake_cdp = lambda method, *args, **kwargs: (
        {"targetId": "target_123"}
        if method == "Target.createTarget"
        else {"sessionId": "session_123"}
        if method == "Target.attachToTarget"
        else {"result": {"value": "Mozilla/5.0"}}
        if method == "Runtime.evaluate"
        else {"identifier": "script_1"}
        if method == "Page.addScriptToEvaluateOnNewDocument"
        else {}
    )

    with (
        patch.dict(os.environ, {"BU_CDP_URL": "http://127.0.0.1:9444"}, clear=False),
        patch("reflex.browser.cdp.ensure_chromium_cdp", side_effect=fake_ensure),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp),
        patch.object(Browser, "navigate", return_value=None),
    ):
        Browser("http://example.com")
        assert passed_ports == [9444]
        assert DEFAULT_CDP_PORT not in passed_ports
