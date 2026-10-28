"""Incomplete-form and low-confidence submit gate.

A submits_form CLICK is not executed when the same form still has an unchecked
radio group or an empty required/aria-required field. Otherwise a joint
confidence below 0.6 escalates as low_conf_submit.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reflex.agent import ReflexSession
from reflex.config import CalibrationThresholds, ReflexConfig
from reflex.gate import evaluate_decision
from reflex.supervisor import SupervisorDecision, UserMessageInterpretation

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_JS = ROOT / "reflex" / "browser" / "snapshot.js"
WEB = ROOT / "reflex" / "web"

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
        ws.addEventListener('error', () => reject(new Error('websocket failed')));
      });
      let nextId = 0;
      const pending = new Map();
      ws.addEventListener('message', event => {
        const data = JSON.parse(event.data);
        if (!data.id || !pending.has(data.id)) return;
        const waiter = pending.get(data.id);
        pending.delete(data.id);
        if (data.error) waiter.reject(new Error(JSON.stringify(data.error)));
        else waiter.resolve(data.result);
      });
      function send(method, params = {}) {
        const id = ++nextId;
        return new Promise((resolve, reject) => {
          pending.set(id, {resolve, reject});
          ws.send(JSON.stringify({id, method, params}));
        });
      }
      await send('Runtime.enable');
      try {
        await send('Emulation.setDeviceMetricsOverride', {
          width: 1400, height: 8000, deviceScaleFactor: 1, mobile: false,
        });
      } catch (_) {}
      await send('Page.enable');
      await send('Page.navigate', {url: pageUrl});
      await waitFor(async () => {
        const ready = await send('Runtime.evaluate', {
          expression: 'document.readyState + "|" + (document.body ? "1" : "0")',
          returnByValue: true,
        });
        if (!ready.result || ready.result.value !== 'complete|1') {
          throw new Error('page not ready');
        }
        return true;
      });
      const evaluated = await send('Runtime.evaluate', {expression: snap, returnByValue: true});
      if (evaluated.exceptionDetails) {
        console.error(JSON.stringify(evaluated.exceptionDetails));
        process.exit(1);
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
            "--window-size=1400,8000",
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


def _snapshot_html(port: int, html_name: str) -> dict[str, Any]:
    node = shutil.which("node")
    assert node
    page_url = (WEB / html_name).resolve().as_uri()
    result = subprocess.run(
        [node, "-e", _NODE_RUNNER, str(port), page_url, str(SNAPSHOT_JS)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    return payload


def _click(action_id: str, confidence: float = 0.95, target_confidence: float = 0.95) -> dict[str, Any]:
    return {
        "operation": "CLICK",
        "target": "1",
        "choice": action_id,
        "confidence": confidence,
        "target_confidence": target_confidence,
    }


def _submit_page() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    radios = []
    for group, label in (("q1", "Remote work"), ("q2", "Four day week")):
        for option, checked in (("Yes", "false"), ("No", "false")):
            radios.append(
                {
                    "id": f"{group}-{option}",
                    "node": len(radios) + 1,
                    "kind": "click",
                    "role": "radio",
                    "label": f"{label} → {option}",
                    "group": group,
                    "group_label": label,
                    "form_key": "f1",
                    "checked": checked,
                }
            )
    required = {
        "id": "name",
        "node": 50,
        "kind": "fill",
        "role": "textbox",
        "label": "Full name",
        "field_label": "Full name",
        "form_key": "f1",
        "required": True,
        "empty": True,
        "value": "",
    }
    optional = {
        "id": "dept",
        "node": 51,
        "kind": "select",
        "role": "combobox",
        "label": "Department → Pick",
        "field_label": "Department",
        "form_key": "f1",
        "empty": True,
        "value": "",
        "current_value": "Pick",
    }
    submit = {
        "id": "submit",
        "node": 90,
        "kind": "click",
        "role": "button",
        "label": "Send",
        "submits_form": True,
        "form_key": "f1",
    }
    actions = [*radios, required, optional, submit]
    return submit, actions


def test_incomplete_form_lists_unchecked_radios_and_empty_required_fields() -> None:
    submit, actions = _submit_page()
    result = evaluate_decision(
        _click(submit["id"]),
        CalibrationThresholds(),
        action=submit,
        page_actions=actions,
    )
    assert result.should_escalate is True
    assert result.trigger_type == "incomplete_form"
    assert "Remote work" in result.reason
    assert "Four day week" in result.reason
    assert "Full name" in result.reason
    assert "Department" not in result.reason.split("incomplete_form:", 1)[-1]


def test_empty_optional_select_does_not_block_a_complete_form() -> None:
    submit, actions = _submit_page()
    completed = []
    for action in actions:
        if action.get("role") == "radio" and action["id"].endswith("-Yes"):
            completed.append({**action, "checked": "true"})
        elif action.get("id") == "name":
            completed.append({**action, "empty": False, "value": "Ada"})
        else:
            completed.append(action)
    result = evaluate_decision(
        _click(submit["id"], confidence=0.8, target_confidence=0.8),
        CalibrationThresholds(),
        action=submit,
        page_actions=completed,
    )
    assert result.trigger_type != "incomplete_form"
    assert result.trigger_type != "low_conf_submit"
    assert result.should_escalate is False


def test_low_conf_submit_only_below_joint_threshold() -> None:
    submit, actions = _submit_page()
    ready = []
    for action in actions:
        if action.get("role") == "radio" and str(action["id"]).endswith("-Yes"):
            ready.append({**action, "checked": "true"})
        elif action.get("required"):
            ready.append({**action, "empty": False, "value": "Ada"})
        else:
            ready.append(action)
    low = evaluate_decision(
        _click(submit["id"], confidence=0.2, target_confidence=1.0),
        CalibrationThresholds(),
        action=submit,
        page_actions=ready,
    )
    assert low.should_escalate is True
    assert low.trigger_type == "low_conf_submit"
    high = evaluate_decision(
        _click(submit["id"], confidence=0.75, target_confidence=0.8),
        CalibrationThresholds(),
        action=submit,
        page_actions=ready,
    )
    assert high.trigger_type != "low_conf_submit"
    other = evaluate_decision(
        _click("next", confidence=0.2, target_confidence=1.0),
        CalibrationThresholds(),
        action={"id": "next", "kind": "click", "role": "button", "label": "Next", "submits_form": False},
        page_actions=ready,
    )
    assert other.trigger_type != "low_conf_submit"


def test_unchecked_radio_blocks_even_without_required_attribute() -> None:
    submit = {"id": "go", "kind": "click", "role": "button", "label": "Finish", "submits_form": True, "form_key": "loose"}
    actions = [
        {
            "id": "a",
            "node": 1,
            "role": "radio",
            "kind": "click",
            "group": "s1",
            "group_label": "Acts without a plan",
            "form_key": "loose",
            "checked": "false",
        },
        {
            "id": "b",
            "node": 2,
            "role": "radio",
            "kind": "click",
            "group": "s1",
            "group_label": "Acts without a plan",
            "form_key": "loose",
            "checked": "false",
        },
        submit,
    ]
    result = evaluate_decision(_click("go"), CalibrationThresholds(), action=submit, page_actions=actions)
    assert result.trigger_type == "incomplete_form"
    assert "Acts without a plan" in result.reason


def _run_blocked_submit(page: dict[str, Any], action: dict[str, Any]) -> Any:
    fake = type("B", (), {})()
    calls: list[str] = []

    def observe(screenshot: bool = True) -> dict[str, Any]:
        return page

    def act(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        calls.append("act")
        return {"executed": "should-not-run"}

    fake.observe = observe  # type: ignore[attr-defined]
    fake.act = act  # type: ignore[attr-defined]
    fake.fresh = lambda *_a, **_k: True  # type: ignore[attr-defined]
    fake.close = lambda: None  # type: ignore[attr-defined]
    decision = {
        "operation": "CLICK",
        "target": "1",
        "choice": action["id"],
        "confidence": 0.95,
        "target_confidence": 0.95,
        "latency_ms": 4,
        "model": "jev-test",
        "usage": {},
    }
    sup = SupervisorDecision(
        action_type="ASK_USER",
        question_for_user="Which answers should I use?",
        reasoning="form still incomplete",
        model="supervisor-test",
        latency_ms=3,
    )
    with (
        patch("reflex.agent.Browser", return_value=fake),
        patch("reflex.agent.interpret_user_message", return_value=UserMessageInterpretation()),
        patch("reflex.agent.choose_action", return_value=decision),
        patch("reflex.agent.resolve_escalation", return_value=sup) as resolve,
    ):
        session = ReflexSession(url=str(page.get("url") or "https://example.test/form"), goal="Fill and send the form", config=ReflexConfig())
        session.step()
    assert calls == []
    assert resolve.call_args.kwargs["trigger_type"] == "incomplete_form"
    return session


def test_qa_survey_and_hidden_choices_do_not_submit_empty(chromium_port: int) -> None:
    survey = _snapshot_html(chromium_port, "qa_survey.html")
    hidden = _snapshot_html(chromium_port, "qa_hidden_choices.html")

    survey_submit = next(action for action in survey["actions"] if action.get("submits_form"))
    hidden_submit = next(action for action in hidden["actions"] if action.get("submits_form"))
    assert survey_submit["kind"] == "click"
    assert hidden_submit["kind"] == "click"

    survey_gate = evaluate_decision(
        _click(survey_submit["id"]),
        CalibrationThresholds(),
        action=survey_submit,
        page_actions=survey["actions"],
    )
    hidden_gate = evaluate_decision(
        _click(hidden_submit["id"]),
        CalibrationThresholds(),
        action=hidden_submit,
        page_actions=hidden["actions"],
    )
    assert survey_gate.trigger_type == "incomplete_form"
    assert hidden_gate.trigger_type == "incomplete_form"
    assert "Uzaktan çalışma" in survey_gate.reason
    assert "Haftalık 4 gün" in survey_gate.reason
    assert "Çalıştığınız Departman" not in survey_gate.reason
    assert "Plan yapmadan" in hidden_gate.reason

    survey_session = _run_blocked_submit(survey, survey_submit)
    hidden_session = _run_blocked_submit(hidden, hidden_submit)
    assert survey_session.escalations[0]["trigger_type"] == "incomplete_form"
    assert hidden_session.escalations[0]["trigger_type"] == "incomplete_form"
    assert "Uzaktan çalışma" in survey_session.escalations[0]["reason"]
    assert "Plan yapmadan" in hidden_session.escalations[0]["reason"]
    assert survey_session.history == []
    assert hidden_session.history == []
