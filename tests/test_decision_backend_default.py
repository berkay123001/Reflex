"""The default System 1 backend is openrouter_jev, and local_turnike stays selectable."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from reflex.config import DEFAULT_DECISION_BACKEND, ReflexConfig
from reflex.memory import SiteMemoryStore
from reflex.server import close_session, current_state, handle_api_post

ROOT = Path(__file__).resolve().parents[1]


class _S1Options(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._select_id: str | None = None
        self.options: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "select":
            self._select_id = values.get("id")
        elif tag == "option" and self._select_id == "top-s1-select":
            self.options.append(values)

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._select_id = None


class _FakeSession:
    def __init__(self, url: str, goal: str, config: ReflexConfig) -> None:
        self.url = url
        self.goal = goal
        self.config = config
        self.site_memory = SiteMemoryStore()
        self.status = "ready"
        self.created.append(self)

    created: list[_FakeSession] = []

    def snapshot(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(), "status": self.status}

    def reply_user(self, _message: str) -> None:
        return None

    def close(self) -> None:
        return None


def test_config_default_is_openrouter_jev() -> None:
    assert DEFAULT_DECISION_BACKEND == "openrouter_jev"
    assert ReflexConfig().decision_backend == "openrouter_jev"


def test_idle_studio_state_uses_the_config_default() -> None:
    close_session()
    state = current_state()
    assert state["status"] == "idle"
    assert state["config"]["decision_backend"] == "openrouter_jev"


def test_omitted_studio_backend_defaults_and_explicit_turnike_is_kept(monkeypatch) -> None:
    monkeypatch.setattr("reflex.server.ReflexSession", _FakeSession)
    close_session()
    _FakeSession.created = []
    try:
        handle_api_post("start", {"url": "https://example.test/form", "goal": "Fill the form"})
        assert _FakeSession.created[-1].config.decision_backend == "openrouter_jev"
        close_session()
        handle_api_post(
            "start",
            {"url": "https://example.test/form", "goal": "Fill the form", "decision_backend": None},
        )
        assert _FakeSession.created[-1].config.decision_backend == "openrouter_jev"
        close_session()
        handle_api_post(
            "start",
            {"url": "https://example.test/form", "goal": "Fill the form", "decision_backend": "  "},
        )
        assert _FakeSession.created[-1].config.decision_backend == "openrouter_jev"
        close_session()
        handle_api_post(
            "start",
            {
                "url": "https://example.test/form",
                "goal": "Fill the form",
                "decision_backend": "local_turnike",
            },
        )
        assert _FakeSession.created[-1].config.decision_backend == "local_turnike"
        close_session()
        handle_api_post("reply", {"message": "continue", "url": "https://example.test/form"})
        assert _FakeSession.created[-1].config.decision_backend == "openrouter_jev"
        close_session()
        handle_api_post(
            "reply",
            {"message": "continue", "url": "https://example.test/form", "decision_backend": "local_turnike"},
        )
        assert _FakeSession.created[-1].config.decision_backend == "local_turnike"
    finally:
        close_session()


def test_studio_control_selects_jev_and_still_lists_turnike() -> None:
    parser = _S1Options()
    parser.feed((ROOT / "reflex/web/index.html").read_text(encoding="utf-8"))
    values = [option.get("value") for option in parser.options]
    assert values == ["local_turnike", "openrouter_jev"]
    selected = [option.get("value") for option in parser.options if "selected" in option]
    assert selected == ["openrouter_jev"]


def test_app_js_fallback_uses_jev_and_keeps_an_explicit_choice() -> None:
    source = (ROOT / "reflex/web/app.js").read_text(encoding="utf-8")
    line = next(
        item
        for item in source.splitlines()
        if "decision_backend" in item and "top-s1-select" in item and "||" in item
    )
    expression = line.split("decision_backend:", 1)[1].strip().rstrip(",")
    script = """
const vm = require("vm");
const expr = process.argv[1];
function run(selectValue) {
  const $ = () => (selectValue === undefined ? null : {value: selectValue});
  return vm.runInNewContext(expr, {$});
}
process.stdout.write(JSON.stringify({
  missing: run(undefined),
  blank: run(""),
  turnike: run("local_turnike"),
  jev: run("openrouter_jev"),
}));
"""
    result = subprocess.run(
        ["node", "-e", script, expression],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "missing": "openrouter_jev",
        "blank": "openrouter_jev",
        "turnike": "local_turnike",
        "jev": "openrouter_jev",
    }


def test_eval_cli_defaults_to_jev_and_accepts_turnike(monkeypatch, tmp_path: Path) -> None:
    from scripts.eval import main

    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    seen: dict[str, str] = {}

    def run_eval(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        seen["backend"] = kwargs["config"].decision_backend
        return [], {"failed_tasks": 0}

    @contextlib.contextmanager
    def quiet(*_args: Any, **_kwargs: Any):
        yield "http://127.0.0.1:9"

    monkeypatch.setattr("scripts.eval.run_eval", run_eval)
    monkeypatch.setattr("scripts.eval.managed_web_server", quiet)
    monkeypatch.setattr("scripts.eval.managed_isolated_browser", quiet)

    assert main([str(tasks), "--cdp-port", "9444", "--quiet"]) == 0
    assert seen["backend"] == "openrouter_jev"
    assert main([str(tasks), "--decision-backend", "local_turnike", "--cdp-port", "9445", "--quiet"]) == 0
    assert seen["backend"] == "local_turnike"

    help_text = subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    assert "local_turnike" in help_text
    assert "openrouter_jev" in help_text
