"""Run modes: Studio, MCP and the eval CLI default to S2-only; the library default stays hybrid."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

from reflex.config import DEFAULT_RUN_MODE, RUN_MODES, ReflexConfig, run_mode_flags, run_mode_of
from reflex.mcp_server import reflex_start, reflex_stop
from reflex.memory import SiteMemoryStore
from reflex.server import close_session, current_state, handle_api_post

ROOT = Path(__file__).resolve().parents[1]


def test_default_run_mode_is_s2_only_and_library_default_stays_hybrid() -> None:
    assert DEFAULT_RUN_MODE == "s2_only"
    assert run_mode_flags(None) == {"s2_every_step": True, "s1_strict": False}
    assert run_mode_flags("unknown") == run_mode_flags(DEFAULT_RUN_MODE)
    assert run_mode_flags("strict") == {"s2_every_step": False, "s1_strict": True}
    assert run_mode_flags("hybrid") == {"s2_every_step": False, "s1_strict": False}
    cfg = ReflexConfig()
    assert run_mode_of(cfg.s2_every_step, cfg.s1_strict) == "hybrid"


def test_run_mode_of_round_trips_every_mode() -> None:
    for mode in RUN_MODES:
        flags = run_mode_flags(mode)
        assert run_mode_of(flags["s2_every_step"], flags["s1_strict"]) == mode


class _FakeSession:
    created: ClassVar[list[_FakeSession]] = []

    def __init__(self, url: str, goal: str, config: ReflexConfig) -> None:
        self.config = config
        self.site_memory = SiteMemoryStore()
        self.status = "ready"
        self.browser = None
        _FakeSession.created.append(self)

    def snapshot(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(), "status": self.status}

    def reply_user(self, _message: str) -> None:
        return None

    def close(self) -> None:
        return None


def test_studio_idle_state_reports_s2_only() -> None:
    close_session()
    cfg = current_state()["config"]
    assert cfg["mode"] == "s2_only"
    assert cfg["s2_every_step"] is True


def test_studio_start_and_chat_apply_requested_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reflex.server.ReflexSession", _FakeSession)
    _FakeSession.created = []
    close_session()
    try:
        state = handle_api_post("start", {"url": "https://example.test", "goal": "Find the price"})
        assert _FakeSession.created[-1].config.s2_every_step is True
        assert state["config"]["mode"] == "s2_only"

        handle_api_post("start", {"url": "https://example.test", "goal": "Find the price", "mode": "strict"})
        cfg = _FakeSession.created[-1].config
        assert (cfg.s2_every_step, cfg.s1_strict) == (False, True)

        close_session()
        handle_api_post("reply", {"message": "Find the price", "mode": "hybrid"})
        cfg = _FakeSession.created[-1].config
        assert (cfg.s2_every_step, cfg.s1_strict) == (False, False)
    finally:
        close_session()


class _ModeOptions(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_mode = False
        self.values: list[str] = []
        self.selected: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "select":
            self._in_mode = values.get("id") == "top-mode-select"
        elif tag == "option" and self._in_mode:
            self.values.append(str(values.get("value")))
            if "selected" in values:
                self.selected.append(str(values.get("value")))

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._in_mode = False


def test_studio_mode_selector_lists_every_mode_and_preselects_default() -> None:
    parser = _ModeOptions()
    parser.feed((ROOT / "reflex" / "web" / "index.html").read_text(encoding="utf-8"))
    assert sorted(parser.values) == sorted(RUN_MODES)
    assert parser.selected == [DEFAULT_RUN_MODE]


@pytest.mark.anyio
async def test_mcp_start_defaults_to_s2_only_and_accepts_explicit_mode() -> None:
    configs: list[ReflexConfig] = []

    def capture(url: str, goal: str, config: ReflexConfig) -> Any:
        configs.append(config)
        raise RuntimeError("captured config")

    with (
        patch("reflex.mcp_server.managed_isolated_browser") as mock_iso,
        patch("reflex.mcp_server.ReflexSession", side_effect=capture),
    ):
        mock_iso.return_value.__enter__.return_value = "http://127.0.0.1:9501"
        mock_iso.return_value.__exit__.return_value = None
        for mode in (None, "hybrid"):
            kwargs = {} if mode is None else {"mode": mode}
            res = await reflex_start(goal="Find the price", show_browser=False, **kwargs)
            await reflex_stop(res["session_id"])

        with pytest.raises(ValueError):
            await reflex_start(goal="Find the price", show_browser=False, mode="fast")

    assert [c.s2_every_step for c in configs] == [True, False]
    assert all(c.interactive for c in configs)
