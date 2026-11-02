"""Unit tests for Milestone 4 / Requirement R3b: Configurable Browser Language & Stealth Consistency.

Verifies:
1. ReflexConfig.browser_language defaults to "tr-TR".
2. Environment variable REFLEX_BROWSER_LANG="en-US" sets config.browser_language == "en-US".
3. Chrome launch flags in isolated.py and cdp.py receive --lang=<browser_language>.
4. navigator.languages script in cdp.py matches configured language.
5. Accept-Language header format in cdp.py matches configured language.
6. All tests run completely offline without external network or spawning real browsers in <1s.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflex.agent import ReflexAgent
from reflex.browser.cdp import (
    Browser,
    build_accept_language_header,
    build_stealth_init_script,
    ensure_chromium_cdp,
)
from reflex.browser.isolated import managed_isolated_browser
from reflex.config import ReflexConfig, get_default_browser_language

# ==============================================================================
# 1. ReflexConfig.browser_language Defaults & Precedence
# ==============================================================================


def test_reflex_config_browser_language_default_is_tr(monkeypatch: pytest.MonkeyPatch):
    """When REFLEX_BROWSER_LANG is unset, ReflexConfig.browser_language defaults to 'tr-TR'."""
    monkeypatch.delenv("REFLEX_BROWSER_LANG", raising=False)
    cfg = ReflexConfig()
    assert cfg.browser_language == "tr-TR"
    assert get_default_browser_language() == "tr-TR"


def test_reflex_config_browser_language_env_var_override(monkeypatch: pytest.MonkeyPatch):
    """When REFLEX_BROWSER_LANG is set, ReflexConfig.browser_language picks it up."""
    monkeypatch.setenv("REFLEX_BROWSER_LANG", "en-US")
    cfg = ReflexConfig()
    assert cfg.browser_language == "en-US"
    assert get_default_browser_language() == "en-US"


def test_reflex_config_browser_language_constructor_overrides_env(monkeypatch: pytest.MonkeyPatch):
    """Explicit constructor argument overrides REFLEX_BROWSER_LANG environment variable."""
    monkeypatch.setenv("REFLEX_BROWSER_LANG", "en-US")
    cfg = ReflexConfig(browser_language="fr-FR")
    assert cfg.browser_language == "fr-FR"


def test_reflex_config_browser_language_empty_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    """Empty or whitespace-only REFLEX_BROWSER_LANG falls back to 'tr-TR'."""
    monkeypatch.setenv("REFLEX_BROWSER_LANG", "   ")
    cfg = ReflexConfig()
    assert cfg.browser_language == "tr-TR"


# ==============================================================================
# 2. Chrome Launch Flags: isolated.py & cdp.py
# ==============================================================================


def test_managed_isolated_browser_launch_flag_default_tr(monkeypatch: pytest.MonkeyPatch):
    """managed_isolated_browser passes --lang=tr-TR by default."""
    monkeypatch.delenv("REFLEX_BROWSER_LANG", raising=False)
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with patch("subprocess.Popen", side_effect=fake_popen):
        try:
            with managed_isolated_browser(port=9998):
                pass
        except RuntimeError:
            pass

    assert captured_cmd, "Subprocess command must be captured"
    assert "--lang=tr-TR" in captured_cmd
    lang_flags = [arg for arg in captured_cmd if arg.startswith("--lang=")]
    assert len(lang_flags) == 1, f"Expected exactly 1 --lang flag, found: {lang_flags}"


def test_managed_isolated_browser_launch_flag_env_en_us(monkeypatch: pytest.MonkeyPatch):
    """managed_isolated_browser passes --lang=en-US when REFLEX_BROWSER_LANG=en-US."""
    monkeypatch.setenv("REFLEX_BROWSER_LANG", "en-US")
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with patch("subprocess.Popen", side_effect=fake_popen):
        try:
            with managed_isolated_browser(port=9998):
                pass
        except RuntimeError:
            pass

    assert captured_cmd, "Subprocess command must be captured"
    assert "--lang=en-US" in captured_cmd
    assert "--lang=tr-TR" not in captured_cmd


def test_managed_isolated_browser_launch_flag_explicit_param():
    """managed_isolated_browser accepts browser_language parameter directly."""
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with patch("subprocess.Popen", side_effect=fake_popen):
        try:
            with managed_isolated_browser(port=9998, browser_language="de-DE"):
                pass
        except RuntimeError:
            pass

    assert captured_cmd, "Subprocess command must be captured"
    assert "--lang=de-DE" in captured_cmd


def test_ensure_chromium_cdp_launch_flag_default_tr(monkeypatch: pytest.MonkeyPatch):
    """ensure_chromium_cdp passes --lang=tr-TR by default."""
    monkeypatch.delenv("REFLEX_BROWSER_LANG", raising=False)
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        patch("reflex.browser.cdp._port_open", return_value=False),
        patch("shutil.which", return_value="/usr/bin/chromium"),
    ):
        try:
            ensure_chromium_cdp(port=9996)
        except RuntimeError:
            pass

    assert captured_cmd, "Subprocess command must be captured"
    assert "--lang=tr-TR" in captured_cmd


def test_ensure_chromium_cdp_launch_flag_env_en_us(monkeypatch: pytest.MonkeyPatch):
    """ensure_chromium_cdp passes --lang=en-US when REFLEX_BROWSER_LANG=en-US."""
    monkeypatch.setenv("REFLEX_BROWSER_LANG", "en-US")
    captured_cmd: list[str] = []

    def fake_popen(cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        raise RuntimeError("Stop after capturing cmd")

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        patch("reflex.browser.cdp._port_open", return_value=False),
        patch("shutil.which", return_value="/usr/bin/chromium"),
    ):
        try:
            ensure_chromium_cdp(port=9996)
        except RuntimeError:
            pass

    assert captured_cmd, "Subprocess command must be captured"
    assert "--lang=en-US" in captured_cmd
    assert "--lang=tr-TR" not in captured_cmd


# ==============================================================================
# 3. navigator.languages Script Generation in cdp.py
# ==============================================================================


def test_build_stealth_init_script_default_tr():
    """build_stealth_init_script for tr-TR produces navigator.languages prioritizing tr-TR."""
    script = build_stealth_init_script("tr-TR")
    assert "navigator, 'languages'" in script
    assert "'tr-TR', 'tr', 'en-US', 'en'" in script


def test_build_stealth_init_script_en_us():
    """build_stealth_init_script for en-US produces navigator.languages prioritizing en-US without tr-TR."""
    script = build_stealth_init_script("en-US")
    assert "navigator, 'languages'" in script
    assert "'en-US', 'en'" in script
    assert "'tr-TR'" not in script


def test_build_stealth_init_script_custom_locale():
    """build_stealth_init_script supports custom locales such as de-DE."""
    script = build_stealth_init_script("de-DE")
    assert "'de-DE', 'de', 'en-US', 'en'" in script


# ==============================================================================
# 4. Accept-Language Header Formatting & Emulation
# ==============================================================================


def test_build_accept_language_header_tr():
    """Accept-Language header for tr-TR matches standard Turkish Chrome preference."""
    hdr = build_accept_language_header("tr-TR")
    assert hdr == "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"


def test_build_accept_language_header_en_us():
    """Accept-Language header for en-US matches standard US Chrome preference."""
    hdr = build_accept_language_header("en-US")
    assert hdr == "en-US,en;q=0.9"


def test_build_accept_language_header_custom_locale():
    """Accept-Language header for other locales includes fallback to en-US."""
    hdr = build_accept_language_header("fr-FR")
    assert hdr.startswith("fr-FR,fr;q=0.9")
    assert "en-US" in hdr


def test_browser_cdp_setUserAgentOverride_receives_configured_accept_language():
    """Browser initialization sets acceptLanguage in Emulation.setUserAgentOverride matching browser_language."""
    captured_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_cdp_call(method: str, **params: Any) -> dict[str, Any]:
        captured_calls.append((method, params))
        if method == "Target.createTarget":
            return {"targetId": "target_1"}
        if method == "Target.attachToTarget":
            return {"sessionId": "session_1"}
        if method == "Browser.getVersion":
            return {"userAgent": "Mozilla/5.0 Chrome/130.0.0.0"}
        return {}

    with (
        patch("reflex.browser.cdp.ensure_chromium_cdp"),
        patch("reflex.browser.cdp.ensure_daemon"),
        patch("reflex.browser.cdp.cdp", side_effect=fake_cdp_call),
        patch.object(Browser, "call", side_effect=fake_cdp_call),
        patch.object(Browser, "evaluate", return_value="Mozilla/5.0 Chrome/130.0.0.0"),
        patch.object(Browser, "navigate"),
    ):
        Browser("about:blank", browser_language="en-US")

        # 1. Verify Page.addScriptToEvaluateOnNewDocument received en-US script
        add_script_calls = [params for m, params in captured_calls if m == "Page.addScriptToEvaluateOnNewDocument"]
        assert len(add_script_calls) >= 1
        script_src = add_script_calls[0].get("source", "")
        assert "'en-US', 'en'" in script_src
        assert "'tr-TR'" not in script_src

        # 2. Verify Emulation.setUserAgentOverride received en-US acceptLanguage
        ua_calls = [params for m, params in captured_calls if m == "Emulation.setUserAgentOverride"]
        assert len(ua_calls) >= 1
        assert ua_calls[0].get("acceptLanguage") == "en-US,en;q=0.9"


def test_reflex_agent_forwards_browser_language_to_browser():
    """ReflexAgent forwards config.browser_language into Browser instantiation kwargs."""
    config = ReflexConfig(browser_language="en-US")

    captured_kwargs: dict[str, Any] = {}

    def fake_browser_init(url: str, **kwargs: Any) -> Any:
        nonlocal captured_kwargs
        captured_kwargs = dict(kwargs)
        mock_b = MagicMock()
        mock_b.observe.return_value = {"url": url, "actions": []}
        return mock_b

    with (
        patch("reflex.agent.Browser", side_effect=fake_browser_init),
        patch("reflex.agent.interpret_user_message") as mock_interp,
    ):
        mock_interp_res = MagicMock()
        mock_interp_res.model_dump.return_value = {}
        mock_interp_res.allow_arbitrary_choices = False
        mock_interp_res.constraints = []
        mock_interp_res.plan = None
        mock_interp.return_value = mock_interp_res

        ReflexAgent("https://example.com", goal="Search", config=config)
        assert captured_kwargs.get("browser_language") == "en-US"
