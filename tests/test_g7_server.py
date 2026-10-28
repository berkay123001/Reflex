"""Tests for Task G7: Server Security & UI Error Visibility."""

from __future__ import annotations

import io
import json
import re
import secrets
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

import reflex.server as server


def test_token_format_and_entropy():
    """Verify server.TOKEN is a 32-character hex string generated per process."""
    assert len(server.TOKEN) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", server.TOKEN)
    # Re-generating with secrets.token_hex(16) produces unique values
    assert secrets.token_hex(16) != server.TOKEN


def test_allow_remote_required_for_non_loopback():
    """main() must reject non-loopback --host if --allow-remote is not specified."""
    test_args = ["reflex", "--host", "0.0.0.0"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(ValueError, match="Binding to non-loopback host '0.0.0.0' requires the --allow-remote flag"):
            server.main()

    test_args_ip = ["reflex", "--host", "192.168.1.50"]
    with patch.object(sys, "argv", test_args_ip):
        with pytest.raises(ValueError, match="Binding to non-loopback host '192.168.1.50' requires the --allow-remote flag"):
            server.main()


@pytest.mark.parametrize("loopback", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_allowed_without_flag(loopback):
    """Loopback addresses do not require --allow-remote."""
    test_args = ["reflex", "--host", loopback, "--port", "9999"]
    with patch.object(sys, "argv", test_args):
        # Mock ThreadingHTTPServer so it does not actually bind or run
        with patch.object(server, "ThreadingHTTPServer") as mock_http:
            mock_server_instance = MagicMock()
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
            mock_http.return_value = mock_server_instance

            server.main()
            mock_http.assert_called_once()
            assert mock_http.call_args[0][0] == (loopback, 9999)


def test_allow_remote_flag_permits_non_loopback():
    """When --allow-remote is provided, non-loopback hosts are accepted and ALLOW_REMOTE is enabled."""
    test_args = ["reflex", "--host", "0.0.0.0", "--allow-remote", "--port", "9999"]
    with patch.object(sys, "argv", test_args):
        with patch.object(server, "ThreadingHTTPServer") as mock_http:
            mock_server_instance = MagicMock()
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
            mock_http.return_value = mock_server_instance

            try:
                server.main()
                assert server.ALLOW_REMOTE is True
                mock_http.assert_called_once_with(("0.0.0.0", 9999), server.ReflexHTTPHandler)
            finally:
                server.ALLOW_REMOTE = False


def test_config_endpoint_runs_under_lock():
    """POST /api/config must acquire _LOCK; if _LOCK is held, return 409."""
    handler = server.ReflexHTTPHandler.__new__(server.ReflexHTTPHandler)
    handler.headers = {
        "Host": "127.0.0.1:8788",
        "X-Reflex-Token": server.TOKEN,
        "Content-Length": "2",
    }
    handler.path = "/api/config"
    handler.rfile = io.BytesIO(b"{}")
    handler.wfile = io.BytesIO()

    sent_status = None
    sent_payload = None

    def fake_send(status, content, mime="application/json; charset=utf-8"):
        nonlocal sent_status, sent_payload
        sent_status = status
        sent_payload = content

    handler._send = fake_send

    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False
    with patch.object(server, "_LOCK", mock_lock):
        handler.do_POST()
        assert sent_status == 409
        assert "Another browser operation is currently in progress" in str(sent_payload)


def test_app_js_displays_error_visibly():
    """app.js runContinuousLoop catch block updates UI status badge and chat-messages visibly."""
    app_js_path = Path(__file__).resolve().parent.parent / "reflex" / "web" / "app.js"
    content = app_js_path.read_text(encoding="utf-8")

    # Verify catch block in runContinuousLoop
    assert 'console.error("Step error:", err);' in content
    assert '$("status-badge")' in content
    assert '$("status-text")' in content
    assert '$("chat-messages")' in content
    assert "status-pill error" in content
    assert "RUNTIME ERROR" in content


def test_allow_remote_emits_security_warning_to_stdout(capsys):
    """When --allow-remote is passed, main() prints a prominent security warning to stdout."""
    test_args = ["reflex", "--host", "0.0.0.0", "--allow-remote", "--port", "8788"]
    with patch.object(sys, "argv", test_args):
        with patch.object(server, "ThreadingHTTPServer") as mock_http:
            mock_server_instance = MagicMock()
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
            mock_http.return_value = mock_server_instance

            try:
                server.main()
            finally:
                server.ALLOW_REMOTE = False

    captured = capsys.readouterr()
    stdout_text = captured.out

    assert "SECURITY WARNING" in stdout_text
    assert "--allow-remote" in stdout_text
    assert "session token is served unauthenticated at '/'" in stdout_text
    assert "anyone on the network can control the browser" in stdout_text.lower()


def test_no_allow_remote_default_does_not_emit_security_warning(capsys):
    """When --allow-remote is not specified, security warning is not emitted."""
    test_args = ["reflex", "--host", "127.0.0.1", "--port", "8788"]
    with patch.object(sys, "argv", test_args):
        with patch.object(server, "ThreadingHTTPServer") as mock_http:
            mock_server_instance = MagicMock()
            mock_server_instance.serve_forever.side_effect = KeyboardInterrupt()
            mock_http.return_value = mock_server_instance

            server.main()

    captured = capsys.readouterr()
    stdout_text = captured.out

    assert "SECURITY WARNING" not in stdout_text
    assert "--allow-remote is enabled" not in stdout_text
    assert "unauthenticated" not in stdout_text
    assert "control the browser" not in stdout_text
    assert "Reflex Studio running at http://127.0.0.1:8788" in stdout_text

