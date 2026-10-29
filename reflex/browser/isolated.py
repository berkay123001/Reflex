# ruff: noqa: BLE001, S110
"""Isolated Chromium browser lifecycle management (`managed_isolated_browser`).

Guarantees that port 9333 (shared monitoring browser) is never used, configures
`BU_CDP_URL` and `browser_harness`'s import-time daemon `NAME` before session creation,
spawns an isolated Chromium instance with a temporary profile directory (headless or
headed/visible), and cleans up the process, daemon, profile directory, and environment
on exit.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger("reflex.browser.isolated")

FORBIDDEN_SHARED_PORT = 9333


def is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Check if a TCP port is currently open and accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


_ORIG_IS_PORT_OPEN = is_port_open


def _check_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Check port openness while honoring test patches on either `isolated` or `scripts.eval`."""
    if is_port_open is not _ORIG_IS_PORT_OPEN:
        return bool(is_port_open(host, port))
    eval_mod = sys.modules.get("scripts.eval")
    eval_fn = getattr(eval_mod, "is_port_open", None) if eval_mod is not None else None
    if callable(eval_fn) and eval_fn is not _ORIG_IS_PORT_OPEN:
        return bool(eval_fn(host, port))
    return is_port_open(host, port, timeout=timeout)


def find_free_port(host: str = "127.0.0.1") -> int:
    """Find an available ephemeral TCP port on `host`, never returning port 9333."""
    for _ in range(10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])
        if port != FORBIDDEN_SHARED_PORT:
            return port
    raise RuntimeError("Could not allocate a free TCP port distinct from 9333")


def _set_harness_daemon_name(name: str) -> dict[str, str]:
    """Point browser_harness's import-time daemon NAME at `name`; return previous values for restore."""
    previous: dict[str, str] = {}
    for module_name in ("browser_harness.admin", "browser_harness.helpers"):
        module = sys.modules.get(module_name)
        if module is None:
            with contextlib.suppress(ImportError):
                module = __import__(module_name, fromlist=["NAME"])
        if module is not None and hasattr(module, "NAME"):
            previous[module_name] = module.NAME
            module.NAME = name
    return previous


def _restore_harness_daemon_name(previous: dict[str, str]) -> None:
    """Restore browser_harness daemon NAME attributes captured by `_set_harness_daemon_name`."""
    for module_name, name in previous.items():
        module = sys.modules.get(module_name)
        if module is not None:
            module.NAME = name


@contextlib.contextmanager
def managed_isolated_browser(
    port: int | None = 9444,
    headless: bool = True,
) -> Generator[str, None, None]:
    """Provide an isolated Chromium browser instance and environment lifecycle.

    Guarantees:
    - Rejects port 9333 with ValueError.
    - Supports `port=None` or `port=0` to allocate a fresh ephemeral port automatically.
    - Sets BU_CDP_URL to http://127.0.0.1:{port} and BU_NAME to eval_{port}.
    - Updates browser_harness import-time daemon NAME before yielding (before ReflexSession init).
    - If no process is listening on {port}, launches an isolated Chromium subprocess
      with a dedicated temporary --user-data-dir (headless or visible/headed).
    - Cleanly shuts down the subprocess and removes the temporary profile directory on exit.
    - Restores previous environment variables and CDP hooks in finally:.
    """
    env_cdp = os.environ.get("BU_CDP_URL")
    if env_cdp:
        try:
            parsed_port = urlsplit(env_cdp).port
            if parsed_port == FORBIDDEN_SHARED_PORT:
                raise ValueError("Port 9333 is reserved for external monitoring; eval runner must not use port 9333.")
            if parsed_port and port is not None and int(port) == 9444:
                port = parsed_port
        except ValueError:
            raise
        except Exception:
            pass

    if port is None or int(port) == 0:
        resolved_port = find_free_port()
    else:
        resolved_port = int(port)

    if resolved_port == FORBIDDEN_SHARED_PORT:
        raise ValueError("Port 9333 is reserved for external monitoring; eval runner must not use port 9333.")

    cdp_url = f"http://127.0.0.1:{resolved_port}"
    daemon_name = f"eval_{resolved_port}"
    old_cdp_url = os.environ.get("BU_CDP_URL")
    old_bu_name = os.environ.get("BU_NAME")
    os.environ["BU_CDP_URL"] = cdp_url
    os.environ["BU_NAME"] = daemon_name
    # browser_harness reads BU_NAME once at import; without this the client keeps talking to the
    # "default" daemon (which may be attached to the shared 9333 browser) instead of eval_{port}.
    harness_names = _set_harness_daemon_name(daemon_name)

    old_ensure = None
    try:
        import reflex.browser.cdp as r_cdp

        old_ensure = r_cdp.ensure_chromium_cdp
        r_cdp.ensure_chromium_cdp = lambda *args, **kwargs: None
    except Exception:
        pass

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    proc: subprocess.Popen[Any] | None = None

    try:
        if not _check_port_open("127.0.0.1", resolved_port):
            binary = (
                shutil.which("chromium")
                or shutil.which("chromium-browser")
                or shutil.which("google-chrome")
            )
            if binary:
                temp_dir_obj = tempfile.TemporaryDirectory(prefix="reflex_eval_profile_")
                user_data_dir = temp_dir_obj.name

                cmd = [
                    binary,
                    f"--remote-debugging-port={resolved_port}",
                    f"--user-data-dir={user_data_dir}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-gpu",
                    "--disk-cache-size=1048576",
                    "--media-cache-size=1048576",
                    "--window-size=1280,900",
                    "--lang=tr-TR",
                ]
                if headless:
                    cmd.insert(1, "--headless=new")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if _check_port_open("127.0.0.1", resolved_port):
                        time.sleep(0.2)
                        break
                    if proc.poll() is not None:
                        logger.warning("Chromium process exited prematurely with code %s", proc.returncode)
                        break
                    time.sleep(0.1)
            else:
                logger.warning("No Chromium executable found in PATH; running without spawned browser.")

        yield cdp_url

    finally:
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            except Exception as exc:
                logger.warning("Error terminating Chromium subprocess: %s", exc)

        if temp_dir_obj is not None:
            try:
                temp_dir_obj.cleanup()
            except Exception as exc:
                logger.warning("Error cleaning up temporary directory: %s", exc)

        try:
            from browser_harness import admin

            if hasattr(admin, "stop_remote_daemon"):
                admin.stop_remote_daemon(daemon_name)
        except Exception:
            pass

        if old_ensure is not None:
            try:
                import reflex.browser.cdp as r_cdp

                r_cdp.ensure_chromium_cdp = old_ensure
            except Exception:
                pass

        if old_cdp_url is not None:
            os.environ["BU_CDP_URL"] = old_cdp_url
        else:
            os.environ.pop("BU_CDP_URL", None)

        if old_bu_name is not None:
            os.environ["BU_NAME"] = old_bu_name
        else:
            os.environ.pop("BU_NAME", None)
        _restore_harness_daemon_name(harness_names)
