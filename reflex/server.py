"""HTTP Studio Server for Reflex (`reflex/server.py`).

Serves the Reflex Studio frontend and JSON API endpoints:
- GET  /               -> index.html
- GET  /style.css      -> style.css
- GET  /app.js         -> app.js
- GET  /fixture.html   -> fixture.html (built-in obstacle / ASK_USER / CONFIRM_USER test portal)
- GET  /api/state      -> current session snapshot
- POST /api/start      -> start a new ReflexSession
- POST /api/step       -> execute one hybrid step (`ReflexSession.step()`)
- POST /api/reply      -> reply to `ASK_USER` or send multi-turn steering/follow-up
- POST /api/confirm    -> approve or reject an irreversible action (`CONFIRM_USER`)
- POST /api/reset      -> close the active browser session
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .agent import ReflexSession
from .browser.screencast import Screencast
from .config import (
    DEFAULT_DECISION_BACKEND,
    DEFAULT_RUN_MODE,
    DEFAULT_SUPERVISOR_MODEL,
    RUN_MODES,
    ReflexConfig,
    get_meta_api_key,
    load_env,
    run_mode_flags,
    run_mode_of,
    save_env_key,
)
from .memory import SiteMemoryStore

WEB_DIR = Path(__file__).resolve().parent / "web"
DEFAULT_PORT = int(os.environ.get("REFLEX_PORT", "8788"))
TOKEN = secrets.token_hex(16)
ALLOW_REMOTE: bool = False
_LOCK = threading.Lock()
_SESSION: ReflexSession | None = None
_SCREENCAST: Screencast | None = None


def _start_screencast(session: ReflexSession) -> None:
    global _SCREENCAST
    try:
        cdp_port = getattr(session.browser, "cdp_port", None)
        if cdp_port:
            target_id = getattr(session.browser, "target", None)
            _SCREENCAST = Screencast(cdp_port=cdp_port, target_id=target_id)
            _SCREENCAST.start()
    except Exception:
        _SCREENCAST = None


def close_session() -> None:
    global _SESSION, _SCREENCAST
    if _SCREENCAST is not None:
        try:
            _SCREENCAST.stop()
        except Exception:
            pass
        _SCREENCAST = None
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:
            pass
        _SESSION = None


def _key_status() -> dict[str, Any]:
    load_env()
    meta_k = get_meta_api_key()
    or_k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    return {
        "meta_api_configured": bool(meta_k),
        "meta_api_masked": (meta_k[:6] + "..." + meta_k[-4:]) if len(meta_k) > 10 else ("configured" if meta_k else ""),
        "openrouter_configured": bool(or_k),
        "openrouter_masked": (or_k[:8] + "..." + or_k[-4:]) if len(or_k) > 12 else ("configured" if or_k else ""),
    }


def current_state() -> dict[str, Any]:
    mem_store = _SESSION.site_memory if _SESSION is not None else SiteMemoryStore()
    if _SESSION is None:
        return {
            "status": "idle",
            "page": None,
            "elements": [],
            "history": [],
            "decisions": [],
            "escalations": [],
            "context": None,
            "config": _config_dump(ReflexConfig(**run_mode_flags(DEFAULT_RUN_MODE))),
            "site_memory": mem_store.all_domains(),
            "keys": _key_status(),
            "metrics": {
                "elapsed_ms": 0,
                "steps": 0,
                "s1_steps": 0,
                "s2_escalations": 0,
                "median_decision_ms": 0,
                "total_cost_usd": 0.0,
            },
        }
    snap = _SESSION.snapshot()
    snap["config"] = _config_dump(_SESSION.config)
    snap["site_memory"] = mem_store.all_domains()
    snap["keys"] = _key_status()
    return snap


def _config_dump(cfg: ReflexConfig) -> dict[str, Any]:
    """Serialize a config for the UI, adding the derived run-mode id."""
    return {**cfg.model_dump(), "mode": run_mode_of(cfg.s2_every_step, cfg.s1_strict, cfg.s2_action_packages)}


def _requested_mode(body: dict[str, Any]) -> str:
    """Use the posted run mode, or DEFAULT_RUN_MODE when it is omitted, blank or unknown."""
    raw = str(body.get("mode") or "").strip().lower()
    return raw if raw in RUN_MODES else DEFAULT_RUN_MODE


def _requested_decision_backend(body: dict[str, Any]) -> str:
    """Use the posted backend, or the config default when the field is omitted or blank."""
    raw = body.get("decision_backend", DEFAULT_DECISION_BACKEND)
    if raw is None or not str(raw).strip():
        return DEFAULT_DECISION_BACKEND
    return str(raw).strip()


def handle_api_post(action: str, body: dict[str, Any]) -> dict[str, Any]:
    global _SESSION
    if action == "start":
        url = str(body.get("url", "")).strip()
        goal = str(body.get("goal", "")).strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        if not (2 <= len(goal) <= 2000):
            raise ValueError("Goal must be between 2 and 2000 characters")

        cfg = ReflexConfig(
            decision_backend=_requested_decision_backend(body),
            supervisor_model=str(body.get("supervisor_model", DEFAULT_SUPERVISOR_MODEL)),
            interactive=True,  # Studio user is watching and can answer questions
            **run_mode_flags(_requested_mode(body)),
            require_confirmation_for_destructive=bool(body.get("require_confirmation_for_destructive", True)),
            verify_done_with_supervisor=bool(body.get("verify_done_with_supervisor", True)),
            enable_site_memory=bool(body.get("enable_site_memory", False)),
            max_steps=max(1, min(200, int(body.get("max_steps", 60)))),
            max_escalations=max(10, min(100, int(body.get("max_steps", 60)) // 2 + 10)),
        )
        close_session()
        _SESSION = ReflexSession(url, goal, cfg)
        _start_screencast(_SESSION)
        return current_state()

    if action == "reset":
        close_session()
        return current_state()

    if _SESSION is None:
        if action == "reply":
            msg = str(body.get("message", "")).strip()
            init_url = str(body.get("url", "")).strip() or "https://www.google.com"
            if not init_url.startswith(("http://", "https://")):
                init_url = "https://www.google.com"
            cfg = ReflexConfig(
                decision_backend=_requested_decision_backend(body),
                supervisor_model=str(body.get("supervisor_model", DEFAULT_SUPERVISOR_MODEL)),
                interactive=True,
                **run_mode_flags(_requested_mode(body)),
                max_steps=60,
                max_escalations=35,
            )
            close_session()
            _SESSION = ReflexSession(init_url, "Kullanıcı talimatı bekleniyor", cfg)
            _start_screencast(_SESSION)
            _SESSION.status = "waiting_user"
            _SESSION.reply_user(msg)
            return current_state()
        raise ValueError("No active session. Click Launch & Run first.")

    if action == "step":
        _SESSION.step()
        return current_state()

    if action == "reply":
        _SESSION.reply_user(str(body.get("message", "")))
        return current_state()

    if action == "confirm":
        _SESSION.confirm_action(
            approved=bool(body.get("approved", False)),
            alternative_instruction=str(body.get("alternative_instruction", "")),
        )
        return current_state()

    if action == "click_xy":
        x = float(body.get("x", 0))
        y = float(body.get("y", 0))
        _SESSION.click_xy(x, y)
        return current_state()

    raise ValueError(f"Unknown API endpoint: {action}")


class ReflexHTTPHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content: bytes | str, mime: str = "application/json; charset=utf-8") -> None:
        payload = content if isinstance(content, bytes) else content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def _valid_host(self) -> bool:
        if ALLOW_REMOTE:
            return True
        host = (self.headers.get("Host") or "").split(":", 1)[0].lower()
        return host in {"127.0.0.1", "localhost", "::1"}

    def do_GET(self) -> None:
        if not self._valid_host():
            return self._send(403, "Forbidden", "text/plain; charset=utf-8")

        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/state":
            if self.headers.get("X-Reflex-Token") != TOKEN:
                return self._send(403, json.dumps({"error": "Invalid session token"}))
            return self._send(200, json.dumps(current_state(), ensure_ascii=False))

        if path == "/api/stream":
            qs = parse_qs(parsed.query)
            tok = (qs.get("token") or [""])[0] or self.headers.get("X-Reflex-Token", "")
            if tok != TOKEN:
                return self._send(403, "Forbidden", "text/plain; charset=utf-8")
            return self._serve_mjpeg_stream()

        static_map = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "application/javascript; charset=utf-8"),
            "/fixture.html": ("fixture.html", "text/html; charset=utf-8"),
            "/qa/survey": ("qa_survey.html", "text/html; charset=utf-8"),
            "/qa/ranking": ("qa_ranking.html", "text/html; charset=utf-8"),
            "/qa/blog": ("qa_blog.html", "text/html; charset=utf-8"),
            "/qa/shop": ("qa_shop.html", "text/html; charset=utf-8"),
            "/qa/tournament": ("qa_tournament.html", "text/html; charset=utf-8"),
            "/qa/crm": ("qa_crm.html", "text/html; charset=utf-8"),
        }
        if path not in static_map:
            return self._send(404, "Not found", "text/plain; charset=utf-8")

        filename, mime = static_map[path]
        text = (WEB_DIR / filename).read_text(encoding="utf-8").replace("__TOKEN__", TOKEN)
        self._send(200, text, mime)

    def do_POST(self) -> None:
        if not self._valid_host() or self.headers.get("X-Reflex-Token") != TOKEN:
            return self._send(403, json.dumps({"error": "Invalid session token or host"}))

        if not _LOCK.acquire(timeout=45.0):
            return self._send(409, json.dumps({"error": "Another browser operation is currently in progress"}))

        try:
            endpoint = urlparse(self.path).path.removeprefix("/api/")
            length = int(self.headers.get("Content-Length", "0"))
            if not (0 <= length <= 65536):
                raise ValueError("Invalid request body length")
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8")) if raw else {}

            if endpoint == "config":
                meta_key = body.get("meta_api_key")
                if isinstance(meta_key, str) and meta_key.strip():
                    save_env_key("META_API_KEY", meta_key.strip())
                    save_env_key("LLAMA_API_KEY", meta_key.strip())
                or_key = body.get("openrouter_api_key")
                if isinstance(or_key, str) and or_key.strip():
                    save_env_key("OPENROUTER_API_KEY", or_key.strip())
                if body.get("clear_site_memory"):
                    mem = _SESSION.site_memory if _SESSION is not None else SiteMemoryStore()
                    mem.clear()
                backend = body.get("decision_backend")
                sup_model = body.get("supervisor_model")
                if _SESSION is not None:
                    if backend in ("local_turnike", "openrouter_jev"):
                        _SESSION.config.decision_backend = backend
                    if sup_model:
                        _SESSION.config.supervisor_model = str(sup_model)
                    if body.get("mode") in RUN_MODES:
                        flags = run_mode_flags(body["mode"])
                        _SESSION.config.s2_every_step = flags["s2_every_step"]
                        _SESSION.config.s1_strict = flags["s1_strict"]
                    if "require_confirmation_for_destructive" in body:
                        _SESSION.config.require_confirmation_for_destructive = bool(body["require_confirmation_for_destructive"])
                    if "verify_done_with_supervisor" in body:
                        _SESSION.config.verify_done_with_supervisor = bool(body["verify_done_with_supervisor"])
                    if "enable_site_memory" in body:
                        _SESSION.config.enable_site_memory = bool(body["enable_site_memory"])
                return self._send(200, json.dumps(current_state(), ensure_ascii=False))

            result = handle_api_post(endpoint, body)
            self._send(200, json.dumps(result, ensure_ascii=False))
        except (ValueError, RuntimeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}))
        except Exception as exc:
            self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            _LOCK.release()

    def _serve_mjpeg_stream(self) -> None:
        """Stream live browser frames over multipart/x-mixed-replace without freezing during active step/reply calls."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=reflexframe")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            after_seq = -1
            last_frame: bytes | None = None
            now = time.monotonic()
            last_sent_time = 0.0
            last_new_frame_time = now
            last_fallback_capture_time = 0.0

            while _SESSION is not None and _SESSION.browser.target is not None:
                frame: bytes | None = None
                sc = _SCREENCAST
                now = time.monotonic()

                if sc is not None and sc.is_running:
                    try:
                        seq, f = sc.wait_frame(after_seq=after_seq, timeout=0.1)
                        now = time.monotonic()
                        if f is not None and seq > after_seq:
                            after_seq = seq
                            frame = f
                            last_frame = f
                            last_new_frame_time = now
                    except Exception:
                        pass

                    # 3-second fallback: if screencast produces no new frames for 3s,
                    # fetch a fresh snapshot via capture_jpeg_bytes (at most once every 3s).
                    if frame is None and (now - last_new_frame_time >= 3.0):
                        if now - last_fallback_capture_time >= 3.0:
                            try:
                                fallback = _SESSION.browser.capture_jpeg_bytes(quality=65)
                                if fallback:
                                    frame = fallback
                                    last_frame = fallback
                                    last_fallback_capture_time = now
                            except Exception:
                                pass

                    # 1-second heartbeat: if no new frame arrives, resend last frame every ~1s.
                    if frame is None:
                        if last_frame is not None and (now - last_sent_time >= 1.0):
                            frame = last_frame
                        elif last_frame is None:
                            # Immediate initial frame on connection if screencast hasn't produced frame 0
                            try:
                                init_frame = _SESSION.browser.capture_jpeg_bytes(quality=65)
                                if init_frame:
                                    frame = init_frame
                                    last_frame = init_frame
                            except Exception:
                                pass
                else:
                    # Screencast inactive or failed: fallback to periodic capture_jpeg_bytes
                    try:
                        frame = _SESSION.browser.capture_jpeg_bytes(quality=65)
                    except Exception:
                        frame = None
                    if not frame:
                        time.sleep(0.09)
                        continue
                    time.sleep(0.09)

                if frame is None:
                    continue

                header = (
                    f"--reflexframe\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                last_sent_time = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *_args: Any) -> None:
        pass


def main() -> None:
    global ALLOW_REMOTE
    parser = argparse.ArgumentParser(description="Start Reflex Studio Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        default=False,
        help="Allow remote connections (required if --host is non-loopback)",
    )
    args = parser.parse_args()

    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback_hosts and not args.allow_remote:
        raise ValueError(
            f"Binding to non-loopback host '{args.host}' requires the --allow-remote flag."
        )
    if args.allow_remote:
        ALLOW_REMOTE = True
        print(
            "\n"
            "======================================================================\n"
            "SECURITY WARNING: --allow-remote is enabled!\n"
            "The session token is served unauthenticated at '/'.\n"
            "Anyone on the network can control the browser.\n"
            "Do NOT expose this server on untrusted or public networks.\n"
            "======================================================================\n",
            flush=True,
        )

    load_env()
    atexit.register(close_session)
    server = ThreadingHTTPServer((args.host, args.port), ReflexHTTPHandler)
    print(f"Reflex Studio running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
