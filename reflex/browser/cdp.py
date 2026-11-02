"""CDP Browser controller with atomic DOM snapshots and recovery primitives."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from browser_harness.admin import ensure_daemon
from browser_harness.helpers import cdp

logger = logging.getLogger(__name__)

READ_STATE = Path(__file__).with_name("snapshot.js").read_text(encoding="utf-8")
MARKER = f"(() => {{ const state={READ_STATE}; return state?.marker ?? null; }})()"
DEFAULT_CDP_PORT = 9333
IFRAME_MIN_WIDTH = 200
IFRAME_MIN_HEIGHT = 100
MAX_IFRAMES = 5
# Page.navigate only answers once the server responds; slow or dead sites exceed the 5 s IPC default.
NAVIGATE_RESPONSE_TIMEOUT_S = 15.0
_AD_BLOCK_URLS = [
    "*://*.googlesyndication.com/*",
    "*://*.doubleclick.net/*",
    "*://*.googleadservices.com/*",
    "*://*.adservice.google.com/*",
    "*://*.adnxs.com/*",
    "*://*.taboola.com/*",
    "*://*.outbrain.com/*",
    "*://*.criteo.com/*",
    "*://*.rubiconproject.com/*",
    "*://*.pubmatic.com/*",
    "*://*.popads.net/*",
    "*://*.moatads.com/*",
    "*://*.amazon-adsystem.com/*",
    "*://*.yandex.ru/ads/*",
]

INTERACTIVE_AX_ROLES: set[str] = {
    "button", "link", "textbox", "searchbox", "combobox", "listbox",
    "option", "checkbox", "radio", "switch", "slider", "spinbutton",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "treeitem",
}


def format_browser_languages(tag: str | None) -> list[str]:
    """Return ordered list of language tags for navigator.languages.

    Given a BCP 47 language tag (e.g. 'tr-TR', 'en-US', 'de-DE', 'tr'):
    - Normalizes underscores to hyphens (e.g. 'tr_TR' -> 'tr-TR')
    - Standardizes case (lower-case language, upper-case 2-letter region)
    - Appends base language if tag has a region (e.g. 'tr-TR' -> 'tr')
    - If primary language is not English, appends English fallbacks (['en-US', 'en'])
    - Deduplicates while preserving order.
    """
    clean = (tag or "").strip().replace("_", "-")
    if not clean:
        clean = "tr-TR"

    parts = clean.split("-")
    base = parts[0].lower()
    if len(parts) == 1:
        normalized_tag = base
    elif len(parts) == 2 and len(parts[1]) == 2:
        normalized_tag = f"{base}-{parts[1].upper()}"
    else:
        normalized_tag = clean

    langs: list[str] = [normalized_tag]
    if base != normalized_tag.lower() and base not in langs:
        langs.append(base)

    if base != "en":
        for fallback in ("en-US", "en"):
            if fallback not in langs:
                langs.append(fallback)
    elif normalized_tag.lower() == "en":
        if "en-US" not in langs:
            langs.append("en-US")

    return langs


def format_accept_language(languages: list[str] | str) -> str:
    """Format an Accept-Language HTTP header string with descending q-values.

    Example:
        ['tr-TR', 'tr', 'en-US', 'en'] -> 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
        ['en-US', 'en'] -> 'en-US,en;q=0.9'
    """
    if isinstance(languages, str):
        langs = format_browser_languages(languages)
    else:
        langs = list(languages)
    if not langs:
        return "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"

    entries: list[str] = [langs[0]]
    q = 0.9
    for lang in langs[1:]:
        entries.append(f"{lang};q={q:.1f}")
        q = max(0.1, round(q - 0.1, 1))
    return ",".join(entries)


def build_accept_language_header(languages: list[str] | str) -> str:
    """Format an Accept-Language HTTP header string with descending q-values (alias)."""
    return format_accept_language(languages)


def build_stealth_init_script(languages: list[str] | str | None = None) -> str:
    """Generate the stealth initialization script with dynamic navigator.languages."""
    if isinstance(languages, str):
        langs = format_browser_languages(languages)
    elif languages is not None:
        langs = list(languages)
    else:
        langs = ["tr-TR", "tr", "en-US", "en"]
    langs_repr = "[" + ", ".join(repr(x) for x in langs) + "]"
    return f"""(() => {{
  try {{
    Object.defineProperty(navigator, 'webdriver', {{ get: () => undefined }});
    window.chrome = window.chrome || {{ runtime: {{}}, app: {{}}, csi: () => ({{}}), loadTimes: () => ({{}}) }};
    Object.defineProperty(navigator, 'languages', {{ get: () => {langs_repr} }});
    Object.defineProperty(navigator, 'plugins', {{ get: () => [1, 2, 3, 4, 5] }});
    if (!window.outerWidth) Object.defineProperty(window, 'outerWidth', {{ get: () => window.innerWidth || 1280 }});
    if (!window.outerHeight) Object.defineProperty(window, 'outerHeight', {{ get: () => (window.innerHeight || 800) + 85 }});
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {{
      if (p === 37445) return 'Intel Inc.';
      if (p === 37446) return 'Intel Iris OpenGL Engine';
      return getParam.apply(this, arguments);
    }};
    window.open = function(url) {{
      if (url && typeof url === 'string' && url !== 'about:blank' && !url.startsWith('javascript:')) {{
        window.location.assign(url);
      }}
      return window;
    }};
    window.__reflexDialogs = window.__reflexDialogs || [];
    const recordDialog = (kind, msg) => {{
      try {{
        const m = String(msg ?? '').replace(/\\s+/g, ' ').trim();
        if (m) {{
          window.__reflexDialogs.push('[' + kind + '] ' + m.slice(0, 400));
          if (window.__reflexDialogs.length > 5) window.__reflexDialogs.shift();
        }}
      }} catch (_) {{}}
    }};
    window.alert = function(msg) {{ recordDialog('Alert', msg); }};
    window.confirm = function(msg) {{ recordDialog('Confirm', msg); return false; }};
    window.prompt = function(msg) {{ recordDialog('Prompt', msg); return null; }};
  }} catch (_) {{}}
}})();"""


_STEALTH_INIT_SCRIPT = build_stealth_init_script(["tr-TR", "tr", "en-US", "en"])


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def ensure_chromium_cdp(
    port: int | None = None,
    browser_language: str | None = None,
) -> None:
    """Ensure a headless Chromium instance is listening on `port` and `BU_CDP_URL` is set."""
    target_port = port if port is not None else DEFAULT_CDP_PORT
    expected_url = f"http://127.0.0.1:{target_port}"
    existing_url = os.environ.get("BU_CDP_URL")
    if existing_url and existing_url != expected_url and port is not None:
        msg = f"BU_CDP_URL is already set to {existing_url!r}, which differs from expected {expected_url!r}"
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    if existing_url:
        try:
            parsed_port = urlsplit(existing_url).port
            if parsed_port:
                target_port = parsed_port
        except (ValueError, AttributeError):
            pass
    port = target_port
    expected_url = f"http://127.0.0.1:{port}"
    os.environ.setdefault("BU_CDP_URL", expected_url)
    if _port_open("127.0.0.1", port):
        return
    binary = (
        shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("google-chrome")
    )
    if not binary:
        return
    raw_lang = (browser_language or os.environ.get("REFLEX_BROWSER_LANG", "tr-TR")).strip() or "tr-TR"
    lang = format_browser_languages(raw_lang)[0]
    profile_dir = Path("/tmp/reflex_chromium_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path("/tmp/reflex_chromium.log").open("a", encoding="utf-8")
    cmd = [
        binary,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
        "--disable-gpu",
        "--disk-cache-size=1048576",
        "--media-cache-size=1048576",
        "--window-size=1280,900",
        f"--lang={lang}",
    ]
    if os.environ.get("REFLEX_HEADED") not in ("1", "true", "True"):
        cmd.insert(1, "--headless=new")
    subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _port_open("127.0.0.1", port):
            time.sleep(0.25)
            return
        time.sleep(0.15)


CLICK_TEXT_QUIET_S = 0.6
CLICK_TEXT_CAP_S = 10.0
# Growth after in-page settle arms the text poll only inside this window.
CLICK_TEXT_ARM_S = 0.25
# Quiet clicks stay under 200 ms. The trailing length read sits inside the margin.
CLICK_SETTLE_BUDGET_S = 0.18
_PAGE_TEXT_LEN = "(document.body && (document.body.innerText || '').length) || 0"
_TIME_TOKEN = re.compile(r"\b\d+\s*(?:s|sn|sec|ms)\b", re.IGNORECASE)
# Live accessible name from the snapshot cache. Empty when the node or its guard is gone.
_ACCESSIBLE_NAME = """(nodeId => {
  const snapshot = window.__jevFast;
  const node = snapshot?.nodes?.get(nodeId);
  if (!node || !snapshot.guard) return null;
  const guard = snapshot.guard(node);
  const name = guard && guard[2];
  return typeof name === 'string' ? name : null;
})"""


def _norm_accessible_label(value: str) -> str:
    cleaned = _TIME_TOKEN.sub("", value or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _recorded_accessible_name(action: dict[str, Any], page: dict[str, Any] | None) -> str:
    """Raw name from the decision, never the decorated label."""
    raw = action.get("accessible_name")
    if isinstance(raw, str) and raw.strip():
        return raw
    node = action.get("node")
    guards = (page or {}).get("guards") or {}
    guard = None
    if type(node) is int:
        guard = guards.get(str(node))
        if guard is None:
            guard = guards.get(node)
    if isinstance(guard, (list, tuple)) and len(guard) > 2 and isinstance(guard[2], str) and guard[2].strip():
        return str(guard[2])
    return ""


class StalePage(ValueError):
    """Raised when the page DOM mutated after observation before execution."""


class EvaluationTimeout(StalePage):
    """Raised when Runtime.evaluate times out even after dismissing any blocking JS dialog."""


class TargetCoveredError(RuntimeError):
    """Raised when the target element is obscured by a modal/overlay at hit-test time."""


class NavigationTimeout(RuntimeError):
    """Raised when Page.navigate gets no response in time (site down or hanging); the tab stays usable."""


_NAVIGATION_CDP_ERRORS = (
    "Inspected target navigated or closed",
    "Cannot find context with specified id",
    "Execution context was destroyed",
    "Session with given id not found",
    "Target closed",
    "Session already detached",
    "Target with given id not found",
    "No target with given id found",
    "Frame was detached",
)


def _is_navigation_cdp_error(exc: BaseException) -> bool:
    """Return True if a CDP exception represents an in-flight navigation or destroyed execution context."""
    msg = str(exc).lower()
    return any(marker.lower() in msg for marker in _NAVIGATION_CDP_ERRORS)


_EPHEMERAL_QUERY_PARAMS = {
    "sxsrf", "ei", "ved", "iflsig", "sca_esv", "gs_lp", "uact", "oq",
    "sclient", "atvm", "ictx", "sa", "source", "sourceid", "aqs",
    "fbclid", "gclid", "msclkid", "_ga", "_gl", "ref", "ref_",
}


def canonical_url(raw_url: str) -> str:
    """Normalize URL by stripping ephemeral tracking query parameters and `/webhp` aliases."""
    u = (raw_url or "").strip()
    if not u:
        return ""
    try:
        parts = urlsplit(u)
        path = parts.path or "/"
        if "google." in parts.netloc.lower() and path == "/webhp":
            path = "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        kept_qs = [
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=False)
            if k.lower() not in _EPHEMERAL_QUERY_PARAMS and not k.lower().startswith("utm_")
        ]
        kept_qs.sort()
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(kept_qs), ""))
    except Exception:
        return u.split("#")[0].rstrip("/").lower()


def fingerprint(state: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 digest of semantic page state (excluding bounding boxes and ephemeral node IDs)."""
    actions_semantic = [
        {
            k: (canonical_url(str(v)) if k == "href" else v)
            for k, v in a.items()
            if k not in ("rect", "node", "guard")
        }
        for a in state.get("actions", [])
    ]
    content = {
        "url": canonical_url(str(state.get("url", ""))),
        "text": state.get("text", ""),
        "actions": actions_semantic,
        "scroll": state.get("scroll", {}),
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()


def clean_surrogates(data: Any) -> Any:
    """Recursively clean unpaired Unicode surrogates from strings, dicts, lists, and tuples."""
    if isinstance(data, str):
        try:
            data.encode("utf-8")
            return data
        except UnicodeEncodeError:
            return data.encode("utf-16", "surrogatepass").decode("utf-16", errors="ignore")
    if isinstance(data, dict):
        return {
            clean_surrogates(k): clean_surrogates(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [clean_surrogates(x) for x in data]
    if isinstance(data, tuple):
        return tuple(clean_surrogates(x) for x in data)
    return data


STEALTH_MIN_DELAY_S: float = 0.05
STEALTH_MAX_DELAY_S: float = 0.25


def _stealth_delay(
    target: Any = None,
    min_s: float = STEALTH_MIN_DELAY_S,
    max_s: float = STEALTH_MAX_DELAY_S,
) -> float:
    """Sleep for a randomized duration between min_s and max_s if target enables light stealth, returning duration."""
    enabled = False
    if target is None:
        val = os.environ.get("REFLEX_LIGHT_STEALTH", "").strip().lower()
        enabled = val in {"1", "true", "yes", "on"}
    elif isinstance(target, bool):
        enabled = target
    elif hasattr(target, "light_stealth"):
        enabled = bool(getattr(target, "light_stealth"))
    elif isinstance(target, (int, float)):
        min_s = float(target)
        val = os.environ.get("REFLEX_LIGHT_STEALTH", "").strip().lower()
        enabled = val in {"1", "true", "yes", "on"}

    if enabled:
        delay = random.uniform(min_s, max_s)
        time.sleep(delay)
        return delay
    return 0.0


class Browser:
    """Single-tab CDP browser session with fresh-state verification and recovery actions."""

    _frame_sessions: dict[str, str] = {}
    _iframe_node_map: dict[int, dict[str, Any]] = {}
    ax_supplement: bool = False
    light_stealth: bool = False
    browser_language: str = "tr-TR"
    browser_languages: list[str] = ["tr-TR", "tr", "en-US", "en"]
    accept_language: str = "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7"

    def __init__(
        self,
        url: str,
        *,
        width: int = 1120,
        height: int = 780,
        port: int | None = None,
        ax_supplement: bool | None = None,
        light_stealth: bool | None = None,
        browser_language: str | None = None,
        config: Any = None,
    ):
        if ax_supplement is not None:
            self.ax_supplement: bool = bool(ax_supplement)
        elif config is not None and hasattr(config, "ax_supplement"):
            self.ax_supplement = bool(config.ax_supplement)
        else:
            val = os.environ.get("REFLEX_AX_SUPPLEMENT", "1").strip().lower()
            self.ax_supplement = val not in {"0", "false", "no", "off"}
        if light_stealth is not None:
            self.light_stealth: bool = bool(light_stealth)
        elif config is not None and hasattr(config, "light_stealth"):
            self.light_stealth = bool(config.light_stealth)
        else:
            val = os.environ.get("REFLEX_LIGHT_STEALTH", "").strip().lower()
            self.light_stealth = val in {"1", "true", "yes", "on"}

        if browser_language is not None:
            self.browser_language = str(browser_language).strip() or "tr-TR"
        elif config is not None and hasattr(config, "browser_language"):
            self.browser_language = str(config.browser_language).strip() or "tr-TR"
        else:
            self.browser_language = os.environ.get("REFLEX_BROWSER_LANG", "tr-TR").strip() or "tr-TR"

        self.browser_languages = format_browser_languages(self.browser_language)
        self.accept_language = format_accept_language(self.browser_languages)

        cdp_port = port or DEFAULT_CDP_PORT
        if port is None:
            bu_cdp = os.environ.get("BU_CDP_URL")
            if bu_cdp:
                try:
                    p = urlsplit(bu_cdp).port
                    if p:
                        cdp_port = p
                except (ValueError, AttributeError):
                    pass
        try:
            ensure_chromium_cdp(port=cdp_port, browser_language=self.browser_language)
        except TypeError:
            ensure_chromium_cdp(port=cdp_port)
        ensure_daemon()
        self.cdp_port: int = cdp_port
        self._cdp_lock = threading.Lock()
        self.last_jpeg_bytes: bytes | None = None
        self.last_action_rect: dict[str, Any] | None = None
        self.target: str | None = cdp("Target.createTarget", url="about:blank", background=False)["targetId"]
        self._owned_targets: set[str] = {self.target} if self.target else set()
        self.session: str = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self._frame_sessions: dict[str, str] = {}
        self._iframe_node_map: dict[int, dict[str, Any]] = {}
        try:
            cdp("Target.setAutoAttach", autoAttach=True, waitForDebuggerOnStart=False, flatten=True)
        except Exception:
            pass
        self.after_input: dict[str, Any] | None = None
        self._native_dialogs: list[str] = []
        try:
            self.call("Page.enable")
        except Exception:
            pass
        self.call("Page.bringToFront")
        self.call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )
        self.call("Emulation.setFocusEmulationEnabled", enabled=True)
        try:
            self.call("Network.enable")
            self.call("Network.setBlockedURLs", urls=_AD_BLOCK_URLS)
            self.call("Network.setExtraHTTPHeaders", headers={"Accept-Language": self.accept_language})
        except Exception:
            pass
        try:
            init_script = build_stealth_init_script(self.browser_languages)
            self.call("Page.addScriptToEvaluateOnNewDocument", source=init_script)
        except Exception:
            pass
        try:
            raw_ua = str(self.evaluate("navigator.userAgent") or "")
            if not raw_ua:
                try:
                    raw_ua = str(cdp("Browser.getVersion").get("userAgent", ""))
                except Exception:
                    pass
            clean_ua = raw_ua.replace("HeadlessChrome", "Chrome").replace("Headless", "")
            if clean_ua:
                self.call(
                    "Emulation.setUserAgentOverride",
                    userAgent=clean_ua,
                    acceptLanguage=self.accept_language,
                    platform="Linux x86_64",
                )
        except Exception:
            pass
        self.navigate(url)

    def _stealth_delay(
        self,
        min_s: float = STEALTH_MIN_DELAY_S,
        max_s: float = STEALTH_MAX_DELAY_S,
    ) -> float:
        """Sleep for a randomized duration if light_stealth is active, returning duration."""
        return _stealth_delay(self.light_stealth, min_s=min_s, max_s=max_s)

    def _wait_for_navigation_and_settle(self, prev_origin: Any = None, max_wait: float = 10.0) -> None:
        """Wait for a newly triggered navigation (or Enter submit) to commit, reach readyState complete, and settle."""
        start = time.monotonic()
        if prev_origin is not None:
            origin_deadline = start + 2.5
            while time.monotonic() < origin_deadline:
                try:
                    cur_origin = self.evaluate("performance.timeOrigin")
                    if cur_origin and cur_origin != prev_origin:
                        break
                except Exception:
                    break
                time.sleep(0.06)

        deadline = start + max_wait
        while time.monotonic() < deadline:
            try:
                if self.evaluate("document.readyState") == "complete":
                    break
            except Exception:
                pass
            time.sleep(0.06)

        settle_deadline = time.monotonic() + float(os.environ.get("REFLEX_SETTLE_WAIT", "2.2") or 2.2)
        last_count = -1
        stable_hits = 0
        while time.monotonic() < settle_deadline:
            try:
                state = self.evaluate(
                    """(() => ({
                        count: document.querySelectorAll('a[href],button,input,textarea,select,[contenteditable="true"],[role="button"],[role="textbox"],[role="searchbox"],[role="combobox"]').length,
                        textLen: (document.body?.innerText || '').trim().length,
                        busy: Boolean(document.querySelector('[aria-busy="true"]'))
                    }))()"""
                ) or {}
                count = int(state.get("count") or 0)
                text_len = int(state.get("textLen") or 0)
                busy = bool(state.get("busy"))
                if busy:
                    stable_hits = 0
                elif count > 0 and count == last_count and text_len > 80:
                    stable_hits += 1
                    needed_hits = 2 if count >= 6 else 5
                    if stable_hits >= needed_hits:
                        break
                else:
                    stable_hits = 0
                last_count = count
            except Exception:
                stable_hits = 0
            time.sleep(0.12)

    def navigate(self, url: str) -> None:
        """Navigate to a URL, wait for readyState complete, and let SPA hydration settle."""
        try:
            prev_origin = self.evaluate("performance.timeOrigin")
        except Exception:
            prev_origin = None
        try:
            self.call("Page.navigate", url=url, _response_timeout=NAVIGATE_RESPONSE_TIMEOUT_S)
        except TimeoutError as exc:
            try:
                self.call("Page.stopLoading")
            except Exception:
                pass
            raise NavigationTimeout(
                f"{url} did not respond within {NAVIGATE_RESPONSE_TIMEOUT_S:g}s (site may be down)"
            ) from exc
        self._wait_for_navigation_and_settle(prev_origin=prev_origin, max_wait=15.0)
        self.after_input = {"kind": "click"}

    def _dismiss_js_dialog(self) -> str | None:
        """Dismiss an active native JavaScript dialog (alert/confirm/prompt/beforeunload) if present."""
        dlg_type = ""
        dlg_msg = ""
        try:
            from browser_harness.helpers import _send as _bh_send

            pending = _bh_send({"meta": "pending_dialog"}).get("dialog")
            if isinstance(pending, dict):
                dlg_type = str(pending.get("type") or "").strip().lower()
                dlg_msg = str(pending.get("message") or "").strip()
        except Exception:
            pass

        accept = dlg_type in {"alert", "beforeunload"}
        try:
            cdp(
                "Page.handleJavaScriptDialog",
                session_id=self.session,
                _response_timeout=1.5,
                accept=accept,
            )
        except Exception:
            return None

        label = (dlg_type or "dialog").capitalize()
        summary = f"[{label}] {dlg_msg[:400]}".strip() if dlg_msg else f"[{label}] dismissed"
        dialogs = getattr(self, "_native_dialogs", None)
        if isinstance(dialogs, list):
            dialogs.append(summary)
            if len(dialogs) > 5:
                del dialogs[:-5]
        return summary

    def call(self, method: str, session_id: str | None = None, **params: Any) -> dict[str, Any]:
        sid = session_id or self.session
        with self._cdp_lock:
            try:
                return cdp(method, session_id=sid, **params)
            except StalePage:
                raise
            except TimeoutError:
                if method != "Page.handleJavaScriptDialog" and self._dismiss_js_dialog() is not None:
                    try:
                        return cdp(method, session_id=sid, **params)
                    except Exception as retry_exc:
                        if _is_navigation_cdp_error(retry_exc):
                            raise StalePage(str(retry_exc)) from retry_exc
                        raise
                raise
            except Exception as exc:
                if _is_navigation_cdp_error(exc):
                    raise StalePage(str(exc)) from exc
                raise

    def capture_jpeg_bytes(self, quality: int = 68) -> bytes | None:
        """Capture a live JPEG frame non-blockingly for the real-time MJPEG stream."""
        if not self.target:
            return self.last_jpeg_bytes
        if not self._cdp_lock.acquire(timeout=0.06):
            return self.last_jpeg_bytes
        try:
            b64 = cdp(
                "Page.captureScreenshot",
                session_id=self.session,
                format="jpeg",
                quality=quality,
                optimizeForSpeed=True,
            ).get("data")
            if b64:
                self.last_jpeg_bytes = base64.b64decode(b64)
        except Exception:
            pass
        finally:
            self._cdp_lock.release()
        return self.last_jpeg_bytes

    def click_xy(self, x: float, y: float) -> None:
        """Dispatch a compositor-level mouse move + click at `(x, y)` (works inside cross-origin Cloudflare iframes)."""
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
        time.sleep(0.03)
        for event in ("mousePressed", "mouseReleased"):
            self.call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
        time.sleep(0.25)

    def evaluate(self, expression: str, session_id: str | None = None) -> Any:
        try:
            response = self.call("Runtime.evaluate", session_id=session_id, expression=expression, returnByValue=True)
        except StalePage:
            raise
        except TimeoutError as exc:
            raise EvaluationTimeout(str(exc)) from exc
        except Exception as exc:
            if _is_navigation_cdp_error(exc):
                raise StalePage(str(exc)) from exc
            raise
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def _page_text_length(self) -> int:
        return int(self.evaluate(_PAGE_TEXT_LEN) or 0)

    def _wait_while_page_text_grows(self) -> None:
        """Poll until page text stops growing.

        Runs only after text grew inside the post-settle arm window. Two equal
        reads do not end the wait. Waiting stops after 600 ms with no growth
        and never lasts past 10 seconds.
        """
        try:
            last = self._page_text_length()
        except Exception:
            return
        last_growth = time.monotonic()
        slept: list[float] = []
        while True:
            now = time.monotonic()
            quiet_left = CLICK_TEXT_QUIET_S - (now - last_growth)
            if quiet_left <= 0:
                return
            # Budget is the sum of slices already issued. Adding 0.05 two
            # hundred times overshoots 10.0 in binary float, so the last
            # slice is whatever remainder keeps that sum on the cap.
            cap_left = CLICK_TEXT_CAP_S - sum(slept)
            if cap_left <= 1e-9:
                return
            delay = min(0.05, quiet_left, cap_left)
            if sum([*slept, delay]) > CLICK_TEXT_CAP_S:
                delay = cap_left
            if delay <= 0:
                return
            time.sleep(delay)
            slept.append(delay)
            if cap_left - delay <= 1e-9:
                return
            try:
                current = self._page_text_length()
            except Exception:
                return
            if current > last:
                last = current
                last_growth = time.monotonic()

    def _arm_text_growth_wait(self, before_len: int | None, started: float) -> None:
        """Arm the 600 ms text poll if text grows within 250 ms after settle.

        A quiet page does not hold that window open past the post-click budget,
        so the static wait stays under 200 ms.
        """
        try:
            current = self._page_text_length()
        except Exception:
            return
        if before_len is not None and current > before_len:
            self._wait_while_page_text_grows()
            return
        deadline = min(time.monotonic() + CLICK_TEXT_ARM_S, started + CLICK_SETTLE_BUDGET_S)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 1e-9:
                return
            time.sleep(min(0.05, remaining))
            try:
                nxt = self._page_text_length()
            except Exception:
                return
            if nxt > current:
                self._wait_while_page_text_grows()
                return
            current = nxt

    def observe(self, screenshot: bool = True) -> dict[str, Any]:
        """Observe the current page state atomically via snapshot.js."""
        if self.after_input is not None:
            action, self.after_input = self.after_input, None
            started = time.monotonic()
            before_len: int | None = None
            if action.get("kind") == "click":
                try:
                    before_len = self._page_text_length()
                except Exception:
                    before_len = None
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field = window.__jevFast?.nodes.get(action.node);
                      const isFill = action.kind === 'fill';
                      const isClick = action.kind === 'click';
                      const startHref = location.href;
                      const autocomplete = isFill && field?.getAttribute('role') === 'combobox';
                      const maxWait = autocomplete ? 260 : (isFill ? 200 : (isClick ? 2000 : 90));
                      const quietMs = isFill ? 60 : 100;

                      let stopped = false;
                      let lastActivity = performance.now();
                      let lastResourceCount = performance.getEntriesByType('resource').length;
                      const startTime = performance.now();

                      const obs = document.body ? new MutationObserver(() => {
                        lastActivity = performance.now();
                      }) : null;
                      if (obs) obs.observe(document.body, { childList: true, subtree: true, attributes: true });

                      const finish = () => {
                        if (stopped) return;
                        stopped = true;
                        try { obs?.disconnect(); } catch (_) {}
                        try { document.getAnimations().forEach(a => a.finish()); } catch (_) {}
                        resolve();
                      };

                      const timer = setTimeout(finish, maxWait);

                      const check = () => {
                        if (stopped) return;
                        const now = performance.now();

                        const resCount = performance.getEntriesByType('resource').length;
                        if (resCount !== lastResourceCount) {
                          lastResourceCount = resCount;
                          lastActivity = now;
                        }

                        if (document.querySelector('[aria-busy="true"]')) {
                          lastActivity = now;
                        }

                        if (autocomplete) {
                          const ids = (field?.getAttribute('aria-controls') || field?.getAttribute('aria-owns') || '').split(/\\s+/).filter(Boolean);
                          const roots = ids.length ? ids.map(id => document.getElementById(id)).filter(Boolean) : [document];
                          const hasOption = roots.some(r => r.querySelector('[role="option"]'));
                          if (hasOption && now - lastActivity >= 40) {
                            clearTimeout(timer);
                            finish();
                            return;
                          }
                        }

                        if (location.href !== startHref || (now - startTime >= 40 && now - lastActivity >= quietMs)) {
                          clearTimeout(timer);
                          finish();
                          return;
                        }

                        requestAnimationFrame(check);
                      };

                      requestAnimationFrame(check);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except Exception:
                pass
            if action.get("kind") == "click":
                self._arm_text_growth_wait(before_len, started)

        stale_wait = float(os.environ.get("REFLEX_STALE_WAIT", "0.15") or 0.15)
        for attempt in range(10):
            try:
                info = self.evaluate(READ_STATE)
                if info is None:
                    raise StalePage("Document is navigating")
                if self._try_pierce_cloudflare(info):
                    time.sleep(1.2)
                    info = self.evaluate(READ_STATE) or info
                if info.get("has_iframes"):
                    self._supplement_iframes(info)
                if getattr(self, "ax_supplement", True):
                    self._supplement_axtree(info)
                native_dlgs = getattr(self, "_native_dialogs", None)
                if isinstance(native_dlgs, list) and native_dlgs:
                    prefix = "\n".join(native_dlgs) + "\n"
                    info["text"] = (prefix + str(info.get("text") or ""))[:6000]
                    info["full_text"] = (prefix + str(info.get("full_text") or ""))[:30000]
                info = clean_surrogates(info)
                info["fingerprint"] = fingerprint(info)
                info["last_action_rect"] = self.last_action_rect
                if screenshot:
                    try:
                        b64 = self.call("Page.captureScreenshot", format="jpeg", quality=65, optimizeForSpeed=True).get("data")
                        if b64:
                            info["screenshot"] = b64
                            self.last_jpeg_bytes = base64.b64decode(b64)
                    except Exception:
                        if self.last_jpeg_bytes:
                            info["screenshot"] = base64.b64encode(self.last_jpeg_bytes).decode("ascii")
                elif self.last_jpeg_bytes is None:
                    try:
                        b64 = self.call("Page.captureScreenshot", format="jpeg", quality=65, optimizeForSpeed=True).get("data")
                        if b64:
                            self.last_jpeg_bytes = base64.b64decode(b64)
                    except Exception:
                        pass
                return info
            except EvaluationTimeout:
                raise
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(stale_wait)
        raise StalePage("Page did not settle")

    def _get_iframe_rect(self, action: dict[str, Any]) -> dict[str, Any]:
        """Retrieve current top-level bounding rect of the iframe holding this action."""
        dom_frame_index = action.get("dom_frame_index") if action.get("dom_frame_index") is not None else action.get("frame_index")
        fallback = action.get("frame_rect") or {"x": 0, "y": 0, "w": 0, "h": 0}
        try:
            if dom_frame_index is not None:
                r = self.evaluate(
                    f"(() => {{ const f = document.querySelectorAll('iframe')[{dom_frame_index}]; "
                    f"return f ? ({{x: f.getBoundingClientRect().x, y: f.getBoundingClientRect().y, "
                    f"w: f.getBoundingClientRect().width, h: f.getBoundingClientRect().height}}) : null; }})()"
                )
                if r:
                    return {"x": round(r["x"]), "y": round(r["y"]), "w": round(r["w"]), "h": round(r["h"])}
            else:
                f_x = fallback.get("x", 0)
                f_y = fallback.get("y", 0)
                r = self.evaluate(f"""(() => {{
                    const frames = document.querySelectorAll('iframe');
                    for (const f of frames) {{
                        const rect = f.getBoundingClientRect();
                        if (Math.abs(rect.x - {f_x}) < 15 && Math.abs(rect.y - {f_y}) < 15) {{
                            return {{x: rect.x, y: rect.y, w: rect.width, h: rect.height}};
                        }}
                    }}
                    return null;
                }})()""")
                if r:
                    return {"x": round(r["x"]), "y": round(r["y"]), "w": round(r["w"]), "h": round(r["h"])}
        except Exception:
            pass
        return fallback

    def _supplement_axtree(self, info: dict[str, Any]) -> None:
        """Supplement snapshot.js actions with interactive elements from CDP Accessibility tree."""
        try:
            snap = self.call("DOMSnapshot.captureSnapshot", computedStyles=["visibility"], includeDOMRects=True)
        except Exception as e:
            logger.debug("DOMSnapshot.captureSnapshot failed: %s", e)
            return

        documents = snap.get("documents", [])
        strings = snap.get("strings", [])
        if not documents:
            return

        vw = int(info.get("w") or 1120)
        vh = int(info.get("h") or 780)

        seen_backend_ids: set[int] = set()
        backend_to_bounds: dict[int, tuple[float, float, float, float]] = {}
        backend_to_vis: dict[int, str] = {}
        backend_to_attrs: dict[int, dict[str, str]] = {}

        for doc in documents:
            nodes = doc.get("nodes", {})
            layout = doc.get("layout", {})
            backend_ids = nodes.get("backendNodeId", [])
            attributes = nodes.get("attributes", [])
            node_indices = layout.get("nodeIndex", [])
            bounds = layout.get("bounds", [])
            styles = layout.get("styles", [])

            for n_idx, bid in enumerate(backend_ids):
                attrs = attributes[n_idx] if n_idx < len(attributes) else None
                if attrs:
                    attr_dict: dict[str, str] = {}
                    for i in range(0, len(attrs) - 1, 2):
                        k_idx = attrs[i]
                        v_idx = attrs[i + 1]
                        if k_idx < len(strings) and v_idx < len(strings):
                            k = strings[k_idx]
                            v = strings[v_idx]
                            attr_dict[k] = v
                            if k == "data-reflex-id":
                                seen_backend_ids.add(bid)
                    backend_to_attrs[bid] = attr_dict

            for l_idx, n_idx in enumerate(node_indices):
                if n_idx < len(backend_ids):
                    bid = backend_ids[n_idx]
                    box = bounds[l_idx] if l_idx < len(bounds) else None
                    if box and len(box) >= 4:
                        backend_to_bounds[bid] = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    style_pair = styles[l_idx] if l_idx < len(styles) else None
                    if style_pair and len(style_pair) > 0:
                        s_idx = style_pair[0]
                        if s_idx < len(strings):
                            backend_to_vis[bid] = strings[s_idx]

        try:
            ax = self.call("Accessibility.getFullAXTree")
            ax_nodes = ax.get("nodes", [])
        except Exception as e:
            logger.debug("Accessibility.getFullAXTree failed: %s", e)
            return

        candidates: list[dict[str, Any]] = []
        seen_candidate_bids: set[int] = set()

        for node in ax_nodes:
            if node.get("ignored"):
                continue
            role_obj = node.get("role")
            role_val = str(role_obj.get("value") if isinstance(role_obj, dict) else (role_obj or "")).lower()
            if not (role_val in INTERACTIVE_AX_ROLES or role_val.startswith("menuitem")):
                continue

            bid = node.get("backendDOMNodeId")
            if bid is None or bid in seen_backend_ids or bid in seen_candidate_bids:
                continue

            props: dict[str, Any] = {}
            for p in node.get("properties", []):
                pname = p.get("name")
                pval = p.get("value")
                if isinstance(pval, dict):
                    pval = pval.get("value")
                props[pname] = pval

            if props.get("disabled") is True or props.get("hidden") is True or props.get("hiddenRoot") is True:
                continue

            node_attrs = backend_to_attrs.get(bid, {})
            if "disabled" in node_attrs or node_attrs.get("aria-disabled") == "true" or node_attrs.get("aria-hidden") == "true":
                continue

            box = backend_to_bounds.get(bid)
            if not box:
                continue
            bx, by, bw, bh = box
            if bw <= 0 or bh <= 0:
                continue

            if backend_to_vis.get(bid) == "hidden":
                continue

            if bx + bw <= 0 or bx >= vw or by + bh <= 0 or by >= vh:
                continue

            name_obj = node.get("name")
            name_val = str(name_obj.get("value") if isinstance(name_obj, dict) else (name_obj or "")).strip()

            candidates.append({
                "backend_id": bid,
                "role": role_val,
                "name": name_val,
                "bounds": (bx, by, bw, bh),
            })
            seen_candidate_bids.add(bid)

        if not candidates:
            return

        element_actions = [a for a in info.get("actions", []) if a.get("kind") not in ("scroll", "wait")]
        tail_actions = [a for a in info.get("actions", []) if a.get("kind") in ("scroll", "wait")]

        max_id_num = 0
        for a in element_actions:
            aid = str(a.get("id", ""))
            if aid.startswith("e") and aid[1:].isdigit():
                max_id_num = max(max_id_num, int(aid[1:]))

        next_id_num = max(max_id_num + 1, len(element_actions) + 1)

        reg_script = """function() {
            if (!this.isConnected) return null;
            let curr = this;
            while (curr) {
                const root = curr.getRootNode ? curr.getRootNode() : null;
                if (root && root.mode === 'closed') return null;
                if (root && root.host && root.host !== curr) {
                    curr = root.host;
                } else {
                    break;
                }
            }
            const doc = this.ownerDocument || document;
            const win = doc.defaultView || window;
            let topWin = window;
            try { if (window.top && window.top.__jevFast) topWin = window.top; } catch (_) {}
            const cache = topWin.__jevFast ||= {ids: new WeakMap(), nodes: new Map(), next: 1};
            if (!cache.ids.has(this)) cache.ids.set(this, cache.next++);
            const id = cache.ids.get(this);
            cache.nodes.set(id, this);
            if (topWin !== win) {
                const localCache = win.__jevFast ||= {ids: new WeakMap(), nodes: new Map(), next: 1};
                localCache.ids.set(this, id);
                localCache.nodes.set(id, this);
            }
            return id;
        }"""

        supplemented_actions: list[dict[str, Any]] = []
        guards = info.setdefault("guards", {})

        for cand in candidates:
            bid = cand["backend_id"]
            role_val = cand["role"]
            name_val = cand["name"]
            bx, by, bw, bh = cand["bounds"]

            try:
                resolved = self.call("DOM.resolveNode", backendNodeId=bid)
                obj_id = resolved.get("object", {}).get("objectId")
                if not obj_id:
                    continue

                reg_res = self.call("Runtime.callFunctionOn", objectId=obj_id, functionDeclaration=reg_script, returnByValue=True)
                node_id = reg_res.get("result", {}).get("value")
                if node_id is None:
                    continue

                guard = self.evaluate(f"window.__jevFast.guard(window.__jevFast.nodes.get({node_id}))")
                if guard:
                    guards[str(node_id)] = guard
                    guards[node_id] = guard

                act_id = f"e{next_id_num}"
                next_id_num += 1

                kind = (
                    "fill"
                    if role_val in ("textbox", "searchbox", "spinbutton", "slider")
                    else ("select" if role_val in ("combobox", "listbox") else "click")
                )
                action = {
                    "id": act_id,
                    "node": node_id,
                    "role": role_val,
                    "label": name_val or role_val,
                    "accessible_name": name_val,
                    "kind": kind,
                    "value": "",
                    "rect": {
                        "x": round(bx),
                        "y": round(by),
                        "w": round(bw),
                        "h": round(bh),
                    },
                    "is_ax_supplement": True,
                }
                supplemented_actions.append(action)
                seen_backend_ids.add(bid)
            except Exception as e:
                logger.debug("Failed to register AX candidate %s: %s", bid, e)
                continue

        if supplemented_actions:
            all_actions = element_actions + supplemented_actions + tail_actions
            info["actions"] = all_actions

    def _supplement_iframes(self, info: dict[str, Any]) -> None:
        """Discover visible iframes (>=200x100, max 5) and merge their interactive actions."""
        try:
            candidates = self.evaluate(f"""(() => {{
                const minW = {IFRAME_MIN_WIDTH}, minH = {IFRAME_MIN_HEIGHT}, maxFrames = {MAX_IFRAMES};
                const frames = Array.from(document.querySelectorAll('iframe'));
                const res = [];
                for (let i = 0; i < frames.length; i++) {{
                    const f = frames[i];
                    try {{
                        const r = f.getBoundingClientRect();
                        if (r.width < minW || r.height < minH) continue;
                        if (f.checkVisibility && !f.checkVisibility({{checkOpacity: true, checkVisibilityCSS: true}})) continue;
                        const st = window.getComputedStyle(f);
                        if (st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') continue;
                        let sameOrigin = false;
                        try {{
                            if (f.contentDocument && f.contentWindow && f.contentDocument.body) {{
                                sameOrigin = true;
                            }}
                        }} catch (_) {{
                            sameOrigin = false;
                        }}
                        res.push({{
                            index: i,
                            src: f.src || f.getAttribute('src') || '',
                            sameOrigin: sameOrigin,
                            rect: {{
                                x: Math.round(r.x),
                                y: Math.round(r.y),
                                w: Math.round(r.width),
                                h: Math.round(r.height),
                                left: Math.round(r.left),
                                top: Math.round(r.top),
                                right: Math.round(r.right),
                                bottom: Math.round(r.bottom)
                            }}
                        }});
                        if (res.length >= maxFrames) break;
                    }} catch (_) {{}}
                }}
                return res;
            }})()""")
        except Exception:
            candidates = None

        if not candidates or not isinstance(candidates, list):
            return

        has_cross = any(not c.get("sameOrigin") for c in candidates)
        child_targets: list[dict[str, Any]] = []
        if has_cross:
            try:
                raw_targets = cdp("Target.getTargets").get("targetInfos", [])
                child_targets = [
                    t for t in raw_targets
                    if t.get("parentId") == self.target and t.get("type") == "iframe"
                ]
            except Exception:
                child_targets = []

        main_actions = [a for a in info.get("actions", []) if a.get("kind") != "wait"]
        wait_action = next((a for a in info.get("actions", []) if a.get("kind") == "wait"), None)

        max_node = 0
        max_id_num = 0
        for a in main_actions:
            n = a.get("node")
            if isinstance(n, int) and n > max_node:
                max_node = n
            aid = str(a.get("id", ""))
            if aid.startswith("e") and aid[1:].isdigit():
                max_id_num = max(max_id_num, int(aid[1:]))

        next_node = max(max_node + 1, len(main_actions) + 1)
        next_id_num = max(max_id_num + 1, len(main_actions) + 1)

        vw = int(info.get("w") or 1120)
        vh = int(info.get("h") or 780)
        guards = info.setdefault("guards", {})

        iframe_actions: list[dict[str, Any]] = []
        extra_texts: list[str] = []
        unmatched_targets = list(child_targets)

        for candidate in candidates:
            f_rect = candidate.get("rect") or {}
            fx = f_rect.get("x", 0)
            fy = f_rect.get("y", 0)
            f_left = f_rect.get("left", fx)
            f_top = f_rect.get("top", fy)
            f_right = f_rect.get("right", fx + f_rect.get("w", 0))
            f_bottom = f_rect.get("bottom", fy + f_rect.get("h", 0))

            clip_left = max(0, f_left)
            clip_top = max(0, f_top)
            clip_right = min(vw, f_right)
            clip_bottom = min(vh, f_bottom)
            if clip_right <= clip_left or clip_bottom <= clip_top:
                continue

            frame_state: dict[str, Any] | None = None
            sid: str | None = None
            frame_index: int | None = None
            dom_frame_index: int | None = candidate.get("index")

            if candidate.get("sameOrigin"):
                frame_index = candidate.get("index")
                try:
                    eval_expr = f"""(() => {{
                        const f = document.querySelectorAll('iframe')[{frame_index}];
                        if (!f || !f.contentWindow) return null;
                        return f.contentWindow.eval({json.dumps(READ_STATE)});
                    }})()"""
                    frame_state = self.evaluate(eval_expr)
                except Exception:
                    frame_state = None
            else:
                matched_target: dict[str, Any] | None = None
                c_src = (candidate.get("src") or "").strip()
                c_canon = canonical_url(c_src)
                for t in unmatched_targets:
                    t_url = (t.get("url") or "").strip()
                    t_canon = canonical_url(t_url)
                    if c_canon and t_canon and c_canon == t_canon:
                        matched_target = t
                        break
                    if c_src and t_url and (c_src in t_url or t_url in c_src):
                        matched_target = t
                        break
                if not matched_target and unmatched_targets:
                    matched_target = unmatched_targets[0]

                if matched_target:
                    unmatched_targets.remove(matched_target)
                    tid = matched_target["targetId"]
                    sid = self._frame_sessions.get(tid)
                    if not sid:
                        try:
                            att = cdp("Target.attachToTarget", targetId=tid, flatten=True)
                            sid = att.get("sessionId")
                            if sid:
                                self._frame_sessions[tid] = sid
                                self._owned_targets.add(tid)
                        except Exception:
                            sid = None
                    if sid:
                        try:
                            frame_state = self.evaluate(READ_STATE, session_id=sid)
                        except Exception:
                            self._frame_sessions.pop(tid, None)
                            frame_state = None

            if not frame_state or not isinstance(frame_state, dict):
                continue

            ftxt = str(frame_state.get("text", "")).strip()
            if ftxt and ftxt not in info.get("text", ""):
                extra_texts.append(ftxt)

            frame_guards = frame_state.get("guards") or {}
            frame_pk = frame_state.get("page_key")

            for child_act in frame_state.get("actions", []):
                if child_act.get("kind") == "wait":
                    continue
                c_rect = child_act.get("rect")
                if not c_rect or not isinstance(c_rect, dict):
                    continue

                cx = c_rect.get("x", 0)
                cy = c_rect.get("y", 0)
                cw = c_rect.get("w", 0)
                ch = c_rect.get("h", 0)

                trans_x = fx + cx
                trans_y = fy + cy
                trans_w = cw
                trans_h = ch

                el_left = trans_x
                el_top = trans_y
                el_right = trans_x + trans_w
                el_bottom = trans_y + trans_h

                vis_left = max(clip_left, el_left)
                vis_top = max(clip_top, el_top)
                vis_right = min(clip_right, el_right)
                vis_bottom = min(clip_bottom, el_bottom)
                vis_w = vis_right - vis_left
                vis_h = vis_bottom - vis_top

                if vis_w <= 0 or vis_h <= 0:
                    continue
                if vis_w < min(16, trans_w * 0.5) or vis_h < min(20, trans_h * 0.35):
                    continue

                orig_node = child_act.get("node")
                unified_node = next_node
                next_node += 1

                unified_id = f"e{next_id_num}"
                next_id_num += 1

                merged_act = dict(child_act)
                merged_act["id"] = unified_id
                merged_act["node"] = unified_node
                merged_act["orig_node"] = orig_node
                merged_act["is_iframe"] = True
                merged_act["frame_rect"] = f_rect
                merged_act["frame_session_id"] = sid
                merged_act["frame_index"] = frame_index
                merged_act["dom_frame_index"] = dom_frame_index
                merged_act["frame_page_key"] = frame_pk
                merged_act["rect"] = {
                    "x": round(trans_x),
                    "y": round(trans_y),
                    "w": round(trans_w),
                    "h": round(trans_h),
                }

                iframe_actions.append(merged_act)

                self._iframe_node_map[unified_node] = {
                    "session_id": sid,
                    "frame_index": frame_index,
                    "dom_frame_index": dom_frame_index,
                    "orig_node": orig_node,
                    "frame_rect": f_rect,
                    "frame_page_key": frame_pk,
                }

                if orig_node is not None:
                    orig_guard = frame_guards.get(str(orig_node)) or frame_guards.get(orig_node)
                    if orig_guard:
                        guards[str(unified_node)] = orig_guard
                        guards[unified_node] = orig_guard

        if extra_texts:
            combo = "\n".join(extra_texts)
            cur_txt = str(info.get("text", "")).strip()
            cur_full = str(info.get("full_text", "")).strip()
            info["text"] = f"{cur_txt}\n{combo}"[:6000] if cur_txt else combo[:6000]
            info["full_text"] = f"{cur_full}\n{combo}"[:30000] if cur_full else combo[:30000]

        all_actions = main_actions + iframe_actions
        if wait_action:
            all_actions.append(wait_action)
        info["actions"] = all_actions

    def _try_pierce_cloudflare(self, info: dict[str, Any]) -> bool:
        """Pierce closed shadow DOM via CDP `DOM.getFlattenedDocument(pierce=True)` to click Cloudflare Turnstile."""
        title = str(info.get("title", "")).lower()
        text = str(info.get("text", "")).lower()
        if not any(
            k in title or k in text
            for k in ("just a moment", "bir dakika lütfen", "verify you are human", "insan olduğunuzu doğrulayın", "attention required")
        ):
            return False
        try:
            nodes = self.call("DOM.getFlattenedDocument", depth=-1, pierce=True).get("nodes", [])
            for node in nodes:
                node_name = str(node.get("nodeName", "")).upper()
                attrs = node.get("attributes") or []
                attr_dict = {attrs[i]: attrs[i + 1] for i in range(0, len(attrs) - 1, 2)}
                src = str(attr_dict.get("src", "")).lower()
                title_attr = str(attr_dict.get("title", "")).lower()
                if node_name == "IFRAME" and ("challenges.cloudflare.com" in src or "cloudflare" in title_attr or "turnstile" in src):
                    box = self.call("DOM.getBoxModel", nodeId=node["nodeId"]).get("model", {}).get("border")
                    if box and len(box) >= 6:
                        x0, y0 = float(box[0]), float(box[1])
                        h = float(box[5]) - y0
                        self.click_xy(x0 + 28.0, y0 + max(16.0, h / 2.0))
                        return True
        except Exception:
            pass
        return False

    def fresh(self, page: dict[str, Any], action: dict[str, Any] | None = None) -> bool:
        """Check whether the page or target node is still fresh."""
        try:
            if action is not None and action.get("kind") in {"click", "select", "fill"}:
                node = action.get("node")
                if type(node) is not int:
                    return False
                is_iframe = bool(action.get("is_iframe"))
                lookup_node = action.get("orig_node") if is_iframe else node
                session_id = action.get("frame_session_id")
                frame_index = action.get("frame_index")

                if is_iframe and frame_index is not None:
                    current = self.evaluate(
                        f"(() => {{ const f = document.querySelectorAll('iframe')[{frame_index}]; "
                        "const c = f?.contentWindow?.__jevFast; "
                        f"return c ? [c.pageKey(), c.guard(c.nodes.get({lookup_node}))] : null; }})()"
                    )
                elif is_iframe:
                    current = self.evaluate(
                        f"(() => {{ const c=window.__jevFast; return c ? [c.pageKey(),c.guard(c.nodes.get({lookup_node}))] : null; }})()",
                        session_id=session_id,
                    )
                else:
                    current = self.evaluate(
                        "(() => { const c=window.__jevFast; "
                        f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
                    )
                if not current or current[1] is None:
                    return False
                guards = page.get("guards") or {}
                expected_guard = guards.get(str(node)) or guards.get(node)
                if not expected_guard:
                    return False
                expected_pk = (action.get("frame_page_key") if is_iframe else page.get("page_key")) or []
                current_pk = current[0] or []
                same_doc = len(expected_pk) >= 2 and len(current_pk) >= 2 and current_pk[:2] == expected_pk[:2]
                if not same_doc:
                    return False
                cur_g = current[1]
                if cur_g[:2] != expected_guard[:2] or cur_g[3:13] != expected_guard[3:13]:
                    return False
                cur_lbl = re.sub(r"\b\d+\s*(?:s|sn|sec|ms)\b", "", str(cur_g[2] or "")).strip()
                exp_lbl = re.sub(r"\b\d+\s*(?:s|sn|sec|ms)\b", "", str(expected_guard[2] or "")).strip()
                return cur_lbl == exp_lbl
            return self.evaluate(MARKER) == page.get("marker")
        except StalePage:
            return False

    def accessible_name_matches(self, action: dict[str, Any], page: dict[str, Any] | None = None) -> bool:
        """True when the live name still equals the raw name recorded at decision time.

        The decorated label is not parsed. An empty or unreadable name does not
        block. A different raw name returns False.
        """
        if action.get("kind") != "click":
            return True
        node = action.get("node")
        if type(node) is not int:
            return True
        recorded = _recorded_accessible_name(action, page)
        if not recorded:
            return True
        is_iframe = bool(action.get("is_iframe"))
        lookup_node = action.get("orig_node") if is_iframe else node
        session_id = action.get("frame_session_id")
        frame_index = action.get("frame_index")
        try:
            if is_iframe and frame_index is not None:
                current = self.evaluate(
                    f"(() => {{ const f = document.querySelectorAll('iframe')[{frame_index}]; "
                    f"return f?.contentWindow ? f.contentWindow.eval({json.dumps(_ACCESSIBLE_NAME)})({json.dumps(lookup_node)}) : null; }})()"
                )
            elif is_iframe and session_id:
                current = self.evaluate(
                    _ACCESSIBLE_NAME + "(" + json.dumps(lookup_node) + ")",
                    session_id=session_id,
                )
            else:
                current = self.evaluate(_ACCESSIBLE_NAME + "(" + json.dumps(lookup_node) + ")")
        except Exception:
            return True
        if not isinstance(current, str) or not current.strip():
            return True
        return _norm_accessible_label(recorded) == _norm_accessible_label(current)

    def _refuse_click_if_name_changed(self, action: dict[str, Any], page: dict[str, Any]) -> None:
        """Do not click when the live accessible name no longer matches the recorded raw name."""
        if not self.accessible_name_matches(action, page):
            raise StalePage("Accessible name changed since this decision. Observe again.")

    def act(self, action: dict[str, Any], page: dict[str, Any], text: str | None = None, value: str | None = None) -> dict[str, Any]:
        """Execute a low-level action on the observed page."""
        if value is not None:
            action = dict(action, value=value)
        kind = action["kind"]
        native_dlgs = getattr(self, "_native_dialogs", None)
        if kind != "wait" and isinstance(native_dlgs, list):
            native_dlgs.clear()
        if kind == "goto_url":
            target_url = str(action.get("url") or text or "").strip()
            if not target_url:
                raise ValueError("goto_url action requires a non-empty url")
            if not target_url.startswith(("http://", "https://")):
                target_url = "https://" + target_url
            self.navigate(target_url)
            self.after_input = None
            return {"executed": f"goto_url:{target_url}"}

        if kind == "reload":
            try:
                prev_origin = self.evaluate("performance.timeOrigin")
            except Exception:
                prev_origin = None
            self.call("Page.reload")
            self._wait_for_navigation_and_settle(prev_origin=prev_origin, max_wait=10.0)
            self.after_input = None
            return {"executed": "reload"}

        if kind in {"press_escape", "press_enter", "go_back"}:
            return self._recovery_act(kind)

        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")

        if kind == "click":
            self._refuse_click_if_name_changed(action, page)

        if kind == "wait":
            wait_s = float(os.environ.get("REFLEX_WAIT_SECONDS", "1.0") or 1.0)
            time.sleep(wait_s)
            self.after_input = None
            return {"executed": action["id"]}

        if kind == "scroll":
            total_delta = int(action.get("delta", 420))
            step_delta = int(total_delta / 3)
            self.last_action_rect = {
                "x": 536,
                "y": 366,
                "w": 48,
                "h": 48,
                "cx": 560,
                "cy": 390,
                "kind": "scroll",
                "label": str(action.get("label", "Scroll")),
            }
            try:
                before_y = int(self.evaluate("window.scrollY || 0") or 0)
            except Exception:
                before_y = 0
            for i in range(3):
                d = total_delta - step_delta * 2 if i == 2 else step_delta
                self.call(
                    "Input.dispatchMouseEvent",
                    type="mouseWheel",
                    x=550,
                    y=450,
                    deltaX=0,
                    deltaY=d,
                )
                time.sleep(0.04)
            try:
                after_y = int(self.evaluate("window.scrollY || 0") or 0)
                if after_y == before_y:
                    self.evaluate(f"window.scrollBy(0, {total_delta})")
            except Exception:
                pass
            if self.light_stealth:
                self._stealth_delay()
            self.after_input = action
            return {"executed": action["id"], "rect": self.last_action_rect}

        if type(action.get("node")) is not int:
            raise ValueError("Invalid observed node")

        is_iframe = bool(action.get("is_iframe"))
        lookup_node = action.get("orig_node") if is_iframe else action["node"]
        session_id = action.get("frame_session_id")
        frame_index = action.get("frame_index")
        action_param = dict(action, node=lookup_node)

        if is_iframe:
            try:
                dom_frame_index = action.get("dom_frame_index") if action.get("dom_frame_index") is not None else action.get("frame_index")
                if dom_frame_index is not None:
                    self.evaluate(f"""(() => {{
                        const f = document.querySelectorAll('iframe')[{dom_frame_index}];
                        if (!f) return;
                        const r = f.getBoundingClientRect();
                        if (r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) {{
                            f.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                        }}
                    }})()""")
                else:
                    self.evaluate(f"""(() => {{
                        const frames = document.querySelectorAll('iframe');
                        for (const f of frames) {{
                            const r = f.getBoundingClientRect();
                            if (Math.abs(r.x - {action.get('frame_rect', {}).get('x', 0)}) < 10) {{
                                if (r.top < 0 || r.bottom > innerHeight || r.left < 0 || r.right > innerWidth) {{
                                    f.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                                }}
                                break;
                            }}
                        }}
                    }})()""")
            except Exception:
                pass

        target_fn = """(action => {
          const node=window.__jevFast?.nodes.get(action.node);
          if (!node?.isConnected || node.matches(':disabled') || node.closest('[aria-disabled="true"],[inert]')) return null;
          const shown=el=>el.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
          let e=node;
          const isChoiceInput=node.tagName==='INPUT' && ['radio','checkbox'].includes(node.type);
          // Custom-styled radio/checkbox: the native input is hidden, its visible <label> takes the click.
          if (!shown(node) && isChoiceInput) {
            e=[...(node.labels||[]), node.closest('label')].find(l=>l && shown(l)) || node;
          }
          const overlayChoice=!shown(e) && e===node && isChoiceInput;
          if (!shown(e) && !overlayChoice) return null;
          if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
          if (action.kind==='click') {
            const link=e.tagName==='A'?e:e.closest('a[href]');
            if (link) link.removeAttribute('target');
            const form=e.form || e.closest('form');
            if (form) form.removeAttribute('target');
            if (e.removeAttribute) {
              e.removeAttribute('target');
              e.removeAttribute('formtarget');
            }
            document.querySelectorAll('base[target]').forEach(b=>b.removeAttribute('target'));
            document.querySelectorAll('[formtarget]').forEach(b=>b.removeAttribute('formtarget'));
          }
          let r=e.getBoundingClientRect();
          if (overlayChoice && (r.width<=0 || r.height<=0)) return null;
          if (r.top<0 || r.bottom>innerHeight || r.left<0 || r.right>innerWidth) {
            e.scrollIntoView({behavior:'instant',block:'center',inline:'nearest'});
            r=e.getBoundingClientRect();
          }
          const visLeft=Math.max(4,r.left), visRight=Math.min(innerWidth-4,r.right);
          const visTop=Math.max(4,r.top), visBottom=Math.min(innerHeight-4,r.bottom);
          if (visRight<=visLeft || visBottom<=visTop) return null;
          const isIframe=e.tagName==='IFRAME';
          const x=isIframe ? r.x+Math.min(28, r.width/2) : (visLeft+visRight)/2;
          const y=(visTop+visBottom)/2;
          if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
          const rx=Math.round(r.x), ry=Math.round(r.y), rw=Math.round(r.width), rh=Math.round(r.height);
          const rootNode=e.getRootNode?.();
          const hit=(rootNode && typeof rootNode.elementFromPoint==='function')
            ? rootNode.elementFromPoint(x,y)
            : document.elementFromPoint(x,y);
          if (overlayChoice && hit!==node) return null;
          const interactiveSel='button,a,[role="button"],[role="link"],[role="option"],[role="menuitem"],[role="tab"]';
          const hitOk=hit && (
            e.contains(hit) ||
            hit===e ||
            (hit.tagName==='LABEL' && hit.control===e) ||
            (e.labels && [...e.labels].some(l=>l.contains(hit))) ||
            (e.closest(interactiveSel) && e.closest(interactiveSel)===hit.closest(interactiveSel))
          );
          if (!hitOk) {
            const modalSel='dialog[open],[role="dialog"],[aria-modal="true"]';
            const blockingModal=hit?.closest(modalSel);
            if (blockingModal && !e.closest(modalSel)) return null;
            if (action.kind==='click') {
              try {
                if (Array.isArray(window.__reflexDialogs)) window.__reflexDialogs.length = 0;
                if (document.activeElement && document.activeElement!==document.body) document.activeElement.blur();
                const prevOrigin = performance.timeOrigin;
                e.click();
                return {x:Math.round(x),y:Math.round(y),rx,ry,rw,rh,jsClicked:true,prevOrigin};
              } catch (_) {
                return null;
              }
            }
            return null;
          }
          if (Array.isArray(window.__reflexDialogs)) window.__reflexDialogs.length = 0;
          if (action.kind==='fill') {
            try { e.focus(); } catch (_) {}
          }
          if (action.kind==='select') {
            if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                !o.disabled && !o.closest('optgroup[disabled]'))) return null;
            e.value=action.value;
            e.dispatchEvent(new Event('input',{bubbles:true}));
            e.dispatchEvent(new Event('change',{bubbles:true}));
          }
          return {x:Math.round(x),y:Math.round(y),rx,ry,rw,rh,jsClicked:false,prevOrigin:performance.timeOrigin};
        })"""

        if is_iframe and frame_index is not None:
            target = self.evaluate(
                f"document.querySelectorAll('iframe')[{frame_index}].contentWindow.eval({json.dumps(target_fn + '(' + json.dumps(action_param) + ')')})"
            )
        elif is_iframe:
            target = self.evaluate(
                target_fn + "(" + json.dumps(action_param) + ")",
                session_id=session_id,
            )
        else:
            target = self.evaluate(
                """(action => {
                  const node=window.__jevFast?.nodes.get(action.node);
                  if (!node?.isConnected || node.matches(':disabled') || node.closest('[aria-disabled="true"],[inert]')) return null;
                  const shown=el=>el.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                  let e=node;
                  const isChoiceInput=node.tagName==='INPUT' && ['radio','checkbox'].includes(node.type);
                  // Custom-styled radio/checkbox: the native input is hidden, its visible <label> takes the click.
                  if (!shown(node) && isChoiceInput) {
                    e=[...(node.labels||[]), node.closest('label')].find(l=>l && shown(l)) || node;
                  }
                  const overlayChoice=!shown(e) && e===node && isChoiceInput;
                  if (!shown(e) && !overlayChoice) return null;
                  if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
                  if (action.kind==='click') {
                    const link=e.tagName==='A'?e:e.closest('a[href]');
                    if (link) link.removeAttribute('target');
                    const form=e.form || e.closest('form');
                    if (form) form.removeAttribute('target');
                    if (e.removeAttribute) {
                      e.removeAttribute('target');
                      e.removeAttribute('formtarget');
                    }
                    document.querySelectorAll('base[target]').forEach(b=>b.removeAttribute('target'));
                    document.querySelectorAll('[formtarget]').forEach(b=>b.removeAttribute('formtarget'));
                  }
                  let r=e.getBoundingClientRect();
                  if (overlayChoice && (r.width<=0 || r.height<=0)) return null;
                  if (r.top<0 || r.bottom>innerHeight || r.left<0 || r.right>innerWidth) {
                    e.scrollIntoView({behavior:'instant',block:'center',inline:'nearest'});
                    r=e.getBoundingClientRect();
                  }
                  const visLeft=Math.max(4,r.left), visRight=Math.min(innerWidth-4,r.right);
                  const visTop=Math.max(4,r.top), visBottom=Math.min(innerHeight-4,r.bottom);
                  if (visRight<=visLeft || visBottom<=visTop) return null;
                  const isIframe=e.tagName==='IFRAME';
                  const x=isIframe ? r.x+Math.min(28, r.width/2) : (visLeft+visRight)/2;
                  const y=(visTop+visBottom)/2;
                  if (!r.width || !r.height || x<0 || y<0 || x>=innerWidth || y>=innerHeight) return null;
                  const rx=Math.round(r.x), ry=Math.round(r.y), rw=Math.round(r.width), rh=Math.round(r.height);
                  const rootNode=e.getRootNode?.();
                  const hit=(rootNode && typeof rootNode.elementFromPoint==='function')
                    ? rootNode.elementFromPoint(x,y)
                    : document.elementFromPoint(x,y);
                  if (overlayChoice && hit!==node) return null;
                  const interactiveSel='button,a,[role="button"],[role="link"],[role="option"],[role="menuitem"],[role="tab"]';
                  const hitOk=hit && (
                    e.contains(hit) ||
                    hit===e ||
                    (hit.tagName==='LABEL' && hit.control===e) ||
                    (e.labels && [...e.labels].some(l=>l.contains(hit))) ||
                    (e.closest(interactiveSel) && e.closest(interactiveSel)===hit.closest(interactiveSel))
                  );
                  if (!hitOk) {
                    const modalSel='dialog[open],[role="dialog"],[aria-modal="true"]';
                    const blockingModal=hit?.closest(modalSel);
                    if (blockingModal && !e.closest(modalSel)) return null;
                    if (action.kind==='click') {
                      try {
                        if (Array.isArray(window.__reflexDialogs)) window.__reflexDialogs.length = 0;
                        if (document.activeElement && document.activeElement!==document.body) document.activeElement.blur();
                        const prevOrigin = performance.timeOrigin;
                        e.click();
                        return {x:Math.round(x),y:Math.round(y),rx,ry,rw,rh,jsClicked:true,prevOrigin};
                      } catch (_) {
                        return null;
                      }
                    }
                    return null;
                  }
                  if (Array.isArray(window.__reflexDialogs)) window.__reflexDialogs.length = 0;
                  if (action.kind==='fill') {
                    try { e.focus(); } catch (_) {}
                  }
                  if (action.kind==='select') {
                    if (e.tagName!=='SELECT' || ![...e.options].some(o=>o.value===action.value &&
                        !o.disabled && !o.closest('optgroup[disabled]'))) return null;
                    e.value=action.value;
                    e.dispatchEvent(new Event('input',{bubbles:true}));
                    e.dispatchEvent(new Event('change',{bubbles:true}));
                  }
                  return {x:Math.round(x),y:Math.round(y),rx,ry,rw,rh,jsClicked:false,prevOrigin:performance.timeOrigin};
                })("""
                + json.dumps(action)
                + ")"
            )

        if target is None:
            if kind == "select":
                raise RuntimeError("Dropdown execution was not confirmed; inspect before retrying.")
            raise TargetCoveredError(
                f"Target '{action.get('label', '')}' is covered by an overlay/modal or offscreen. Dismiss the overlay first."
            )

        if is_iframe:
            cur_f_rect = self._get_iframe_rect(action)
            fx = cur_f_rect.get("x", 0)
            fy = cur_f_rect.get("y", 0)
            target["x"] = round(fx + target["x"])
            target["y"] = round(fy + target["y"])
            target["rx"] = round(fx + target.get("rx", target["x"] - 20))
            target["ry"] = round(fy + target.get("ry", target["y"] - 12))

        self.last_action_rect = {
            "x": target.get("rx", target["x"] - 20),
            "y": target.get("ry", target["y"] - 12),
            "w": target.get("rw", 40),
            "h": target.get("rh", 24),
            "cx": target["x"],
            "cy": target["y"],
            "kind": kind,
            "label": str(action.get("label", "")),
        }

        if kind != "select" and not target.get("jsClicked"):
            x, y = target["x"], target["y"]
            if self.light_stealth:
                self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
                self._stealth_delay()
                self.call("Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", clickCount=1)
                self._stealth_delay(min_s=0.03, max_s=0.08)
                self.call("Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", clickCount=1)
                self._stealth_delay()
            else:
                if is_iframe:
                    self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
                    time.sleep(0.03)
                for event in ("mousePressed", "mouseReleased"):
                    self.call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
            if kind == "fill":
                if text is None:
                    raise ValueError("fill action requires non-null text")
                clear_js = """(nodeId => {
                  const e = window.__jevFast?.nodes.get(nodeId);
                  if (!e) return;
                  try { e.focus(); } catch (_) {}
                  if ('value' in e) {
                    const proto = e.tagName === 'TEXTAREA'
                      ? window.HTMLTextAreaElement?.prototype
                      : window.HTMLInputElement?.prototype;
                    const nativeSetter = proto ? Object.getOwnPropertyDescriptor(proto, 'value')?.set : null;
                    if (nativeSetter) nativeSetter.call(e, '');
                    else e.value = '';
                  } else if (e.isContentEditable) {
                    e.innerText = '';
                  }
                  e.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'deleteContentBackward'}));
                  e.dispatchEvent(new Event('change', {bubbles: true}));
                })"""
                try:
                    if is_iframe and frame_index is not None:
                        self.evaluate(
                            f"document.querySelectorAll('iframe')[{frame_index}].contentWindow.eval({json.dumps(clear_js + '(' + json.dumps(lookup_node) + ')')})"
                        )
                    elif is_iframe:
                        self.evaluate(clear_js + "(" + json.dumps(lookup_node) + ")", session_id=session_id)
                    else:
                        self.evaluate(
                            """(nodeId => {
                              const e = window.__jevFast?.nodes.get(nodeId);
                              if (!e) return;
                              try { e.focus(); } catch (_) {}
                              if ('value' in e) {
                                const proto = e.tagName === 'TEXTAREA'
                                  ? window.HTMLTextAreaElement?.prototype
                                  : window.HTMLInputElement?.prototype;
                                const nativeSetter = proto ? Object.getOwnPropertyDescriptor(proto, 'value')?.set : null;
                                if (nativeSetter) nativeSetter.call(e, '');
                                else e.value = '';
                              } else if (e.isContentEditable) {
                                e.innerText = '';
                              }
                              e.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'deleteContentBackward'}));
                              e.dispatchEvent(new Event('change', {bubbles: true}));
                            })(""" + json.dumps(action.get("node")) + ")"
                        )
                except Exception:
                    pass
                self.call("Input.insertText", text=text)
                if self.light_stealth:
                    self._stealth_delay()
                try:
                    setter_js = """((nodeId, textVal) => {
                      const e = window.__jevFast?.nodes.get(nodeId);
                      if (!e) return;
                      const specialTypes = ['date', 'time', 'datetime-local', 'month', 'week', 'range'];
                      if (e.tagName === 'INPUT' && specialTypes.includes(e.type)) {
                        const proto = window.HTMLInputElement?.prototype;
                        const nativeSetter = proto ? Object.getOwnPropertyDescriptor(proto, 'value')?.set : null;
                        if (nativeSetter) nativeSetter.call(e, textVal);
                        else e.value = textVal;
                        e.dispatchEvent(new Event('input', {bubbles: true}));
                        e.dispatchEvent(new Event('change', {bubbles: true}));
                      } else if (e.getAttribute('role') === 'slider' && !('value' in e)) {
                        e.setAttribute('aria-valuenow', textVal);
                        e.dispatchEvent(new Event('input', {bubbles: true}));
                        e.dispatchEvent(new Event('change', {bubbles: true}));
                      }
                    })"""
                    if is_iframe and frame_index is not None:
                        self.evaluate(
                            f"document.querySelectorAll('iframe')[{frame_index}].contentWindow.eval({json.dumps(setter_js + '(' + json.dumps(lookup_node) + ', ' + json.dumps(text) + ')')})"
                        )
                    elif is_iframe:
                        self.evaluate(
                            setter_js + "(" + json.dumps(lookup_node) + ", " + json.dumps(text) + ")",
                            session_id=session_id,
                        )
                    else:
                        self.evaluate(
                            setter_js + "(" + json.dumps(action.get("node")) + ", " + json.dumps(text) + ")"
                        )
                except Exception:
                    pass
                search_js = """(nodeId => {
                  const e = window.__jevFast?.nodes.get(nodeId);
                  if (!e) return false;
                  try {
                    e.dispatchEvent(new Event('input', {bubbles: true}));
                    e.dispatchEvent(new Event('change', {bubbles: true}));
                  } catch (_) {}
                  if (e.getAttribute('role') === 'combobox' || e.getAttribute('aria-autocomplete') || e.getAttribute('aria-haspopup') === 'listbox') {
                    return false;
                  }
                  return e.type === 'search' || e.getAttribute('role') === 'searchbox' || Boolean(e.closest('form[role="search"]'));
                })"""
                try:
                    if is_iframe and frame_index is not None:
                        is_search = bool(self.evaluate(
                            f"document.querySelectorAll('iframe')[{frame_index}].contentWindow.eval({json.dumps(search_js + '(' + json.dumps(lookup_node) + ')')})"
                        ))
                    elif is_iframe:
                        is_search = bool(self.evaluate(search_js + "(" + json.dumps(lookup_node) + ")", session_id=session_id))
                    else:
                        is_search = bool(
                            self.evaluate(
                                """(nodeId => {
                                  const e = window.__jevFast?.nodes.get(nodeId);
                                  if (!e) return false;
                                  try {
                                    e.dispatchEvent(new Event('input', {bubbles: true}));
                                    e.dispatchEvent(new Event('change', {bubbles: true}));
                                  } catch (_) {}
                                  if (e.getAttribute('role') === 'combobox' || e.getAttribute('aria-autocomplete') || e.getAttribute('aria-haspopup') === 'listbox') {
                                    return false;
                                  }
                                  return e.type === 'search' || e.getAttribute('role') === 'searchbox' || Boolean(e.closest('form[role="search"]'));
                                })(""" + json.dumps(action.get("node")) + ")"
                            )
                        )
                    if is_search:
                        try:
                            prev_origin = self.evaluate("performance.timeOrigin", session_id=session_id)
                        except Exception:
                            prev_origin = None
                        time.sleep(0.05)
                        self.call("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", windowsVirtualKeyCode=13)
                        self.call("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)
                        self._wait_for_navigation_and_settle(prev_origin=prev_origin, max_wait=8.0)
                        self.after_input = None
                        return {"executed": action["id"], "auto_submitted": True, "rect": self.last_action_rect}
                except Exception:
                    pass

        if kind == "click":
            link_href = str(action.get("href") or "").strip()
            if link_href.startswith(("http://", "https://")):
                cur_base = str(page.get("url") or "").split("#")[0]
                target_base = link_href.split("#")[0]
                if target_base and target_base != cur_base:
                    self._wait_for_navigation_and_settle(prev_origin=target.get("prevOrigin"), max_wait=8.0)
                    self.after_input = None
                    return {"executed": action["id"], "rect": self.last_action_rect}

        if kind == "select" and self.light_stealth:
            self._stealth_delay()
        self.after_input = action
        return {"executed": action["id"], "rect": self.last_action_rect}

    def scroll_to_node(self, node: int) -> bool:
        """Scroll a specific DOM node into view deterministically using instant center alignment."""
        if not isinstance(node, int):
            return False
        iframe_info = getattr(self, "_iframe_node_map", {}).get(node)
        if iframe_info:
            session_id = iframe_info.get("session_id")
            dom_frame_index = iframe_info.get("dom_frame_index") if iframe_info.get("dom_frame_index") is not None else iframe_info.get("frame_index")
            frame_index = iframe_info.get("frame_index")
            orig_node = iframe_info.get("orig_node", node)
            try:
                if dom_frame_index is not None:
                    self.evaluate(f"document.querySelectorAll('iframe')[{dom_frame_index}]?.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}})")
                else:
                    f_rect = iframe_info.get("frame_rect") or {}
                    self.evaluate(f"""(() => {{
                        const frames = document.querySelectorAll('iframe');
                        for (const f of frames) {{
                            const r = f.getBoundingClientRect();
                            if (Math.abs(r.x - {f_rect.get('x', 0)}) < 10) {{
                                f.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                                break;
                            }}
                        }}
                    }})()""")

                if frame_index is not None:
                    res = self.evaluate(
                        f"""(() => {{
                          const f = document.querySelectorAll('iframe')[{frame_index}];
                          const el = f?.contentWindow?.__jevFast?.nodes?.get({orig_node});
                          if (!el || !el.isConnected) return false;
                          el.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                          return true;
                        }})()"""
                    )
                    return bool(res)
                else:
                    res = self.evaluate(
                        f"""(() => {{
                          const el = window.__jevFast?.nodes?.get({orig_node});
                          if (!el || !el.isConnected) return false;
                          el.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                          return true;
                        }})()""",
                        session_id=session_id,
                    )
                    return bool(res)
            except Exception:
                return False

        try:
            res = self.evaluate(
                f"""(() => {{
                  const el = window.__jevFast?.nodes?.get({node});
                  if (!el || !el.isConnected) return false;
                  el.scrollIntoView({{behavior: 'instant', block: 'center', inline: 'nearest'}});
                  return true;
                }})()"""
            )
            return bool(res)
        except Exception:
            return False

    def _recovery_act(self, kind: str) -> dict[str, Any]:
        """Execute a supervisor-level recovery action (Escape, Enter, History Back)."""
        if kind == "press_escape":
            self.call("Input.dispatchKeyEvent", type="keyDown", key="Escape", code="Escape", windowsVirtualKeyCode=27)
            self.call("Input.dispatchKeyEvent", type="keyUp", key="Escape", code="Escape", windowsVirtualKeyCode=27)
            time.sleep(0.15)
            return {"executed": "press_escape"}
        if kind == "press_enter":
            try:
                prev_origin = self.evaluate("performance.timeOrigin")
            except Exception:
                prev_origin = None
            self.call("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", windowsVirtualKeyCode=13)
            self.call("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)
            self._wait_for_navigation_and_settle(prev_origin=prev_origin, max_wait=6.0)
            return {"executed": "press_enter"}
        if kind == "go_back":
            self.evaluate("history.back()")
            time.sleep(0.35)
            return {"executed": "go_back"}
        raise ValueError(f"Unsupported recovery action: {kind}")

    def close(self) -> None:
        targets_to_close = set(self._owned_targets) if hasattr(self, "_owned_targets") else set()
        if self.target:
            targets_to_close.add(self.target)
        for tid in targets_to_close:
            try:
                cdp("Target.closeTarget", targetId=tid)
            except Exception:
                pass
        if hasattr(self, "_owned_targets"):
            self._owned_targets.clear()
        if hasattr(self, "_frame_sessions"):
            self._frame_sessions.clear()
        self.target = None

    def __enter__(self) -> Browser:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
