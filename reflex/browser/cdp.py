"""CDP Browser controller with atomic DOM snapshots and recovery primitives."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
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

_STEALTH_INIT_SCRIPT = """(() => {
  try {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    window.chrome = window.chrome || { runtime: {}, app: {}, csi: () => ({}), loadTimes: () => ({}) };
    Object.defineProperty(navigator, 'languages', { get: () => ['tr-TR', 'tr', 'en-US', 'en'] });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    if (!window.outerWidth) Object.defineProperty(window, 'outerWidth', { get: () => window.innerWidth || 1280 });
    if (!window.outerHeight) Object.defineProperty(window, 'outerHeight', { get: () => (window.innerHeight || 800) + 85 });
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
      if (p === 37445) return 'Intel Inc.';
      if (p === 37446) return 'Intel Iris OpenGL Engine';
      return getParam.apply(this, arguments);
    };
    window.open = function(url) {
      if (url && typeof url === 'string' && url !== 'about:blank' && !url.startsWith('javascript:')) {
        window.location.assign(url);
      }
      return window;
    };
  } catch (_) {}
})();"""


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def ensure_chromium_cdp(port: int | None = None) -> None:
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
    profile_dir = Path("/tmp/reflex_chromium_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path("/tmp/reflex_chromium.log").open("a", encoding="utf-8")
    subprocess.Popen(
        [
            binary,
            "--headless=new",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-blink-features=AutomationControlled",
            "--disable-gpu",
            "--disk-cache-size=1048576",
            "--media-cache-size=1048576",
            "--window-size=1280,900",
            "--lang=tr-TR",
        ],
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
# Read the live accessible name. The token keeps this evaluate distinct from the click script.
_ACCESSIBLE_NAME = """(nodeId => {
  const reflexAccessibleName = window.__jevFast;
  const node = reflexAccessibleName?.nodes?.get(nodeId);
  if (!node || !reflexAccessibleName.guard) return null;
  const guard = reflexAccessibleName.guard(node);
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


class TargetCoveredError(RuntimeError):
    """Raised when the target element is obscured by a modal/overlay at hit-test time."""


_NAVIGATION_CDP_ERRORS = (
    "Inspected target navigated or closed",
    "Cannot find context with specified id",
    "Execution context was destroyed",
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


class Browser:
    """Single-tab CDP browser session with fresh-state verification and recovery actions."""

    def __init__(self, url: str, *, width: int = 1120, height: int = 780):
        cdp_port = DEFAULT_CDP_PORT
        bu_cdp = os.environ.get("BU_CDP_URL")
        if bu_cdp:
            try:
                p = urlsplit(bu_cdp).port
                if p:
                    cdp_port = p
            except (ValueError, AttributeError):
                pass
        ensure_chromium_cdp(port=cdp_port)
        ensure_daemon()
        self._cdp_lock = threading.Lock()
        self.last_jpeg_bytes: bytes | None = None
        self.last_action_rect: dict[str, Any] | None = None
        self.target: str | None = cdp("Target.createTarget", url="about:blank", background=False)["targetId"]
        self._owned_targets: set[str] = {self.target} if self.target else set()
        self.session: str = cdp("Target.attachToTarget", targetId=self.target, flatten=True)["sessionId"]
        self.after_input: dict[str, Any] | None = None
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
        except Exception:
            pass
        try:
            raw_ua = str(self.evaluate("navigator.userAgent") or "")
            clean_ua = raw_ua.replace("HeadlessChrome/", "Chrome/")
            if clean_ua:
                self.call(
                    "Emulation.setUserAgentOverride",
                    userAgent=clean_ua,
                    acceptLanguage="tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                    platform="Linux x86_64",
                )
            self.call("Page.addScriptToEvaluateOnNewDocument", source=_STEALTH_INIT_SCRIPT)
        except Exception:
            pass
        self.navigate(url)

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
        self.call("Page.navigate", url=url)
        self._wait_for_navigation_and_settle(prev_origin=prev_origin, max_wait=15.0)
        self.after_input = {"kind": "click"}

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        with self._cdp_lock:
            try:
                return cdp(method, session_id=self.session, **params)
            except StalePage:
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

    def evaluate(self, expression: str) -> Any:
        try:
            response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        except StalePage:
            raise
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
            except StalePage:
                if attempt == 9:
                    raise
                time.sleep(stale_wait)
        raise StalePage("Page did not settle")

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
        if action is not None and action.get("kind") in {"click", "select", "fill"}:
            node = action.get("node")
            if type(node) is not int:
                return False
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
            expected_pk = page.get("page_key") or []
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
        try:
            current = self.evaluate(_ACCESSIBLE_NAME + "(" + json.dumps(node) + ")")
        except Exception:
            return True
        if not isinstance(current, str) or not current.strip():
            return True
        return _norm_accessible_label(recorded) == _norm_accessible_label(current)

    def _refuse_click_if_name_changed(self, action: dict[str, Any], page: dict[str, Any]) -> None:
        """Do not click when the live accessible name no longer matches the recorded raw name."""
        if not self.accessible_name_matches(action, page):
            raise StalePage("Accessible name changed since this decision. Observe again.")

    def act(self, action: dict[str, Any], page: dict[str, Any], text: str | None = None) -> dict[str, Any]:
        """Execute a low-level action on the observed page."""
        kind = action["kind"]
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
            self.after_input = action
            return {"executed": action["id"], "rect": self.last_action_rect}

        if type(action.get("node")) is not int:
            raise ValueError("Invalid observed node")

        target = self.evaluate(
            """(action => {
              const node=window.__jevFast?.nodes.get(action.node);
              if (!node?.isConnected || node.matches(':disabled') || node.closest('[aria-disabled="true"],[inert]')) return null;
              const shown=el=>el.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
              let e=node;
              // Custom-styled radio/checkbox: the native input is hidden, its visible <label> takes the click.
              if (!shown(node) && node.tagName==='INPUT' && ['radio','checkbox'].includes(node.type)) {
                e=[...(node.labels||[]), node.closest('label')].find(l=>l && shown(l)) || node;
              }
              if (!shown(e)) return null;
              if (action.kind==='fill' && (e.readOnly || e.getAttribute('aria-readonly')==='true')) return null;
              if (action.kind==='click') {
                const link=e.tagName==='A'?e:e.closest('a[href]');
                if (link) link.removeAttribute('target');
                if (e.removeAttribute) e.removeAttribute('target');
                document.querySelectorAll('base[target]').forEach(b=>b.removeAttribute('target'));
              }
              let r=e.getBoundingClientRect();
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
              const hit=document.elementFromPoint(x,y);
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
            for event in ("mousePressed", "mouseReleased"):
                self.call("Input.dispatchMouseEvent", type=event, x=x, y=y, button="left", clickCount=1)
            if kind == "fill":
                if text is None:
                    raise ValueError("fill action requires non-null text")
                try:
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
                try:
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
                            prev_origin = self.evaluate("performance.timeOrigin")
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

        self.after_input = action
        return {"executed": action["id"], "rect": self.last_action_rect}

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
        self.target = None

    def __enter__(self) -> Browser:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
