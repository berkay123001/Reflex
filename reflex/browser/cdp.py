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


def ensure_chromium_cdp(port: int = DEFAULT_CDP_PORT) -> None:
    """Ensure a headless Chromium instance is listening on `port` and `BU_CDP_URL` is set."""
    expected_url = f"http://127.0.0.1:{port}"
    existing_url = os.environ.get("BU_CDP_URL")
    if existing_url and existing_url != expected_url:
        msg = f"BU_CDP_URL is already set to {existing_url!r}, which differs from expected {expected_url!r}"
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
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


class StalePage(ValueError):
    """Raised when the page DOM mutated after observation before execution."""


class TargetCoveredError(RuntimeError):
    """Raised when the target element is obscured by a modal/overlay at hit-test time."""


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
        ensure_chromium_cdp()
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
                        isGoogleSearch: location.hostname.includes('google.') && location.pathname.startsWith('/search'),
                        hasSearchResults: Boolean(document.querySelector('#search a[href], #rso a[href], #botstuff'))
                    }))()"""
                ) or {}
                count = int(state.get("count") or 0)
                text_len = int(state.get("textLen") or 0)
                is_google_search = bool(state.get("isGoogleSearch"))
                has_results = bool(state.get("hasSearchResults"))
                if is_google_search and not has_results and text_len < 300:
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
            return cdp(method, session_id=self.session, **params)

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
        response = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if response.get("exceptionDetails"):
            raise StalePage("Document changed during evaluation")
        return response.get("result", {}).get("value")

    def observe(self, screenshot: bool = True) -> dict[str, Any]:
        """Observe the current page state atomically via snapshot.js."""
        if self.after_input is not None:
            action, self.after_input = self.after_input, None
            try:
                self.call(
                    "Runtime.evaluate",
                    expression="""(action => new Promise(resolve => {
                      const field=window.__jevFast?.nodes.get(action.node);
                      const isFill=action.kind==='fill';
                      const isClick=action.kind==='click';
                      const startHref=location.href;
                      const autocomplete=isFill && field?.getAttribute('role')==='combobox';
                      let frames=0, idleFrames=0, stopped=false, mutated=false;
                      const obs=isClick && document.body ? new MutationObserver(()=>{ mutated=true; idleFrames=0; }) : null;
                      if (obs) obs.observe(document.body,{childList:true,subtree:true,attributes:true});
                      const finish=()=>{
                        if (stopped) return;
                        stopped=true;
                        try { obs?.disconnect(); } catch (_) {}
                        try { document.getAnimations().forEach(a => a.finish()); } catch (_) {}
                        resolve();
                      };
                      setTimeout(finish, autocomplete ? 260 : (isFill ? 200 : (isClick ? 2400 : 90)));
                      const ready=()=>{
                        if (stopped) return;
                        frames++;
                        let rouletteSpinning=false;
                        if (isClick && frames<140) {
                          const posCounts=new Map();
                          for (const img of document.querySelectorAll('img.opacity-0,[aria-busy="true"]')) {
                            const r=img.getBoundingClientRect();
                            if (r.width>24 && r.height>24 && r.bottom>0 && r.top<innerHeight) {
                              if (img.getAttribute('aria-busy')==='true') { rouletteSpinning=true; break; }
                              const posKey=`${Math.round(r.left/12)},${Math.round(r.top/12)}`;
                              const c=(posCounts.get(posKey)||0)+1;
                              posCounts.set(posKey,c);
                              if (c>=3) { rouletteSpinning=true; break; }
                            }
                          }
                        }
                        if (rouletteSpinning) {
                          idleFrames=0;
                          requestAnimationFrame(ready);
                          return;
                        }
                        idleFrames++;
                        const ids=(field?.getAttribute('aria-controls')||field?.getAttribute('aria-owns')||'')
                          .split(/\\s+/).filter(Boolean);
                        const roots=ids.length ? ids.map(id=>document.getElementById(id)).filter(Boolean) : [document];
                        const options=roots.flatMap(root=>[...root.querySelectorAll('[role="option"]')]);
                        const hasVisibleOption=options.some(e=>{
                          const r=e.getBoundingClientRect();
                          return r.width && r.height && r.bottom>0 && r.top<innerHeight &&
                            e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
                        });
                        const clickSettled=!isClick || (location.href!==startHref && frames>=5) || (mutated && (idleFrames>=2 || frames>=24) && frames>=5) || (!mutated && frames>=8);
                        const fillSettled=!isFill || (autocomplete ? (frames>=3 && hasVisibleOption) : (frames>=4 && idleFrames>=2));
                        if (frames>=3 && clickSettled && fillSettled) finish();
                        else requestAnimationFrame(ready);
                      };
                      requestAnimationFrame(ready);
                    }))(""" + json.dumps(action) + ")",
                    awaitPromise=True,
                    returnByValue=True,
                )
            except Exception:
                pass

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

        if kind in {"press_escape", "press_enter", "go_back"}:
            return self._recovery_act(kind)

        if not self.fresh(page, action):
            raise StalePage("Page changed since this decision. Observe again.")

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
              const e=window.__jevFast?.nodes.get(action.node);
              if (!e?.isConnected || e.matches(':disabled') || e.closest('[aria-disabled="true"],[inert]') ||
                  !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
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
                    submit_kind = str(
                        self.evaluate(
                            """(nodeId => {
                              const e = window.__jevFast?.nodes.get(nodeId);
                              if (!e) return '';
                              try {
                                e.dispatchEvent(new Event('input', {bubbles: true}));
                                e.dispatchEvent(new Event('change', {bubbles: true}));
                              } catch (_) {}
                              const aria = (e.getAttribute('aria-label') || e.placeholder || e.title || '').toLowerCase();
                              if (e.type === 'search' || e.getAttribute('role') === 'searchbox' ||
                                  e.name === 'q' || /(^|\\b)(search|ara)(\\b|$)/i.test(aria) ||
                                  Boolean(e.closest('form[role="search"], form[action*="search"]'))) {
                                return 'search';
                              }
                              if (/(istem girin|mesaj yaz|send a message|ask anything|ask gemini|message chatgpt|bir soru sorun|sohbet)/i.test(aria)) {
                                return 'chat';
                              }
                              return '';
                            })(""" + json.dumps(action.get("node")) + ")"
                        ) or ""
                    )
                    if submit_kind == "search":
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
                    if submit_kind == "chat":
                        time.sleep(0.08)
                        self.call("Input.dispatchKeyEvent", type="keyDown", key="Enter", code="Enter", windowsVirtualKeyCode=13)
                        self.call("Input.dispatchKeyEvent", type="keyUp", key="Enter", code="Enter", windowsVirtualKeyCode=13)
                        time.sleep(0.25)
                        # If the chat input still has text and a Send button is enabled, click the Send button
                        self.evaluate(
                            """(() => {
                              const btn = document.querySelector('button[aria-label*="Mesaj gönder" i]:not([disabled]), button[aria-label*="Send message" i]:not([disabled]), button[aria-label*="Gönder" i]:not([disabled])');
                              if (btn) { try { btn.click(); } catch (_) {} }
                            })()"""
                        )
                        # Wait up to 4.5s for the AI chat response to stream and settle
                        for _ in range(18):
                            time.sleep(0.25)
                            busy = self.evaluate(
                                """Boolean(document.querySelector('[aria-busy="true"], button[aria-label*="durdur" i], button[aria-label*="Stop" i]'))"""
                            )
                            if not busy and _ >= 5:
                                break
                        self.after_input = None
                        return {"executed": action["id"], "auto_submitted": True, "rect": self.last_action_rect}
                except Exception:
                    pass

        if kind == "click":
            link_href = str(action.get("href") or "").strip()
            if link_href.startswith(("http://", "https://")) and "#google_vignette" not in link_href:
                try:
                    time.sleep(0.08)
                    cur_hash = str(self.evaluate("location.hash || ''") or "")
                    if "google_vignette" in cur_hash:
                        self.navigate(link_href)
                        self.after_input = None
                        return {"executed": action["id"], "rect": self.last_action_rect}
                except Exception:
                    pass
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
