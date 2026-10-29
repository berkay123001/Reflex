"""Human demonstration recorder for System 1 training data.

Opens a visible Chromium window, injects `kaydedici.js` into every document and records each
human action (click, typing, dropdown choice, wheel scroll) together with the Reflex snapshot
taken immediately before it. Finishing a task successfully adds a final DONE step.

Usage:
    .venv/bin/python scripts/kayit/kaydedici.py            # UI: http://127.0.0.1:8790
Recordings: artifacts/demolar/<timestamp>-<slug>.jsonl (one session per file).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import websockets

from reflex.browser.cdp import _AD_BLOCK_URLS  # same ad blocklist as the agent's runtime browser

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts" / "demolar"
PROFILE_DIR = Path.home() / ".cache" / "reflex_kayit_profili"
VIEWPORT = {"width": 1120, "height": 780}  # same as reflex.browser.cdp.Browser
FORBIDDEN_CDP_PORT = 9333
MIN_SCROLL_TAIL_PX = 150
DEFAULT_START_URL = "https://www.google.com"
CHROMIUM_LOG = Path("/tmp/reflex_kayit_chromium.log")
HISTORY_OPS = {"GO_BACK", "GO_FORWARD", "RELOAD"}


def build_injected_script() -> str:
    snapshot = (ROOT / "reflex" / "browser" / "snapshot.js").read_text(encoding="utf-8")
    return (HERE / "kaydedici.js").read_text(encoding="utf-8").replace("__REFLEX_SNAPSHOT__", snapshot)


def port_open(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_chromium(port: int) -> subprocess.Popen | None:
    if port_open(port):
        return None
    binary = shutil.which("chromium") or shutil.which("chromium-browser") or shutil.which("google-chrome")
    if not binary:
        raise RuntimeError("Chromium bulunamadı.")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            binary,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={PROFILE_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
            "--lang=tr-TR",
            "--window-size=1180,980",
            "about:blank",
        ],
        stdout=CHROMIUM_LOG.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not port_open(port):
        time.sleep(0.2)
    if not port_open(port):
        raise RuntimeError(f"Chromium {port} portunda açılmadı.")
    return proc


def first_page_ws(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as resp:
        targets = json.load(resp)
    pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
    if not pages:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.load(resp)["webSocketDebuggerUrl"]
    return pages[0]["webSocketDebuggerUrl"]


class Cdp:
    """Minimal CDP client over one page websocket."""

    def __init__(self, ws: Any, on_event: Any) -> None:
        self.ws = ws
        self.on_event = on_event
        self.next_id = 0
        self.waiting: dict[int, asyncio.Future] = {}

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self.waiting[self.next_id] = fut
        await self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        msg = await asyncio.wait_for(fut, timeout=30)
        if "error" in msg:
            raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

    async def read_loop(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)
            if "id" in msg:
                fut = self.waiting.pop(msg["id"], None)
                if fut and not fut.done():
                    fut.set_result(msg)
            else:
                self.on_event(msg.get("method", ""), msg.get("params", {}))


class BrowserLink:
    """Keeps a CDP connection to the recording tab; relaunches Chromium if the window was closed."""

    def __init__(self, port: int, on_event: Any) -> None:
        self.port = port
        self.on_event = on_event
        self.cdp: Cdp | None = None
        self.ws: Any = None
        self.script = build_injected_script()

    @property
    def alive(self) -> bool:
        return self.cdp is not None

    async def ensure(self) -> Cdp:
        if self.cdp is not None:
            return self.cdp
        await asyncio.to_thread(launch_chromium, self.port)
        ws_url = await asyncio.to_thread(first_page_ws, self.port)
        self.ws = await websockets.connect(ws_url, max_size=None)
        cdp = Cdp(self.ws, self.on_event)
        asyncio.create_task(self._read(cdp))
        await cdp.call("Runtime.enable")
        await cdp.call("Page.enable")
        await cdp.call("Emulation.setDeviceMetricsOverride", deviceScaleFactor=1, mobile=False, **VIEWPORT)
        await cdp.call("Network.enable")
        await cdp.call("Network.setBlockedURLs", urls=_AD_BLOCK_URLS)
        await cdp.call("Runtime.addBinding", name="__reflexKayit")
        await cdp.call("Page.addScriptToEvaluateOnNewDocument", source=self.script)
        await cdp.call("Runtime.evaluate", expression=self.script)
        self.cdp = cdp
        return cdp

    async def _read(self, cdp: Cdp) -> None:
        reason = "kapandı"
        try:
            await cdp.read_loop()
        except Exception as exc:  # noqa: BLE001 - window closed or browser crashed
            reason = type(exc).__name__
        finally:
            if self.cdp is cdp:
                self.cdp = None
            print(f"Kayıt tarayıcısıyla bağlantı koptu ({reason}); Başlat'ta yeniden açılacak.", flush=True)


class Recorder:
    """Turns raw page events into System-1-shaped steps (state before action + operation)."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.link: BrowserLink | None = None
        self.summaries: dict[str, tuple[float, dict[str, Any]]] = {}
        self.active = False
        self.meta: dict[str, Any] = {}
        self.steps: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.burst_dir: str | None = None
        self.last_saved: str | None = None
        self.last_state: dict[str, Any] | None = None
        self.visited: list[str] = []
        self.main_frame_id: str | None = None

    # ---- page events (event-loop thread) ----
    def on_cdp_event(self, method: str, params: dict[str, Any]) -> None:
        if method == "Runtime.bindingCalled" and params.get("name") == "__reflexKayit":
            try:
                self.on_page_event(json.loads(params.get("payload") or "{}"))
            except (ValueError, TypeError):
                pass
        elif method == "Page.frameNavigated" and not (params.get("frame") or {}).get("parentId"):
            self.flush_pending()
            self.burst_dir = None
            self.main_frame_id = (params.get("frame") or {}).get("id")
            self.note_visit(str((params.get("frame") or {}).get("url") or ""))
        elif method == "Page.navigatedWithinDocument" and params.get("frameId") == self.main_frame_id:
            self.note_visit(str(params.get("url") or ""))

    def note_visit(self, url: str) -> None:
        if url and (not self.visited or self.visited[-1] != url):
            self.visited.append(url)

    def on_page_event(self, data: dict[str, Any]) -> None:
        if not self.active:
            return
        kind = data.get("type")
        node = data.get("node")
        pending = self.pending
        if kind == "history":
            self.on_history(str(data.get("kind")), str(data.get("url") or ""))  # uses the PREVIOUS page state
        if isinstance(data.get("state"), dict) and not data["state"].get("snapshot_error"):
            self.last_state = data["state"]
        if kind == "history":
            return
        if kind == "pointer":
            self.flush_pending()
            self.burst_dir = None
            if node is None:
                if data.get("unmapped"):  # not an S1 label: the snapshot cannot see this control
                    self.add_step("UNMAPPED_CLICK", data.get("state"), unmapped=data["unmapped"])
                return
            kinds = data.get("kinds") or []
            base = {"node": node, "label": data.get("label"), "state": data.get("state"), "kinds": kinds}
            if "fill" in kinds:
                initial = data.get("value") or ""
                self.pending = {**base, "op": "TYPE_TEXT", "initial": initial, "value": initial, "enter": False}
            elif "select" in kinds:
                self.pending = {**base, "op": "SELECT", "value": None, "option": None}
            else:
                self.add_step("CLICK", data.get("state"), node=node, label=data.get("label"))
        elif kind == "focus":
            if pending and pending["op"] == "TYPE_TEXT" and pending["node"] == node:
                return
            self.flush_pending()
            if node is not None:
                initial = data.get("value") or ""
                self.pending = {"op": "TYPE_TEXT", "node": node, "label": None, "state": data.get("state"),
                                "kinds": ["fill"], "initial": initial, "value": initial, "enter": False}
        elif kind in {"input", "enter"}:
            if pending and pending["op"] == "TYPE_TEXT" and node in (None, pending["node"]):
                pending["value"] = data.get("value") or ""
                if kind == "enter":
                    pending["enter"] = True
                    self.flush_pending()
        elif kind == "select":
            if pending and pending["op"] == "SELECT" and node in (None, pending["node"]):
                pending["value"] = data.get("value")
                pending["option"] = data.get("option")
                self.flush_pending()
        elif kind == "scroll":
            self.flush_pending()
            direction = data.get("dir")
            op = "SCROLL_DOWN" if direction == "down" else "SCROLL_UP"
            burst_start = self.burst_dir != direction
            self.burst_dir = direction
            self.add_step(op, data.get("state"), scroll_y=data.get("y"), burst_start=burst_start)
        elif kind == "scroll_end":
            self.burst_dir = None
            last = self.steps[-1] if self.steps else None
            if not last or not last["op"].startswith("SCROLL"):
                return
            if last.get("burst_start"):
                if not data.get("moved"):  # wheel on a page/area that could not scroll
                    self.steps.pop()
            elif abs(float(data.get("y") or 0) - float(last.get("scroll_y") or 0)) < MIN_SCROLL_TAIL_PX:
                self.steps.pop()

    def on_history(self, nav: str, url: str) -> None:
        """Toolbar back/forward/reload. Not an S1 operation (GO_BACK is S2-only), but recorded so the
        trajectory and later history stay consistent. The pre-state is the last observed state."""
        self.flush_pending()
        self.burst_dir = None
        last = self.steps[-1] if self.steps else None
        if (last and last["op"] in HISTORY_OPS and last.get("target_url") == url
                and time.time() - last["time"] < 1.0):
            return  # popstate + document load can both report one history move
        prev_url = (self.last_state or {}).get("url")
        if nav == "reload":
            op = "RELOAD"
        else:  # provisional; refined from the browser's own navigation index below
            earlier = self.visited[:-1] if self.visited and self.visited[-1] == url else self.visited
            op = "GO_BACK" if url in earlier else "GO_FORWARD"
        self.add_step(op, self.last_state or {"url": None}, target_url=url, pre_state_approx=True)
        if nav != "reload" and prev_url and self.link and self.link.cdp and self.steps:
            asyncio.get_running_loop().create_task(self.classify_history(self.steps[-1], prev_url))

    async def classify_history(self, step: dict[str, Any], prev_url: str) -> None:
        """Back vs forward from the navigation index: where the previous page sits relative to the current one."""
        try:
            hist = await self.link.cdp.call("Page.getNavigationHistory")
        except Exception:  # noqa: BLE001 - keep the provisional label
            return
        cur = int(hist.get("currentIndex", -1))
        matches = [i for i, e in enumerate(hist.get("entries") or []) if e.get("url") == prev_url]
        if cur < 0 or not matches:
            return
        prev_idx = min(matches, key=lambda i: abs(i - cur))
        step["op"] = "GO_BACK" if prev_idx > cur else "GO_FORWARD" if prev_idx < cur else "RELOAD"

    def flush_pending(self) -> None:
        p, self.pending = self.pending, None
        if not p or not self.active:
            return
        if p["op"] == "TYPE_TEXT":
            value = str(p.get("value") or "")
            if value.strip() and value != p.get("initial"):
                self.add_step("TYPE_TEXT", p["state"], node=p["node"], label=p.get("label"), text=value,
                              submitted_with_enter=bool(p.get("enter")))
            elif "click" in (p.get("kinds") or []):
                self.add_step("CLICK", p["state"], node=p["node"], label=p.get("label"))
        elif p["op"] == "SELECT" and p.get("value") is not None:
            self.add_step("SELECT", p["state"], node=p["node"], label=p.get("label"),
                          value=p["value"], option=p.get("option"))

    def add_step(self, op: str, state: Any, **fields: Any) -> None:
        if not isinstance(state, dict) or state.get("snapshot_error"):
            return
        if not state and op not in HISTORY_OPS | {"UNMAPPED_CLICK"}:
            return
        self.steps.append({
            "type": "step",
            "step": len(self.steps) + 1,
            "time": round(time.time(), 3),
            "op": op,
            "url": state.get("url"),
            **{k: v for k, v in fields.items() if v is not None},
            "state": state,
        })

    # ---- UI commands (scheduled onto the event loop) ----
    async def start(self, goal: str, url: str) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("Kayıt zaten açık.")
        if not goal.strip():
            raise ValueError("Görev boş olamaz.")
        start_url = url.strip() or DEFAULT_START_URL
        if not re.match(r"^https?://", start_url):
            start_url = "https://" + start_url
        cdp = await self.link.ensure()
        self.steps, self.pending, self.burst_dir = [], None, None
        self.last_state, self.visited = None, []
        self.meta = {"type": "session", "goal": goal.strip(), "start_url": start_url,
                     "started": round(time.time(), 3), "viewport": VIEWPORT}
        await cdp.call("Page.bringToFront")
        await cdp.call("Page.navigate", url=start_url)
        await asyncio.sleep(1.5)
        self.active = True
        return await self.status()

    async def finish(self, success: bool, note: str = "") -> dict[str, Any]:
        if not self.active:
            raise RuntimeError("Açık kayıt yok.")
        self.flush_pending()
        final_state = None
        if self.link and self.link.cdp:
            with contextlib.suppress(Exception):
                res = await self.link.cdp.call("Runtime.evaluate", expression="window.__reflexKayitDurum?.()",
                                               returnByValue=True)
                final_state = (res.get("result") or {}).get("value")
        if success:
            if final_state is None:
                note = (note + " " if note else "") + "tarayıcı kapalıydı, DONE adımı eklenemedi"
            self.add_step("DONE", final_state)
        self.active = False
        end = {"type": "session_end", "success": success, "note": note or None, "steps": len(self.steps),
               "duration_s": round(time.time() - self.meta["started"], 1)}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ascii_goal = self.meta["goal"].lower().translate(str.maketrans("ıiğüşöç", "iigusoc"))
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_goal)[:40].strip("-") or "gorev"
        path = self.out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in [self.meta, *self.steps, end]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.last_saved = str(path)
        return await self.status()

    async def cancel(self) -> dict[str, Any]:
        self.active, self.pending, self.steps = False, None, []
        return await self.status()

    async def undo(self) -> dict[str, Any]:
        if self.pending:
            self.pending = None
        elif self.steps:
            self.steps.pop()
        return await self.status()

    def session_summary(self, path: Path) -> dict[str, Any]:
        mtime = path.stat().st_mtime
        cached = self.summaries.get(path.name)
        if cached and cached[0] == mtime:
            return cached[1]
        with path.open("rb") as fh:
            meta = json.loads(fh.readline() or b"{}")
            fh.seek(max(0, path.stat().st_size - 4096))
            tail = fh.read().decode("utf-8", errors="ignore").strip().splitlines()
        end: dict[str, Any] = {}
        with contextlib.suppress(ValueError, IndexError):
            end = json.loads(tail[-1])
        summary = {"file": path.name, "goal": meta.get("goal"), "steps": end.get("steps", 0),
                   "success": end.get("success"), "started": meta.get("started"),
                   "duration_s": end.get("duration_s")}
        self.summaries[path.name] = (mtime, summary)
        return summary

    async def delete(self, name: str) -> dict[str, Any]:
        files = {p.name: p for p in self.out_dir.glob("*.jsonl")}
        if name not in files:
            raise ValueError("Kayıt bulunamadı.")
        files[name].unlink()
        self.summaries.pop(name, None)
        return await self.status()

    async def status(self) -> dict[str, Any]:
        sessions = []
        if self.out_dir.is_dir():
            sessions = [self.session_summary(p) for p in self.out_dir.glob("*.jsonl")]
        sessions.sort(key=lambda x: x.get("started") or 0, reverse=True)
        recent = [
            {"step": s["step"], "op": s["op"],
             "label": s.get("label") or (s.get("unmapped") or {}).get("text"), "text": s.get("text"),
             "option": s.get("option")}
            for s in self.steps[-8:]
        ]
        return {"active": self.active, "goal": self.meta.get("goal"), "steps": len(self.steps),
                "pending": (self.pending or {}).get("op"), "recent": recent,
                "browser": bool(self.link and self.link.alive),
                "saved_sessions": len(sessions), "saved_steps": sum(int(x.get("steps") or 0) for x in sessions),
                "successful_sessions": sum(1 for x in sessions if x.get("success")),
                "sessions": sessions[:50], "last_saved": self.last_saved}


def make_handler(rec: Recorder, loop: asyncio.AbstractEventLoop) -> type[BaseHTTPRequestHandler]:
    page = (HERE / "arayuz.html").read_bytes()

    def run(coro: Any) -> dict[str, Any]:
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=40)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: Any) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path == "/api/durum":
                return self._json(200, run(rec.status()))
            return self._json(404, {"error": "bulunamadı"})

        def do_POST(self) -> None:
            # Custom header forces a CORS preflight, so pages open in the recorded browser cannot drive the UI.
            if self.headers.get("X-Kayit") != "1":
                return self._json(403, {"error": "yasak"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/basla":
                    return self._json(200, run(rec.start(str(body.get("goal", "")), str(body.get("url", "")))))
                if self.path == "/api/bitir":
                    return self._json(200, run(rec.finish(bool(body.get("success")), str(body.get("note", "")))))
                if self.path == "/api/iptal":
                    return self._json(200, run(rec.cancel()))
                if self.path == "/api/geri_al":
                    return self._json(200, run(rec.undo()))
                if self.path == "/api/sil":
                    return self._json(200, run(rec.delete(str(body.get("file", "")))))
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                return self._json(400, {"error": str(exc)})
            return self._json(404, {"error": "bulunamadı"})

    return Handler


async def main_async(cdp_port: int, ui_port: int) -> None:
    if cdp_port == FORBIDDEN_CDP_PORT:
        raise ValueError("9333 ortak tarayıcıya ayrılmış; başka port seçin.")
    rec = Recorder(OUT_DIR)
    rec.link = BrowserLink(cdp_port, rec.on_cdp_event)
    server = ThreadingHTTPServer(("127.0.0.1", ui_port), make_handler(rec, asyncio.get_running_loop()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Kayıt arayüzü: http://127.0.0.1:{ui_port}  |  kayıtlar: {OUT_DIR}", flush=True)
    try:
        await rec.link.ensure()
    except Exception as exc:  # noqa: BLE001 - retried on Başlat
        print(f"Kayıt tarayıcısı açılamadı ({exc}); Başlat'ta tekrar denenecek.", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex insan gösterimi kaydedici")
    parser.add_argument("--cdp-port", type=int, default=9450)
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    asyncio.run(main_async(args.cdp_port, args.port))


if __name__ == "__main__":
    main()
