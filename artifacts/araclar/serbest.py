"""Kayitsiz (insan referansi olmayan) gorevlerde Reflex kosusu: sadece-S2 + varsayilan S2 ayari.

Kullanim: REFLEX_KOD=<kod kopyasi> PYTHONPATH=<kod kopyasi> python serbest.py gorevler.json cikti_klasoru port_tabani
gorevler.json: [{"id": "...", "url": "...", "goal": "...", "max_steps": 40}, ...]
"""
import json, os, sys, time, traceback
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, os.environ["REFLEX_KOD"] + "/scripts")
import eval as ev  # noqa: E402
from reflex.agent import ReflexSession  # noqa: E402
from reflex.config import CalibrationThresholds, ReflexConfig  # noqa: E402

gorevler = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
port0 = int(sys.argv[3])
WALL = int(os.environ.get("KOS_SURE_SINIRI", "900"))

for i, g in enumerate(gorevler):
    if os.environ.get("KOS_MOD") == "kati":  # kos.py ile ayni kati hibrit: S1 yalniz emin tiklama/yazma
        cfg = ReflexConfig(decision_backend=os.environ.get("REFLEX_S1", "local_turnike"),
                           local_turnike_url=os.environ.get("TURNIKE_URL", "http://127.0.0.1:8794/v1/systemone"),
                           s1_strict=True, max_steps=g.get("max_steps", 40), max_escalations=1000, screenshots=False)
        cfg.thresholds = CalibrationThresholds(
            op_min_conf={**CalibrationThresholds().op_min_conf, "CLICK": 0.70, "TYPE_TEXT": 0.70, "SELECT": 0.70},
            target_min_conf=0.50, joint_min_conf=0.40, target_min_margin=0.15)
    else:
        cfg = ReflexConfig(s2_every_step=True, max_steps=g.get("max_steps", 40), max_escalations=1000, screenshots=False)
    if os.environ.get("REFLEX_TRACE_PATH"):
        cfg.trace_path = os.environ["REFLEX_TRACE_PATH"]
    if os.environ.get("KOS_S2_MODEL"):
        cfg.supervisor_model = os.environ["KOS_S2_MODEL"]
    sonuc = {"id": g["id"], "hedef": g["goal"], "baslangic": g["url"], "s2_model": cfg.supervisor_model, "mod": os.environ.get("KOS_MOD", "s2_only"), "s1": cfg.decision_backend if cfg.s1_strict else None}
    t0 = time.time(); session = None
    try:
        with ev.managed_isolated_browser(port=port0 + i):
            session = ReflexSession(url=g["url"], goal=g["goal"], config=cfg)
            while time.time() - t0 < WALL:
                if session.status == "waiting_confirmation":
                    session.confirm_action(approved=False); continue
                if session.status != "ready" or len(session.history) >= cfg.max_steps:
                    break
                session.step()
            snap = session.snapshot()
            hist = snap.get("history", [])
            chat = (snap.get("context") or {}).get("chat_log") or []
            cevap = next((m.get("content") for m in reversed(chat)
                          if m.get("role") == "assistant" and m.get("kind", "chat") == "chat"), None)
            ops = Counter(h.get("operation") for h in hist)
            notes = [str(h.get("escalation_note") or "") for h in hist]
            urls = [str(h.get("url") or "") for h in hist]
            url_ac = Counter(u for j, u in enumerate(urls) if u and (j == 0 or urls[j - 1] != u))
            sonuc.update({
                "durum": snap.get("status"), "son_hata": snap.get("last_error"), "adim": len(hist),
                "sure_sn": round(time.time() - t0, 1), "maliyet": (snap.get("metrics") or {}).get("total_cost_usd"),
                "token": (snap.get("metrics") or {}).get("usage"),
                "kaydirma": ops.get("SCROLL_DOWN", 0) + ops.get("SCROLL_UP", 0),
                "osilasyon_uyarisi": sum("[oscillation]" in n for n in notes),
                "dongu_uyarisi": sum("[loop_detected]" in n for n in notes),
                "tekrar_acilan_sayfa": {u: c for u, c in url_ac.items() if c > 1},
                "aktor": dict(Counter(str(h.get("actor")) for h in hist)),
                "cevap": cevap,
                "akis": [(h.get("operation"), str(h.get("action") or "")[:60], urlsplit(str(h.get("url") or "")).path[:60],
                          str(h.get("escalation_note") or "")[:120], h.get("actor")) for h in hist],
            })
    except Exception as exc:  # noqa: BLE001
        sonuc.update({"durum": "kosucu_hatasi", "son_hata": str(exc), "iz": traceback.format_exc()[-1200:],
                      "sure_sn": round(time.time() - t0, 1)})
    finally:
        if session is not None:
            try: session.close()
            except Exception: pass  # noqa: BLE001
    (out / f"{i:02d}_{g['id']}.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {g['id']}: {sonuc.get('durum')} | {sonuc.get('adim')} adim | {sonuc.get('sure_sn')} sn | "
          f"kaydirma {sonuc.get('kaydirma')} | osilasyon {sonuc.get('osilasyon_uyarisi')} | tekrar {len(sonuc.get('tekrar_acilan_sayfa') or {})}", flush=True)
