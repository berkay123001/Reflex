"""Kullanici kayitlarindan secilen senaryolarda Reflex (S1 Jev, S2 Muse Spark) kosusu; insanla yan yana rapor."""
import json
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

import os
sys.path.insert(0, os.environ["REFLEX_KOD"] + "/scripts")
import eval as ev  # noqa: E402  (HEAD: izole tarayici + harness NAME duzeltmesi)
from reflex.agent import ReflexSession  # noqa: E402
from reflex.config import CalibrationThresholds, ReflexConfig  # noqa: E402

DEMO = Path("/home/berkay/Projeler/Reflex/artifacts/demolar")
OUT = Path(os.environ.get("KOS_CIKTI", "/tmp/altin/sonuc"))
OUT.mkdir(parents=True, exist_ok=True)
S1 = os.environ.get("REFLEX_S1", "openrouter_jev")
TURNIKE_URL = os.environ.get("TURNIKE_URL", "http://127.0.0.1:8794/v1/systemone")
WALL_LIMIT_S = int(os.environ.get("KOS_SURE_SINIRI", "480"))
ADIM_SINIRI = {"okb_testi": 80, "quizei_siralama": 80}  # 48 adimlik insan kaydi + kaydirma; ilerlemenin korunmasini olcmek icin

SENARYOLAR = [
    ("kor_siralama", "20261028-163454", None),
    ("quizei_deniz", "20261028-184414", None),
    ("wikipedia", "20261028-165324", None),
    ("flights", "20261028-171401", None),
    ("yeditepe", "20261028-174854", None),
    ("baykara_ofis", "20261028-163944", None),
    ("hf_populer", "20261028-175104", None),
    ("okb_testi", "20261028-180535",
     "benim için OKB bozukluk testlerine bak ve beğendiğin bir tanesini rastgele çöz. "
     "Sonuç için e-posta veya kişisel bilgi isterse girme, orada dur ve bana bildir."),
    ("ml_hackathon", "20261028-184728", None),
    ("ai_yazilim_gorus", "20261028-191510", None),
]

# ML-T1 bolme.json "test" oturumlari: egitimde hic gorulmemis gorevler (ayni aileden ikinci quiz atlandi).
SENARYOLAR_TEST = [
    ("baykara_tasarim", "20261028-163715", None),
    ("baykara_ofis", "20261028-163944", None),
    ("intertene_quiz", "20261028-173736", None),
    ("ozal_arastirma", "20261028-180223", None),
    ("kizlarsoruyor", "20261028-180707", None),
    ("quizei_siralama", "20261028-184222", None),
    ("cv_arastirma", "20261028-190803", None),
    ("freebuds_fiyat", "20261029-130506", None),
    ("kulaklik_platform", "20261029-130911", None),
]
if os.environ.get("KOS_SET") == "test":
    SENARYOLAR = SENARYOLAR_TEST
if os.environ.get("KOS_SADECE"):  # virgulle ayrilmis senaryo adlari
    _sec = set(os.environ["KOS_SADECE"].split(","))
    SENARYOLAR = [x for x in SENARYOLAR + SENARYOLAR_TEST if x[0] in _sec]


def insan(prefix):
    f = next(DEMO.glob(prefix + "*.jsonl"))
    rows = [json.loads(line) for line in f.open(encoding="utf-8")]
    meta, end = rows[0], rows[-1]
    steps = [r for r in rows if r["type"] == "step"]
    hosts = []
    for s in steps:
        h = urlsplit(s.get("url") or "").hostname or ""
        if h and (not hosts or hosts[-1] != h):
            hosts.append(h)
    return meta, {"adim": end.get("steps"), "sure_sn": end.get("duration_s"), "basari": end.get("success"),
                  "siteler": hosts, "not": end.get("note")}


def kos(i, ad, prefix, goal_override):
    meta, ref = insan(prefix)
    goal = goal_override or meta["goal"]
    max_steps = ADIM_SINIRI.get(ad) or max(25, min(60, int(ref["adim"] or 20) * 2))
    cfg = ReflexConfig(decision_backend=S1, local_turnike_url=TURNIKE_URL, max_steps=max_steps, screenshots=False)
    if os.environ.get("REFLEX_TRACE_PATH"):
        cfg.trace_path = os.environ["REFLEX_TRACE_PATH"]
    if os.environ.get("KOS_S2_MODEL"):
        cfg.supervisor_model = os.environ["KOS_S2_MODEL"]
    if os.environ.get("KOS_MOD") == "kati":  # kati hibrit: S1 yalniz emin tiklama/yazma; kaydirma ve keyfi secim S2'de
        cfg.s1_strict = True
        cfg.thresholds = CalibrationThresholds(
            op_min_conf={**CalibrationThresholds().op_min_conf, "CLICK": 0.70, "TYPE_TEXT": 0.70, "SELECT": 0.70},
            target_min_conf=0.50, joint_min_conf=0.40, target_min_margin=0.15)
        cfg.max_escalations = 1000
    if os.environ.get("KOS_MOD") == "s2_only":  # ablasyon: S1 hic cagrilmaz, her adimi S2 verir
        cfg.s2_every_step = True
        cfg.max_escalations = 1000
    sonuc = {"id": ad, "mod": os.environ.get("KOS_MOD", "hibrit"), "s2_model": cfg.supervisor_model, "s1": S1, "turnike_url": TURNIKE_URL if S1 == "local_turnike" else None, "hedef": goal, "baslangic": meta["start_url"], "insan": ref, "max_steps": max_steps}
    t0 = time.time()
    session = None
    reddedilen_onay = 0
    try:
        with ev.managed_isolated_browser(port=int(os.environ.get("KOS_PORT_TABANI", "9470")) + i):
            session = ReflexSession(url=meta["start_url"], goal=goal, config=cfg)
            while time.time() - t0 < WALL_LIMIT_S:
                if session.status == "waiting_confirmation":
                    reddedilen_onay += 1
                    session.confirm_action(approved=False)
                    continue
                if session.status != "ready" or len(session.history) >= max_steps:
                    break
                session.step()
            snap = session.snapshot()
            chat = (snap.get("context") or {}).get("chat_log") or []
            cevap = next((m.get("content") for m in reversed(chat)
                          if m.get("role") == "assistant" and m.get("kind", "chat") == "chat"), None)
            sorular = [m.get("content") for m in chat if m.get("kind") == "ask_user"]
            hosts = []
            for h in snap.get("history", []):
                host = urlsplit(str(h.get("url") or "")).hostname or ""
                if host and (not hosts or hosts[-1] != host):
                    hosts.append(host)
            met = snap.get("metrics", {})
            sonuc.update({
                "durum": snap.get("status"), "son_hata": snap.get("last_error"), "adim": len(snap.get("history", [])),
                "s2": met.get("s2_escalations"), "maliyet": met.get("total_cost_usd"), "token": met.get("usage"),
                "sure_sn": round(time.time() - t0, 1), "siteler": hosts, "cevap": cevap, "sorular": sorular,
                "reddedilen_onay": reddedilen_onay,
                "plan": (snap.get("context") or {}).get("plan"),
                "notlar": [f"{m.get('kind')}: {str(m.get('content'))[:300]}" for m in chat
                           if m.get("kind") in ("escalation_note", "chat", "ask_user")][-25:],
                "akis": [(h.get("actor"), h.get("operation"), str(h.get("action") or "")[:50],
                          urlsplit(str(h.get("url") or "")).hostname, h.get("escalation_note"),
                          {k: h.get(k) for k in ("s1_ms", "s2_ms", "act_ms", "observe_ms", "step_ms")})
                         for h in snap.get("history", [])],
            })
    except Exception as exc:  # noqa: BLE001
        sonuc.update({"durum": "kosucu_hatasi", "son_hata": f"{exc}", "iz": traceback.format_exc()[-1500:],
                      "sure_sn": round(time.time() - t0, 1)})
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass
    (OUT / f"{i:02d}_{ad}.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{time.strftime('%H:%M:%S')}] {ad}: {sonuc.get('durum')} | {sonuc.get('adim')} adim "
          f"(insan {ref['adim']}) | {sonuc.get('sure_sn')} sn (insan {ref['sure_sn']}) | "
          f"siteler {len(sonuc.get('siteler') or [])} (insan {len(ref['siteler'])}) | cevap: {str(sonuc.get('cevap'))[:90]!r}",
          flush=True)


secim = sys.argv[1:]  # bos = hepsi
for i, (ad, prefix, goal) in enumerate(SENARYOLAR):
    if not secim or ad in secim:
        kos(i, ad, prefix, goal)
print("BITTI", flush=True)
