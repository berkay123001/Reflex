"""A/B: Reflex turnike_server hedef skorlama yolu vs resmi (temiz istem) yol, Jev kayitlarina karsi."""

import importlib.util
import json
import random
import sys
import time

import torch

KAYIT = sys.argv[1]
N = int(sys.argv[2])
CIKTI = sys.argv[3]

spec = importlib.util.spec_from_file_location("ts", "/home/berkay/Projeler/Reflex/reflex/turnike_server.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
eng = ts.TurnikeWeb2Engine(model_path="/home/berkay/modeller/turnike-4b-web2-fp8", done_engeli=0.0)
DEV = eng.device


def pairs(soru):
    crit = soru.get("criteria") or {}
    return [(str(k), str(v) if v not in (None, "") else str(k)) for k, v in list(crit.items())[: len(ts.SEMBOLLER)]]


def probs(h, n):
    return torch.softmax(torch.nn.functional.linear(h.float(), eng.sembol_agirlik)[:n], -1).cpu().tolist()


@torch.inference_mode()
def clean(state, ad, soru):
    """Resmi karar.py ile ayni: tek alanli, ayri ve temiz istem."""
    c = pairs(soru)
    ids, kon = eng._split_and_tokenize(eng._build_template(state, [(ad, str(soru.get("instructions", "")), c)]))
    h = eng.govde(input_ids=torch.tensor([ids], device=DEV), use_cache=False).last_hidden_state[0, kon[0]]
    return dict(zip([k for k, _ in c], probs(h, len(c))))


@torch.inference_mode()
def reflex_path(state, sorular, hedef):
    """turnike_server.karar() guncel hedef skorlama yolu; operasyon olasiliklari + verilen hedef anahtari."""
    op_soru = sorular["operation"]
    sample_alan = [("operation", str(op_soru.get("instructions", "")), pairs(op_soru))]
    pre_ids, pkv = eng._prefill_state(state, sample_alan)
    op_ans = eng._score_question_with_cache(state, "operation", op_soru, pre_ids, pkv)
    op_p = op_ans.get("probabilities", {})
    if hedef is None or hedef not in sorular:
        return op_p, None
    t_ans = eng._score_question_with_cache(state, hedef, sorular[hedef], pre_ids, pkv)
    t_p = t_ans.get("probabilities", {})
    return op_p, t_p


def choice_of(ans):
    if isinstance(ans, dict):
        if ans.get("choice") is not None:
            return str(ans["choice"])
        p = ans.get("probabilities")
        if isinstance(p, dict) and p:
            return max(p, key=p.get)
    return None


uygun = []
for line in open(KAYIT, encoding="utf-8"):
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    q = (d.get("istek") or {}).get("questions") or {}
    ans = ((d.get("yanit") or {}).get("answers")) or {}
    jop = choice_of(ans.get("operation"))
    if "operation" not in q or jop is None:
        continue
    hk = f"{jop.lower()}_target"
    n_opt = len((q.get(hk) or {}).get("criteria") or {})
    if hk in q and choice_of(ans.get(hk)) is not None and not (2 <= n_opt <= 60):
        continue
    if len((q["operation"].get("criteria") or {})) > 60:
        continue
    uygun.append(d)

random.Random(0).shuffle(uygun)
ornek = uygun[:N]
print(f"uygun kayit: {len(uygun)}, ornek: {len(ornek)}", flush=True)

# Dogrulama: kopyalanan yol, engine.karar() ile ayni sonucu veriyor mu?
for d in ornek[:4]:
    govde = d["istek"]
    state, sorular = ts.laya_istegi(govde, max_options=eng.max_options)
    state = eng._fit_state(state, sorular)
    resmi = eng.karar(govde)["answers"]
    rop = resmi["operation"]["choice"]
    hk = f"{rop.lower()}_target"
    op_p, t_p = reflex_path(state, sorular, hk if hk in sorular else None)
    assert max(op_p, key=op_p.get) == rop, "operasyon kopyasi uyusmuyor"
    if t_p is not None and len(t_p) > 1:
        assert max(t_p, key=t_p.get) == resmi[hk]["choice"], "hedef kopyasi uyusmuyor"
print("kopya dogrulandi (engine.karar ile ayni)", flush=True)

t0 = time.time()
with open(CIKTI, "w", encoding="utf-8") as f:
    for i, d in enumerate(ornek):
        govde = d["istek"]
        ans = d["yanit"]["answers"]
        jop = choice_of(ans["operation"])
        hk = f"{jop.lower()}_target"
        state, sorular = ts.laya_istegi(govde, max_options=eng.max_options)
        state = eng._fit_state(state, sorular)
        has_t = hk in sorular and choice_of(ans.get(hk)) is not None
        op_p, mal = reflex_path(state, sorular, hk if has_t else None)
        cln = clean(state, hk, sorular[hk]) if has_t else None
        kayit = {
            "i": i,
            "jev_op": jop,
            "turnike_op": max(op_p, key=op_p.get),
            "turnike_op_p": round(max(op_p.values()), 4),
            "hedef": hk if has_t else None,
            "jev_t": choice_of(ans.get(hk)) if has_t else None,
            "reflex_t": max(mal, key=mal.get) if mal else None,
            "reflex_t_p": sorted(mal.values(), reverse=True)[:2] if mal else None,
            "temiz_t": max(cln, key=cln.get) if cln else None,
            "temiz_t_p": sorted(cln.values(), reverse=True)[:2] if cln else None,
            "n_opt": len(cln) if cln else None,
        }
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
        f.flush()
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(ornek)}  {time.time() - t0:.0f}s", flush=True)
print("bitti", flush=True)
