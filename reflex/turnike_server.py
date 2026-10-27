"""Standalone Local Turnike-4B-Web2-FP8 Server for Reflex (`POST /v1/systemone`).

Serves `/home/berkay/modeller/turnike-4b-web2-fp8` on `http://127.0.0.1:8794/v1/systemone` with:
- Single-token symbol readout (62 symbols x H in FP32, vision tower & 248k lm_head freed)
- `<= 60` option cap per question (avoids multi-pass tournament elimination)
- Two-stage inference (`iki_asama=True`: first `operation`, then `{chosen_op}_target`)
- Shared `## State` `past_key_values` `PrefixCache` (computes `## State` prefill once per step,
  reusing KV cache for `operation` and `_target` suffixes)
- `done_engeli=0.6` guard against premature `DONE`/`BLOCKED` on step 0 or low probability
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class _MockTorch:  # type: ignore[no-redef]
        @staticmethod
        def inference_mode(*_args: Any, **_kwargs: Any) -> Any:
            def decorator(fn: Any) -> Any:
                return fn
            return decorator

    torch = _MockTorch()  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    Qwen3_5ForConditionalGeneration = None  # type: ignore[assignment]

DEFAULT_MODEL_PATH = os.environ.get("TURNIKE_MODEL_PATH", "/home/berkay/modeller/turnike-4b-web2-fp8")
DEFAULT_MAX_LENGTH = 8192
MAX_OPTIONS = 60

ISARET = "<decision>"
YER_TOKEN = "?"
SEMBOLLER = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
SISTEM = (
    "You are a careful decision assistant. Use the state and decision schema in the user message to make the "
    "requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed "
    "options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names "
    "and symbols exactly as given. Do not include explanations, Markdown, or extra text."
)
KULLANICI_GIRIS = "Return one answer for every field using the supplied answer symbols.\n\n## State\n"
SCHEMA_MARKER = "\n## Decision schema\n"

SORU_METNI = {
    "operation": "Which single operation should be executed next on the current page to advance the goal?",
    "_target": "If the next operation is {op}, which observed element should it act on?",
}
GENEL_SORU = "Which option is the best decision for the goal?"


def _metin(deger: Any) -> str:
    if isinstance(deger, str):
        return deger
    if isinstance(deger, list):
        return "\n".join(_metin(v) for v in deger)
    return json.dumps(deger, ensure_ascii=False)


def _secenek_metni(deger: Any) -> str:
    if not isinstance(deger, dict):
        return _metin(deger)
    parcalar = [str(deger.get("element", ""))]
    if deger.get("current_value") not in (None, ""):
        parcalar.append(f"current value: {deger['current_value']}")
    parcalar += [f"{k}: {v}" for k, v in deger.items() if k not in ("element", "current_value")]
    return " | ".join(p for p in parcalar if p)


def durum_metni(state: Any, kurallar: list[str], hedef: str, max_elements: int = MAX_OPTIONS) -> str:
    satirlar: list[str] = []
    if hedef:
        satirlar.append(f"Goal: {hedef}")
    for kural in kurallar:
        satirlar.append(f"Rules:\n{kural}")
    if not isinstance(state, dict):
        satirlar.append(_metin(state))
        return "\n".join(satirlar)
    sayfa = state.get("page", {}) or {}
    satirlar.append(f"URL: {sayfa.get('url', '')}\nTitle: {sayfa.get('title', '')}")
    ogeler = (state.get("elements") or [])[:max_elements]
    if ogeler:
        satirlar.append("Elements:")
        for oge in ogeler:
            ek = {k: v for k, v in oge.items() if k not in ("index", "label", "role", "operations", "options")}
            ek_metin = "".join(f" | {k}: {v}" for k, v in ek.items() if v not in (None, ""))
            ops = ",".join(oge.get("operations", []))
            satirlar.append(f"[{oge.get('index')}] {oge.get('role', '')} {oge.get('label', '')}{ek_metin} | ops: {ops}")
            for secenek in (oge.get("options") or [])[:max_elements]:
                satirlar.append(f"  option [{secenek.get('index')}] {secenek.get('label', '')}")
    gecmis = state.get("recent_actions") or []
    if gecmis:
        satirlar.append("Recent actions:")
        satirlar += [
            f"- {h.get('kind')}: {h.get('action')}"
            + (f" text={h['text']!r}" if h.get("text") else "")
            + f" (page changed: {h.get('page_changed')})"
            for h in gecmis
        ]
    for anahtar, deger in state.items():
        if anahtar not in ("page", "elements", "recent_actions"):
            satirlar.append(f"{anahtar}: {_metin(deger)}")
    satirlar.append(f"Page text:\n{sayfa.get('text', '')}")
    return "\n".join(satirlar)


def soru_metni(anahtar: str, talimat: Any) -> str:
    if anahtar == "operation":
        soru = SORU_METNI["operation"]
    elif anahtar.endswith("_target"):
        op = talimat.get("operation", anahtar[: -len("_target")]) if isinstance(talimat, dict) else anahtar
        soru = SORU_METNI["_target"].format(op=op)
    else:
        soru = GENEL_SORU
    if isinstance(talimat, dict):
        ek = [f"{k}: {_metin(v)}" for k, v in talimat.items() if k not in ("rules", "goal", "operation")]
        return "\n".join([soru, *ek])
    return f"{_metin(talimat)}\n{soru}" if talimat else soru


def laya_istegi(govde: dict[str, Any], max_options: int = MAX_OPTIONS) -> tuple[str, dict[str, Any]]:
    sorular = govde.get("questions", {})
    kurallar: list[str] = []
    hedef = ""
    for soru in sorular.values():
        talimat = soru.get("instructions")
        if isinstance(talimat, dict):
            hedef = hedef or _metin(talimat.get("goal", ""))
            raw_rules = talimat.get("rules") or []
            for kural in raw_rules if isinstance(raw_rules, list) else [raw_rules]:
                if kural and kural not in kurallar:
                    kurallar.append(kural)
    donusmus: dict[str, Any] = {}
    for anahtar, soru in sorular.items():
        kriter = soru.get("criteria")
        if isinstance(kriter, dict):
            items = list(kriter.items())[:max_options]
            kriter = {str(k): _secenek_metni(v) for k, v in items}
        donusmus[anahtar] = {
            "type": soru.get("type", "choice"),
            "criteria": kriter,
            "instructions": soru_metni(anahtar, soru.get("instructions")),
        }
    return durum_metni(govde.get("state"), kurallar, hedef, max_elements=max_options), donusmus


def done_engelle(cevaplar: dict[str, Any], govde: dict[str, Any], esik: float) -> dict[str, Any]:
    """Prevent premature DONE on step 0 (`not gecmis`) when `P(DONE) < esik`.

    On subsequent steps (`len(gecmis) > 0`), preserve `DONE` when it is the model's argmax so
    Reflex's calibrated `DONE` gate (`op_min_conf["DONE"] = 0.72` + `verify_done_with_supervisor`)
    can verify completion with the System 2 Supervisor instead of forcing spurious clicks.
    """
    op = cevaplar.get("operation")
    if not isinstance(op, dict) or not isinstance(op.get("probabilities"), dict):
        return cevaplar
    secim = op.get("choice")
    if secim != "DONE":
        return cevaplar
    gecmis = (govde.get("state") or {}).get("recent_actions") if isinstance(govde.get("state"), dict) else None
    ilk_adim = not gecmis
    if not ilk_adim or float(op["probabilities"].get(secim, 0.0)) >= esik:
        return cevaplar
    digerleri = {k: v for k, v in op["probabilities"].items() if k not in ("DONE", "BLOCKED")}
    if not digerleri:
        return cevaplar
    yeni = max(digerleri, key=digerleri.get)
    olas = {k: float(v) for k, v in op["probabilities"].items()}
    olas[yeni] = round(olas.get(yeni, 0.0) + olas.get("DONE", 0.0), 4)
    olas["DONE"] = 0.0
    return {
        **cevaplar,
        "operation": {
            **op,
            "choice": yeni,
            "probabilities": olas,
            "confidence": olas[yeni],
            "engel": secim,
        },
    }


class TurnikeWeb2Engine:
    """Fast Turnike-4B-Web2-FP8 engine with shared `## State` PrefixCache (`past_key_values`)."""

    def __init__(
        self,
        model_path: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        done_engeli: float = 0.6,
        max_options: int = MAX_OPTIONS,
    ) -> None:
        effective_model_path = model_path or os.environ.get("TURNIKE_MODEL_PATH") or DEFAULT_MODEL_PATH
        self.model_path = Path(effective_model_path)
        self.max_length = max_length
        self.done_engeli = done_engeli
        if not HAS_TORCH or AutoTokenizer is None:
            raise ImportError(
                "Turnike server requires 'torch' and 'transformers'. "
                "Install them with `pip install .[turnike]`."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(self.model_path),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=str(self.device),
            local_files_only=True,
        ).eval()
        model.requires_grad_(False)

        self.yer_id = self.tok.encode(YER_TOKEN, add_special_tokens=False)[0]
        self.sembol_id = torch.tensor([self.tok.encode(s, add_special_tokens=False)[0] for s in SEMBOLLER])
        self.govde = model.model.language_model
        bas = model.lm_head
        self.sembol_agirlik = bas.weight[self.sembol_id.to(bas.weight.device)].detach().float().clone()
        model.lm_head = None
        if getattr(model.model, "visual", None) is not None:
            model.model.visual = None
        self.model = model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.ad = f"turnike-4b-web2-fp8 ({self.model_path.name})"
        self._warmup()
        print(f"[{self.ad}] Loaded and warmed up in {time.perf_counter() - t0:.2f}s", flush=True)

    def _warmup(self) -> None:
        dummy_govde = {
            "state": {
                "page": {"url": "https://example.com", "title": "Example", "text": "Warmup"},
                "elements": [{"index": "1", "role": "button", "label": "Start", "operations": ["CLICK"]}],
                "recent_actions": [],
            },
            "questions": {
                "operation": {
                    "type": "choice",
                    "criteria": {"CLICK": "Click", "DONE": "Done"},
                    "instructions": {"goal": "Click Start", "rules": "Click Start."},
                },
                "click_target": {
                    "type": "choice",
                    "criteria": {"1": {"element": "[1] Start"}},
                    "instructions": {"goal": "Click Start", "operation": "CLICK", "rules": ["Click Start."]},
                },
            },
        }
        self.karar(dummy_govde)

    def _fit_state(self, state: str, sorular: dict[str, Any]) -> str:
        """Ensure state + longest single question fits within `self.max_length`."""
        state = str(state).replace(ISARET, "<decision >")
        longest_q = max(
            sorular.items(),
            key=lambda kv: len( json.dumps(kv[1], ensure_ascii=False) ),
        )
        ad, soru = longest_q
        ciftler = [(str(k), str(v) if v not in (None, "") else str(k)) for k, v in (soru.get("criteria") or {}).items()]
        alanlar = [(ad, str(soru.get("instructions", "")), ciftler[: len(SEMBOLLER)])]
        full_txt = self._build_template(state, alanlar)
        ids, _ = self._split_and_tokenize(full_txt)
        if len(ids) <= self.max_length:
            return state

        empty_ids, _ = self._split_and_tokenize(self._build_template("", alanlar))
        st_ids = self.tok(state, add_special_tokens=False)["input_ids"]
        butce = max(self.max_length - len(empty_ids) - 64, 16)
        while True:
            cand_state = self.tok.decode(st_ids[:butce])
            ids, _ = self._split_and_tokenize(self._build_template(cand_state, alanlar))
            if len(ids) <= self.max_length or butce <= 16:
                return cand_state
            butce = int(butce * 0.9)

    def _build_template(self, state: str, alanlar: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
        satirlar: list[str] = []
        for ad, talimat, ciftler in alanlar:
            satirlar.append(f"{ad}: {talimat}")
            satirlar += [f"    {s} = {deger}: {aciklama}" for s, (deger, aciklama) in zip(SEMBOLLER, ciftler)]
        durum = json.dumps(state, ensure_ascii=False, indent=2)
        user_txt = KULLANICI_GIRIS + durum + SCHEMA_MARKER + "\n".join(satirlar)
        iskelet = json.dumps({ad: ISARET for ad, _, _ in alanlar}, ensure_ascii=False, indent=4)
        mesajlar = [
            {"role": "system", "content": SISTEM},
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": iskelet},
        ]
        return self.tok.apply_chat_template(
            mesajlar, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )

    def _split_and_tokenize(self, full_txt: str) -> tuple[list[int], list[int]]:
        parcalar = full_txt.split(ISARET)
        ids: list[int] = []
        konumlar: list[int] = []
        for i, parca in enumerate(parcalar):
            ids += self.tok(parca, add_special_tokens=False)["input_ids"]
            if i < len(parcalar) - 1:
                konumlar.append(len(ids) - 1)
                ids.append(self.yer_id)
        return ids, konumlar

    @torch.inference_mode()
    def _prefill_state(self, state: str, sample_alan: list[tuple[str, str, list[tuple[str, str]]]]) -> tuple[list[int], Any]:
        full_txt = self._build_template(state, sample_alan)
        pre_txt = full_txt.split(SCHEMA_MARKER, 1)[0] + SCHEMA_MARKER
        pre_ids = self.tok(pre_txt, add_special_tokens=False)["input_ids"]
        pre_tensor = torch.tensor([pre_ids], device=self.device)
        out = self.govde(input_ids=pre_tensor, use_cache=True)
        return pre_ids, out.past_key_values

    @torch.inference_mode()
    def _score_question_with_cache(
        self,
        state: str,
        ad: str,
        soru: dict[str, Any],
        pre_ids: list[int],
        past_key_values: Any,
    ) -> dict[str, Any]:
        crit = soru.get("criteria") or {}
        ciftler = [(str(k), str(v) if v not in (None, "") else str(k)) for k, v in list(crit.items())[: len(SEMBOLLER)]]
        etiketler = [k for k, _ in ciftler]
        if len(ciftler) == 1:
            return {"choice": etiketler[0], "confidence": 1.0, "probabilities": {etiketler[0]: 1.0}}

        alanlar = [(ad, str(soru.get("instructions", "")), ciftler)]
        full_txt = self._build_template(state, alanlar)
        ids, konumlar = self._split_and_tokenize(full_txt)
        suf_ids = ids[len(pre_ids) :]
        suf_konum = konumlar[0] - len(pre_ids)

        suf_tensor = torch.tensor([suf_ids], device=self.device)
        gizli = self.govde(
            input_ids=suf_tensor,
            past_key_values=past_key_values,
            use_cache=True,
        ).last_hidden_state[0, suf_konum]
        logit = torch.nn.functional.linear(gizli.float(), self.sembol_agirlik)[: len(ciftler)]
        probs = torch.softmax(logit, dim=-1).cpu().tolist()
        prob_map = {k: round(float(p), 6) for k, p in zip(etiketler, probs)}
        best = max(prob_map, key=prob_map.get)
        return {
            "choice": best,
            "confidence": round(prob_map[best], 4),
            "probabilities": prob_map,
        }

    @torch.inference_mode()
    def karar(self, govde: dict[str, Any]) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()

        state, sorular = laya_istegi(govde, max_options=self.max_options)
        if not sorular:
            raise ValueError("Empty questions in request")

        state = self._fit_state(state, sorular)
        cevaplar: dict[str, Any] = {}
        if "operation" in sorular:
            op_soru = sorular["operation"]
            op_ciftler = [
                (str(k), str(v) if v not in (None, "") else str(k))
                for k, v in list((op_soru.get("criteria") or {}).items())[: len(SEMBOLLER)]
            ]
            op_etiketler = [k for k, _ in op_ciftler]
            op_txt = self._build_template(state, [("operation", str(op_soru.get("instructions", "")), op_ciftler)])
            op_ids, op_kon = self._split_and_tokenize(op_txt)

            # Pass 1: score "operation" directly at op_kon[0] and build past_key_values
            pre_ids = op_ids[: op_kon[0] + 1]
            inp1 = torch.tensor([pre_ids], device=self.device)
            out1 = self.govde(input_ids=inp1, use_cache=True)
            gizli_op = out1.last_hidden_state[0, -1]
            logit_op = torch.nn.functional.linear(gizli_op.float(), self.sembol_agirlik)[: len(op_etiketler)]
            probs_op = torch.softmax(logit_op, dim=-1).cpu().tolist()
            pmap_op = {k: round(float(p), 6) for k, p in zip(op_etiketler, probs_op)}
            best_op = max(pmap_op, key=pmap_op.get)
            cevaplar["operation"] = {
                "choice": best_op,
                "confidence": round(pmap_op[best_op], 4),
                "probabilities": pmap_op,
            }
            if self.done_engeli > 0:
                cevaplar = done_engelle(cevaplar, govde, self.done_engeli)

            # Pass 2: only if the chosen operation requires a target
            hedef_anahtar = f"{str(cevaplar['operation'].get('choice', '')).lower()}_target"
            if hedef_anahtar in sorular:
                ts = sorular[hedef_anahtar]
                tc = [
                    (str(k), str(v) if v not in (None, "") else str(k))
                    for k, v in list((ts.get("criteria") or {}).items())[: len(SEMBOLLER)]
                ]
                t_etiketler = [k for k, _ in tc]
                if len(tc) == 1:
                    cevaplar[hedef_anahtar] = {
                        "choice": t_etiketler[0],
                        "confidence": 1.0,
                        "probabilities": {t_etiketler[0]: 1.0},
                    }
                else:
                    inst_str = str(ts.get("instructions", ""))
                    tgt_lines = [SCHEMA_MARKER, f"{hedef_anahtar}: {inst_str}"]
                    tgt_lines += [f"    {s} = {d}: {a}" for s, (d, a) in zip(SEMBOLLER, tc)]
                    suf_txt = (
                        "\n".join(tgt_lines)
                        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        + json.dumps({hedef_anahtar: ISARET}, ensure_ascii=False, indent=4)
                    )
                    suf_ids, suf_kon = self._split_and_tokenize(suf_txt)
                    op_rem = op_ids[op_kon[0] + 1 :]
                    tail_ids = (op_rem + suf_ids)[: len(op_rem) + suf_kon[0] + 1]
                    inp2 = torch.tensor([tail_ids], device=self.device)
                    out2 = self.govde(input_ids=inp2, past_key_values=out1.past_key_values, use_cache=True)
                    gizli_t = out2.last_hidden_state[0, -1]
                    logit_t = torch.nn.functional.linear(gizli_t.float(), self.sembol_agirlik)[: len(t_etiketler)]
                    probs_t = torch.softmax(logit_t, dim=-1).cpu().tolist()
                    pmap_t = {k: round(float(p), 6) for k, p in zip(t_etiketler, probs_t)}
                    best_t = max(pmap_t, key=pmap_t.get)
                    cevaplar[hedef_anahtar] = {
                        "choice": best_t,
                        "confidence": round(pmap_t[best_t], 4),
                        "probabilities": pmap_t,
                    }
        else:
            first_ad, first_soru = next(iter(sorular.items()))
            first_ciftler = [
                (str(k), str(v) if v not in (None, "") else str(k))
                for k, v in list((first_soru.get("criteria") or {}).items())[: len(SEMBOLLER)]
            ]
            pre_ids, pkv = self._prefill_state(
                state, [(first_ad, str(first_soru.get("instructions", "")), first_ciftler)]
            )
            items = list(sorular.items())
            for idx, (ad, soru) in enumerate(items):
                cur_pkv = copy.deepcopy(pkv) if idx < len(items) - 1 else pkv
                cevaplar[ad] = self._score_question_with_cache(state, ad, soru, pre_ids, cur_pkv)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sure_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answers": cevaplar,
            "model": self.ad,
            "usage": {"cost": 0.0},
            "sure_ms": sure_ms,
            "state_karakter": len(state),
            "prefix_tokens": len(pre_ids),
        }


def make_handler(engine: TurnikeWeb2Engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.rstrip("/") in ("/health", "/v1/health"):
                payload = json.dumps({"ok": True, "model": engine.ad}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            govde = json.loads(self.rfile.read(length))
            try:
                sonuc = engine.karar(govde)
                kod = 200
            except Exception as exc:
                sonuc = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                kod = 502
            veri = json.dumps(sonuc, ensure_ascii=False).encode("utf-8")
            self.send_response(kod)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(veri)))
            self.end_headers()
            self.wfile.write(veri)

        def log_message(self, *_):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex Local Turnike-4B-Web2-FP8 Server")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to Turnike model (defaults to TURNIKE_MODEL_PATH or default model path)",
    )
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--done-engeli", type=float, default=0.6)
    parser.add_argument("--max-options", type=int, default=MAX_OPTIONS)
    args = parser.parse_args()

    chosen_model = args.model or os.environ.get("TURNIKE_MODEL_PATH") or DEFAULT_MODEL_PATH
    engine = TurnikeWeb2Engine(
        model_path=chosen_model,
        max_length=args.max_length,
        done_engeli=args.done_engeli,
        max_options=args.max_options,
    )
    print(f"READY: http://127.0.0.1:{args.port}/v1/systemone", flush=True)
    HTTPServer(("127.0.0.1", args.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()
