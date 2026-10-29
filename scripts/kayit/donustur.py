"""İnsan gösterim kayıtlarını (artifacts/demolar/*.jsonl) System 1 eğitim kaydına çevirir.

Kullanım:
    .venv/bin/python scripts/kayit/donustur.py [--girdi artifacts/demolar] [--cikti artifacts/demolar_cikti]
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reflex.browser.cdp import canonical_url, fingerprint
from reflex.config import ReflexConfig
from reflex.context import DualTierContext
from reflex.system1 import MAX_SYSTEM1_ELEMENTS, build_system1_request

DEFAULT_IN_DIR = ROOT / "artifacts" / "demolar"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "demolar_cikti"
SCROLL_STEP_PX = 420.0
MIN_SCROLL_MERGE_PX = 315.0  # 420 px kaydırma adımının %75'i
SAHTE_GECMIS_PENCERE_SN = 2.0
ALTIN_KUTLE = 0.9
HEDEF_OPLAR = {"CLICK", "TYPE_TEXT", "SELECT"}
KAYDIRMA_OPLAR = {"SCROLL_DOWN", "SCROLL_UP"}
GECMIS_OPLAR = {"GO_BACK", "GO_FORWARD", "RELOAD"}
OP_TO_KIND = {
    "CLICK": "click",
    "TYPE_TEXT": "fill",
    "SELECT": "select",
    "SCROLL_DOWN": "scroll",
    "SCROLL_UP": "scroll",
    "GO_BACK": "go_back",
    "GO_FORWARD": "go_forward",
    "RELOAD": "reload",
    "UNMAPPED_CLICK": "click",
}


def op_dagilimi_yuzde(op_sayilari: dict[str, int]) -> dict[str, float]:
    """Operasyon sayılarını toplam içindeki yüzdelerine dönüştürür (toplam 100.0)."""
    toplam = sum(op_sayilari.values())
    if toplam <= 0:
        return {}
    return {op: round((count / toplam) * 100.0, 2) for op, count in op_sayilari.items()}


def haric_tutulanlari_yukle(haric_yolu: Path | None = None) -> list[dict[str, Any]]:
    """artifacts/demolar/haric.json dosyasını yükler, yoksa boş liste döner."""
    yol = haric_yolu or (ROOT / "artifacts" / "demolar" / "haric.json")
    if not yol.is_file():
        return []
    try:
        with yol.open("r", encoding="utf-8") as f:
            veri = json.load(f)
            return veri if isinstance(veri, list) else []
    except Exception:
        return []


def adim_haric_mi(
    oturum_adi: str,
    adim_no: int,
    haric_listesi: list[dict[str, Any]],
) -> str | None:
    """Belirtilen adımın haric.json kurallarına göre elenip elenmeyeceğini kontrol eder."""
    for kural in haric_listesi:
        hedef = str(kural.get("oturum", ""))
        if hedef == oturum_adi or fnmatch.fnmatch(oturum_adi, hedef):
            aralik = kural.get("adimlar")
            if isinstance(aralik, (list, tuple)) and len(aralik) == 2:
                ilk, son = aralik
                if ilk <= adim_no <= son:
                    return str(kural.get("neden") or "haric_tutuldu")
    return None


def kaydirma_kosularini_siniflandir(
    adimlar: list[dict[str, Any]],
    haric_listesi: list[dict[str, Any]],
    oturum_adi: str,
) -> dict[int, str]:
    """Her kaydırma adımı için elenme nedenini (okuma_kaydirmasi, uzun_kosu_kirpma, haric) belirler."""
    elenme_nedenleri: dict[int, str] = {}
    n = len(adimlar)

    # 1. haric.json kurallarını uygula
    for i, adim in enumerate(adimlar):
        adim_no = adim.get("step")
        if type(adim_no) is not int:
            adim_no = i + 1
        neden = adim_haric_mi(oturum_adi, adim_no, haric_listesi)
        if neden:
            elenme_nedenleri[i] = neden

    # 2. Kaydırma koşularını belirle ve filtrele
    i = 0
    while i < n:
        op = str(adimlar[i].get("op") or "")
        if op not in KAYDIRMA_OPLAR:
            i += 1
            continue

        start_idx = i
        while i < n and str(adimlar[i].get("op") or "") in KAYDIRMA_OPLAR:
            i += 1
        end_idx = i - 1
        run_indices = list(range(start_idx, end_idx + 1))

        aktif_indeksler = [idx for idx in run_indices if idx not in elenme_nedenleri]
        if not aktif_indeksler:
            continue

        # D3 Otomatik hariç tutma: bitişiğinde UNMAPPED_CLICK olan koşular
        prev_is_unmapped = (start_idx > 0 and str(adimlar[start_idx - 1].get("op") or "") == "UNMAPPED_CLICK")
        next_is_unmapped = (end_idx + 1 < n and str(adimlar[end_idx + 1].get("op") or "") == "UNMAPPED_CLICK")
        if prev_is_unmapped or next_is_unmapped:
            for idx in aktif_indeksler:
                elenme_nedenleri[idx] = "unmapped_click_bitisigi"
            continue

        # D2: Koşudan sonraki ilk adım var mı?
        if end_idx + 1 >= n:
            for idx in aktif_indeksler:
                elenme_nedenleri[idx] = "okuma_kaydirmasi"
            continue

        sonraki_adim = adimlar[end_idx + 1]
        sonraki_op = str(sonraki_adim.get("op") or "")
        sonraki_node = sonraki_adim.get("node")

        bas_state = adimlar[start_idx].get("state") or {}
        bas_actions = bas_state.get("actions") or []
        bas_nodeler = {a.get("node") for a in bas_actions if type(a.get("node")) is int}

        uygun_etkilesim = (
            sonraki_op in HEDEF_OPLAR
            and type(sonraki_node) is int
            and sonraki_node not in bas_nodeler
        )

        if not uygun_etkilesim:
            for idx in aktif_indeksler:
                elenme_nedenleri[idx] = "okuma_kaydirmasi"
            continue

        # Uzun koşu kırpma: 4'ten uzunsa ilk adım ve son 3 adım tutulur
        run_len = len(run_indices)
        if run_len > 4:
            kirpilacaklar = run_indices[1 : run_len - 3]
            for idx in kirpilacaklar:
                if idx not in elenme_nedenleri:
                    elenme_nedenleri[idx] = "uzun_kosu_kirpma"

    return elenme_nedenleri


def _adim_scroll_y(adim: dict[str, Any]) -> float:
    sy = adim.get("scroll_y")
    if sy is None:
        sy = ((adim.get("state") or {}).get("scroll") or {}).get("y", 0)
    return float(sy or 0.0)


def sahte_gecmis_temizle(
    adimlar: list[dict[str, Any]],
    pencere_sn: float = SAHTE_GECMIS_PENCERE_SN,
) -> tuple[list[dict[str, Any]], int]:
    """D5: iframe kaynaklı sahte geçmiş adımlarını (target_url http/https olmayan veya
    bir geçmiş adımından sonraki 2 sn içinde gelen RELOAD/GO_FORWARD) filtreler."""
    tutulan: list[dict[str, Any]] = []
    atilan = 0
    son_gecmis_zamani: float | None = None
    son_tutulan_gecmis_zamani: float | None = None

    for adim in adimlar:
        op = str(adim.get("op") or "")
        if op not in GECMIS_OPLAR:
            tutulan.append(adim)
            continue

        zaman_ham = adim.get("time")
        zaman = float(zaman_ham) if isinstance(zaman_ham, (int, float)) else None

        target_url = str(adim.get("target_url") or "")
        http_degil = not target_url.startswith(("http://", "https://"))

        cok_yakin = False
        if op in {"RELOAD", "GO_FORWARD"} and zaman is not None:
            if son_gecmis_zamani is not None and (zaman - son_gecmis_zamani) <= pencere_sn or son_tutulan_gecmis_zamani is not None and (zaman - son_tutulan_gecmis_zamani) <= pencere_sn:
                cok_yakin = True

        if zaman is not None:
            son_gecmis_zamani = zaman

        if http_degil or cok_yakin:
            atilan += 1
            continue

        if zaman is not None:
            son_tutulan_gecmis_zamani = zaman
        tutulan.append(adim)

    return tutulan, atilan


def kaydirma_birlestir(
    adimlar: list[dict[str, Any]],
    esik_px: float = MIN_SCROLL_MERGE_PX,
) -> tuple[list[dict[str, Any]], int]:
    """Art arda gelen aynı yönlü SCROLL adımlarında ilkini ve son tutulandan >=315px uzak olanları tut."""
    tutulan: list[dict[str, Any]] = []
    atilan = 0
    son_kaydirma_op: str | None = None
    son_tutulan_y: float | None = None

    for adim in adimlar:
        op = str(adim.get("op") or "")
        if op in KAYDIRMA_OPLAR:
            y = _adim_scroll_y(adim)
            if son_kaydirma_op == op and son_tutulan_y is not None:
                if abs(y - son_tutulan_y) < esik_px:
                    atilan += 1
                    continue
            son_kaydirma_op = op
            son_tutulan_y = y
            tutulan.append(adim)
        else:
            son_kaydirma_op = None
            son_tutulan_y = None
            tutulan.append(adim)

    return tutulan, atilan


def olasilik_dagilimi(
    kriter: dict[str, Any] | list[str],
    altin_secim: str,
    kutle: float = ALTIN_KUTLE,
) -> dict[str, float]:
    """Laya tek_sicak/yumusak ile aynı: altın seçenek 0.9, kalan 0.1 diğerlerine eşit; tek seçenekte 1.0."""
    anahtarlar = list(kriter.keys()) if isinstance(kriter, dict) else list(kriter)
    if altin_secim not in anahtarlar:
        raise ValueError(f"Altın seçim ({altin_secim!r}) seçenek anahtarlarında yok: {anahtarlar}")
    n = len(anahtarlar)
    if n == 1:
        return {anahtarlar[0]: 1.0}
    diger = round((1.0 - kutle) / (n - 1), 8)
    altin = round(kutle, 8)
    return {k: (altin if k == altin_secim else diger) for k in anahtarlar}


def _atilma_nedeni_bul(
    op: str,
    node: Any,
    value: Any,
    ham_actions: list[dict[str, Any]],
) -> str:
    """Eşleşmeyen CLICK/TYPE_TEXT/SELECT adımının neden S1 hedeflerinde yer almadığını sınıflandır."""
    if type(node) is not int:
        return "oge_s1_listesinde_degil"
    node_eylemleri = [a for a in ham_actions if a.get("node") == node]
    if not node_eylemleri:
        return "oge_s1_listesinde_degil"

    if op == "CLICK":
        for a in node_eylemleri:
            rol = str(a.get("role", "")).lower()
            if (
                a.get("kind") == "click"
                and rol in {"radio", "menuitemradio"}
                and str(a.get("checked", "")).lower() == "true"
            ):
                return "isaretli_radio_filtresi"
        if any(a.get("kind") == "click" for a in node_eylemleri):
            return f"{MAX_SYSTEM1_ELEMENTS}_siniri"
        return "oge_s1_listesinde_degil"

    if op == "TYPE_TEXT":
        if any(a.get("kind") == "fill" for a in node_eylemleri):
            return f"{MAX_SYSTEM1_ELEMENTS}_siniri"
        return "oge_s1_listesinde_degil"

    if op == "SELECT":
        sec_eylemleri = [a for a in node_eylemleri if a.get("kind") == "select"]
        if not sec_eylemleri:
            return "oge_s1_listesinde_degil"
        if any(str(a.get("value", "")) == str(value or "") for a in sec_eylemleri):
            return f"{MAX_SYSTEM1_ELEMENTS}_siniri"
        return "secenek_bulunamadi"

    return "oge_s1_listesinde_degil"


def altin_eslestir(
    adim: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    controls: dict[str, dict[str, Any]],
    operation_criteria: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, dict[str, Any] | None, str | None]:
    """Kayıttaki insan adımını S1 (operation, target_choice, matched_action, drop_reason) ile eşleştir."""
    op = str(adim.get("op") or "")
    state = adim.get("state") or {}
    ham_actions = state.get("actions") or []

    if op in ("CLICK", "TYPE_TEXT"):
        node = adim.get("node")
        adaylar = targets.get(op) or {}
        if type(node) is int:
            for idx, aksiyon in adaylar.items():
                if aksiyon.get("node") == node:
                    return op, idx, aksiyon, None
        return None, None, None, _atilma_nedeni_bul(op, node, None, ham_actions)

    if op == "SELECT":
        node = adim.get("node")
        val = adim.get("value")
        adaylar = targets.get("SELECT") or {}
        if type(node) is int and val is not None:
            for idx, aksiyon in adaylar.items():
                if aksiyon.get("node") == node and str(aksiyon.get("value", "")) == str(val):
                    return "SELECT", idx, aksiyon, None
        return None, None, None, _atilma_nedeni_bul("SELECT", node, val, ham_actions)

    if op in KAYDIRMA_OPLAR:
        if op in controls and (operation_criteria is None or op in operation_criteria):
            return op, None, controls[op], None
        for k, ctrl in controls.items():
            if k.upper() == op and (operation_criteria is None or k in operation_criteria):
                return k, None, ctrl, None
        return None, None, None, "kontrol_yok"

    if op == "DONE":
        if operation_criteria is not None and "DONE" not in operation_criteria:
            return None, None, None, "kontrol_yok"
        return "DONE", None, None, None

    if op in {"GO_BACK", "GO_FORWARD", "RELOAD"}:
        return None, None, None, "s1_disi_op"

    if op == "UNMAPPED_CLICK":
        return None, None, None, "unmapped_click"

    return None, None, None, "bilinmeyen_op"


def gecmis_kaydi_kur(
    adim: dict[str, Any],
    eslesen_aksiyon: dict[str, Any] | None,
    page_changed: bool,
) -> dict[str, Any]:
    """agent.py'nin history kayıtlarıyla aynı alanları (action, kind, text, page_changed) üret."""
    op = str(adim.get("op") or "")
    if op == "GO_BACK":
        return {"action": "Go back", "kind": "go_back", "text": None, "page_changed": bool(page_changed)}
    if op == "GO_FORWARD":
        return {"action": "Go forward", "kind": "go_forward", "text": None, "page_changed": bool(page_changed)}
    if op == "RELOAD":
        return {"action": "Reload", "kind": "reload", "text": None, "page_changed": bool(page_changed)}
    if op == "UNMAPPED_CLICK":
        unmapped = adim.get("unmapped") or {}
        text = str(unmapped.get("text") or adim.get("label") or "")
        return {"action": text, "kind": "click", "text": None, "page_changed": bool(page_changed)}

    kind = OP_TO_KIND.get(op, (eslesen_aksiyon or {}).get("kind", ""))
    if op == "SCROLL_DOWN":
        etiket = str((eslesen_aksiyon or {}).get("label") or "Scroll down")
    elif op == "SCROLL_UP":
        etiket = str((eslesen_aksiyon or {}).get("label") or "Scroll up")
    else:
        ham_etiket = adim.get("label")
        if ham_etiket is None or str(ham_etiket) == "":
            ham_etiket = (eslesen_aksiyon or {}).get("label") or ""
        etiket = str(ham_etiket)

    text_val = adim.get("text") if op == "TYPE_TEXT" else None
    return {
        "action": etiket,
        "kind": kind,
        "text": text_val,
        "page_changed": bool(page_changed),
    }


def kayit_dogrula(kayit: dict[str, Any]) -> None:
    """Üretilen her eğitim kaydının şema, seçenek anahtarı ve olasılık toplamı kurallarını doğrula."""
    beklenen_ust_alanlar = {"istek", "yanit", "kaynak", "oturum", "adim", "insan"}
    if set(kayit.keys()) != beklenen_ust_alanlar:
        raise ValueError(f"Kayıt üst alanları uyumsuz: {set(kayit.keys())} != {beklenen_ust_alanlar}")
    if kayit.get("kaynak") != "insan_demo":
        raise ValueError(f"kaynak alanı 'insan_demo' olmalı: {kayit.get('kaynak')}")
    beklenen_insan_alanlari = {"op", "node", "label", "text", "option"}
    insan = kayit.get("insan")
    if not isinstance(insan, dict) or set(insan.keys()) != beklenen_insan_alanlari:
        raise ValueError(f"insan alanı şeması uyumsuz: {insan}")

    istek = kayit.get("istek") or {}
    yanit = kayit.get("yanit") or {}
    sorular = istek.get("questions") or {}
    cevaplar = yanit.get("answers") or {}

    if "operation" not in cevaplar or "operation" not in sorular:
        raise ValueError("Kayıtta 'operation' sorusu veya yanıtı eksik.")

    altin_op = cevaplar["operation"].get("choice")
    beklenen_anahtarlar = {"operation"}
    if altin_op in HEDEF_OPLAR:
        beklenen_anahtarlar.add(f"{altin_op.lower()}_target")

    if set(cevaplar.keys()) != beklenen_anahtarlar:
        raise ValueError(
            f"Hedef sorusu yalnız altın op ({altin_op}) için olmalı: "
            f"beklenen={beklenen_anahtarlar}, bulunan={set(cevaplar.keys())}"
        )

    for soru_adi, cevap in cevaplar.items():
        if soru_adi not in sorular:
            raise ValueError(f"Yanıttaki soru ({soru_adi!r}) istek.questions içinde yok.")
        kriter = (sorular[soru_adi] or {}).get("criteria") or {}
        secim = cevap.get("choice")
        if secim not in kriter:
            raise ValueError(
                f"{soru_adi}: choice={secim!r} ilgili sorunun criteria anahtarlarında yok ({list(kriter.keys())})."
            )
        olasiliklar = cevap.get("probabilities") or {}
        if set(olasiliklar.keys()) != set(kriter.keys()):
            raise ValueError(
                f"{soru_adi}: olasılık anahtarları criteria anahtarlarıyla birebir aynı değil."
            )
        toplam = sum(float(v) for v in olasiliklar.values())
        if abs(toplam - 1.0) > 1e-6:
            raise ValueError(f"{soru_adi}: olasılık toplamı 1 (±1e-6) değil: {toplam}")


def oturum_satirlarini_donustur(
    satirlar: list[dict[str, Any]],
    oturum_adi: str,
    *,
    basarisizlari_dahil_et: bool = False,
    config: ReflexConfig | None = None,
    haric_listesi: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tek bir oturumun JSONL satırlarını S1 eğitim kayıtlarına ve oturum raporuna dönüştür."""
    cfg = config or ReflexConfig()
    meta = next((s for s in satirlar if s.get("type") == "session"), {})
    bitis = next((s for s in reversed(satirlar) if s.get("type") == "session_end"), {})
    ham_adimlar = [s for s in satirlar if s.get("type") == "step"]

    goal = str(meta.get("goal") or "").strip()
    success = bool(bitis.get("success"))
    birlestirme_oncesi_op = Counter(str(a.get("op") or "") for a in ham_adimlar)

    type_text_enter_sayisi = sum(
        1 for a in ham_adimlar
        if str(a.get("op") or "") == "TYPE_TEXT" and bool(a.get("submitted_with_enter"))
    )
    unmapped_click_sayisi = sum(
        1 for a in ham_adimlar
        if str(a.get("op") or "") == "UNMAPPED_CLICK"
    )

    temiz_adimlar, sahte_gecmis_sayisi = sahte_gecmis_temizle(ham_adimlar)

    if not success and not basarisizlari_dahil_et:
        return [], {
            "oturum": oturum_adi,
            "hedef": goal,
            "success": success,
            "dahil_edildi": False,
            "ham_adim": len(ham_adimlar),
            "birlestirme_sonrasi_adim": 0,
            "tutulan_adim": 0,
            "atilan_adim": len(ham_adimlar),
            "type_text_enter_sayisi": type_text_enter_sayisi,
            "unmapped_click_sayisi": unmapped_click_sayisi,
            "sahte_gecmis": sahte_gecmis_sayisi,
            "atilan_nedenleri": {"basarisiz_oturum": len(ham_adimlar)} if ham_adimlar else {},
            "birlestirme_oncesi_op": dict(birlestirme_oncesi_op),
            "birlestirme_oncesi_op_yuzde": op_dagilimi_yuzde(dict(birlestirme_oncesi_op)),
            "birlestirme_sonrasi_op": {},
            "birlestirme_sonrasi_op_yuzde": {},
            "tutulan_op": {},
            "tutulan_op_yuzde": {},
        }

    birlestirilmis_adimlar, kaydirma_atilan = kaydirma_birlestir(temiz_adimlar)
    birlestirme_sonrasi_op = Counter(str(a.get("op") or "") for a in birlestirilmis_adimlar)

    atilan_nedenleri: Counter[str] = Counter()
    if sahte_gecmis_sayisi:
        atilan_nedenleri["sahte_gecmis"] += sahte_gecmis_sayisi
    if kaydirma_atilan:
        atilan_nedenleri["kaydirma_birlestirme"] += kaydirma_atilan

    haric_kurallari = haric_listesi if haric_listesi is not None else haric_tutulanlari_yukle()
    on_elenenler = kaydirma_kosularini_siniflandir(birlestirilmis_adimlar, haric_kurallari, oturum_adi)

    has_explicit_go_back = any(str(a.get("op") or "") == "GO_BACK" for a in temiz_adimlar)

    # Pre-calculate fingerprints for page_changed
    fingerprints: list[str | None] = []
    for adim in birlestirilmis_adimlar:
        st = adim.get("state")
        if isinstance(st, dict) and not st.get("snapshot_error"):
            p = copy.deepcopy(st)
            try:
                fingerprints.append(fingerprint(p))
            except Exception:
                fingerprints.append(None)
        else:
            fingerprints.append(None)

    ctx = DualTierContext(primary_goal=goal)
    compiled_goal = ctx.compile_system1_goal("")
    done_rejected_url: tuple[str, int] | None = None

    history: list[dict[str, Any]] = []
    cikti_kayitlar: list[dict[str, Any]] = []
    tutulan_op: Counter[str] = Counter()
    visited_urls: list[str] = []

    for idx, adim in enumerate(birlestirilmis_adimlar):
        cur_fp = fingerprints[idx]
        next_fp = fingerprints[idx + 1] if idx + 1 < len(fingerprints) else None
        page_changed = bool(cur_fp and next_fp and cur_fp != next_fp)

        op = str(adim.get("op") or "")
        st = adim.get("state")
        raw_url = str(st.get("url", "")) if isinstance(st, dict) else ""
        cur_url = canonical_url(raw_url)

        # Legacy URL return detection
        if not has_explicit_go_back and cur_url:
            prev_adim = birlestirilmis_adimlar[idx - 1] if idx > 0 else None
            prev_st = prev_adim.get("state") if isinstance(prev_adim, dict) else None
            prev_raw_url = str(prev_st.get("url", "")) if isinstance(prev_st, dict) else ""
            prev_url = canonical_url(prev_raw_url)

            if prev_url and cur_url != prev_url and cur_url in visited_urls[:-1]:
                was_link_click = False
                if prev_adim and str(prev_adim.get("op") or "") == "CLICK":
                    prev_node = prev_adim.get("node")
                    prev_actions = (prev_st.get("actions") or []) if isinstance(prev_st, dict) else []
                    for a in prev_actions:
                        if a.get("node") == prev_node and a.get("href"):
                            target_url = canonical_url(urljoin(prev_raw_url, str(a["href"])))
                            if target_url == cur_url:
                                was_link_click = True
                                break
                if not was_link_click:
                    history.append({
                        "action": "Go back",
                        "kind": "go_back",
                        "text": None,
                        "page_changed": True,
                    })

        if cur_url:
            visited_urls.append(cur_url)

        # Check pre-classified discard (D2 & D3)
        if idx in on_elenenler:
            neden = on_elenenler[idx]
            atilan_nedenleri[neden] += 1
            if op != "DONE":
                history.append(gecmis_kaydi_kur(adim, None, page_changed))
            continue

        # Check invalid state
        if not isinstance(st, dict) or st.get("snapshot_error"):
            atilan_nedenleri["gecersiz_state"] += 1
            continue

        # Non-S1 operations (D1)
        if op in {"GO_BACK", "GO_FORWARD", "RELOAD"}:
            atilan_nedenleri["s1_disi_op"] += 1
            history.append(gecmis_kaydi_kur(adim, None, page_changed))
            continue

        if op == "UNMAPPED_CLICK":
            atilan_nedenleri["unmapped_click"] += 1
            history.append(gecmis_kaydi_kur(adim, None, page_changed))
            continue

        page = copy.deepcopy(st)
        page["fingerprint"] = cur_fp or fingerprint(page)

        cur_url_str = str(page.get("url", ""))
        cur_norm_url = cur_url_str.split("#")[0].rstrip("/")
        suppress_done = (
            done_rejected_url is not None
            and done_rejected_url[0] == cur_norm_url
            and (len(history) - done_rejected_url[1]) <= 8
        )

        istek, targets, controls = build_system1_request(
            page,
            compiled_goal,
            history,
            cfg,
            suppress_done=suppress_done,
        )
        op_kriter = istek["questions"]["operation"]["criteria"]
        altin_op, altin_hedef, eslesen_aksiyon, neden = altin_eslestir(
            adim, targets, controls, operation_criteria=op_kriter
        )
        if neden is not None or altin_op is None:
            atilan_nedenleri[neden or "eslesme_yok"] += 1
            if op != "DONE":
                history.append(gecmis_kaydi_kur(adim, eslesen_aksiyon, page_changed))
            continue

        cevaplar: dict[str, Any] = {
            "operation": {
                "choice": altin_op,
                "probabilities": olasilik_dagilimi(op_kriter, altin_op),
            }
        }
        if altin_op in HEDEF_OPLAR:
            hedef_anahtar = f"{altin_op.lower()}_target"
            hedef_kriter = istek["questions"][hedef_anahtar]["criteria"]
            if altin_hedef is None or altin_hedef not in hedef_kriter:
                atilan_nedenleri["oge_s1_listesinde_degil"] += 1
                if op != "DONE":
                    history.append(gecmis_kaydi_kur(adim, eslesen_aksiyon, page_changed))
                continue
            cevaplar[hedef_anahtar] = {
                "choice": altin_hedef,
                "probabilities": olasilik_dagilimi(hedef_kriter, altin_hedef),
            }

        kayit = {
            "istek": istek,
            "yanit": {"answers": cevaplar},
            "kaynak": "insan_demo",
            "oturum": oturum_adi,
            "adim": adim.get("step"),
            "insan": {
                "op": adim.get("op"),
                "node": adim.get("node"),
                "label": adim.get("label"),
                "text": adim.get("text"),
                "option": adim.get("option"),
            },
        }
        kayit_dogrula(kayit)
        cikti_kayitlar.append(kayit)
        tutulan_op[altin_op] += 1

        if altin_op != "DONE":
            history.append(gecmis_kaydi_kur(adim, eslesen_aksiyon, page_changed))

    toplam_atilan = sum(atilan_nedenleri.values())
    rapor = {
        "oturum": oturum_adi,
        "hedef": goal,
        "success": success,
        "dahil_edildi": True,
        "ham_adim": len(ham_adimlar),
        "birlestirme_sonrasi_adim": len(birlestirilmis_adimlar),
        "tutulan_adim": len(cikti_kayitlar),
        "atilan_adim": toplam_atilan,
        "type_text_enter_sayisi": type_text_enter_sayisi,
        "unmapped_click_sayisi": unmapped_click_sayisi,
        "sahte_gecmis": sahte_gecmis_sayisi,
        "atilan_nedenleri": dict(atilan_nedenleri),
        "birlestirme_oncesi_op": dict(birlestirme_oncesi_op),
        "birlestirme_oncesi_op_yuzde": op_dagilimi_yuzde(dict(birlestirme_oncesi_op)),
        "birlestirme_sonrasi_op": dict(birlestirme_sonrasi_op),
        "birlestirme_sonrasi_op_yuzde": op_dagilimi_yuzde(dict(birlestirme_sonrasi_op)),
        "tutulan_op": dict(tutulan_op),
        "tutulan_op_yuzde": op_dagilimi_yuzde(dict(tutulan_op)),
    }
    return cikti_kayitlar, rapor


def oturum_dosyasi_donustur(
    dosya: Path,
    *,
    basarisizlari_dahil_et: bool = False,
    config: ReflexConfig | None = None,
    haric_listesi: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Salt okunur olarak bir .jsonl oturum dosyasını oku ve dönüştür."""
    with dosya.open("r", encoding="utf-8") as f:
        satirlar = [json.loads(line) for line in f if line.strip()]
    return oturum_satirlarini_donustur(
        satirlar,
        dosya.name,
        basarisizlari_dahil_et=basarisizlari_dahil_et,
        config=config,
        haric_listesi=haric_listesi,
    )


def klasor_donustur(
    girdi_dir: Path,
    cikti_dir: Path,
    *,
    basarisizlari_dahil_et: bool = False,
    config: ReflexConfig | None = None,
    haric_yolu: Path | None = None,
) -> dict[str, Any]:
    """Girdi klasöründeki tüm oturumları dönüştür, insan_demo_kayit.jsonl ve rapor.json dosyalarını yaz."""
    cfg = config or ReflexConfig()
    dosyalar = sorted(girdi_dir.glob("*.jsonl")) if girdi_dir.is_dir() else []

    tum_kayitlar: list[dict[str, Any]] = []
    oturum_raporlari: list[dict[str, Any]] = []

    toplam_ham_adim = 0
    toplam_birlestirme_sonrasi = 0
    toplam_tutulan = 0
    toplam_atilan = 0
    toplam_type_text_enter = 0
    toplam_unmapped_click = 0
    toplam_sahte_gecmis = 0
    toplam_atilan_nedenleri: Counter[str] = Counter()
    toplam_oncesi_op: Counter[str] = Counter()
    toplam_sonrasi_op: Counter[str] = Counter()
    toplam_tutulan_op: Counter[str] = Counter()
    islenen_oturum = 0
    atlanan_basarisiz_oturum = 0

    haric_listesi = haric_tutulanlari_yukle(haric_yolu)

    for dosya in dosyalar:
        kayitlar, oturum_rapor = oturum_dosyasi_donustur(
            dosya,
            basarisizlari_dahil_et=basarisizlari_dahil_et,
            config=cfg,
            haric_listesi=haric_listesi,
        )
        oturum_raporlari.append(oturum_rapor)
        if not oturum_rapor["dahil_edildi"]:
            atlanan_basarisiz_oturum += 1
            continue

        islenen_oturum += 1
        tum_kayitlar.extend(kayitlar)
        toplam_ham_adim += int(oturum_rapor["ham_adim"])
        toplam_birlestirme_sonrasi += int(oturum_rapor["birlestirme_sonrasi_adim"])
        toplam_tutulan += int(oturum_rapor["tutulan_adim"])
        toplam_atilan += int(oturum_rapor["atilan_adim"])
        toplam_type_text_enter += int(oturum_rapor.get("type_text_enter_sayisi", 0))
        toplam_unmapped_click += int(oturum_rapor.get("unmapped_click_sayisi", 0))
        toplam_sahte_gecmis += int(oturum_rapor.get("sahte_gecmis", 0))
        toplam_atilan_nedenleri.update(oturum_rapor["atilan_nedenleri"])
        toplam_oncesi_op.update(oturum_rapor["birlestirme_oncesi_op"])
        toplam_sonrasi_op.update(oturum_rapor["birlestirme_sonrasi_op"])
        toplam_tutulan_op.update(oturum_rapor["tutulan_op"])

    toplam_ozet = {
        "toplam_oturum": len(dosyalar),
        "islenen_oturum": islenen_oturum,
        "atlanan_basarisiz_oturum": atlanan_basarisiz_oturum,
        "ham_adim": toplam_ham_adim,
        "birlestirme_sonrasi_adim": toplam_birlestirme_sonrasi,
        "tutulan_adim": toplam_tutulan,
        "atilan_adim": toplam_atilan,
        "type_text_enter_sayisi": toplam_type_text_enter,
        "unmapped_click_sayisi": toplam_unmapped_click,
        "sahte_gecmis": toplam_sahte_gecmis,
        "atilan_nedenleri": dict(toplam_atilan_nedenleri),
        "birlestirme_oncesi_op": dict(toplam_oncesi_op),
        "birlestirme_oncesi_op_yuzde": op_dagilimi_yuzde(dict(toplam_oncesi_op)),
        "birlestirme_sonrasi_op": dict(toplam_sonrasi_op),
        "birlestirme_sonrasi_op_yuzde": op_dagilimi_yuzde(dict(toplam_sonrasi_op)),
        "tutulan_op": dict(toplam_tutulan_op),
        "tutulan_op_yuzde": op_dagilimi_yuzde(dict(toplam_tutulan_op)),
    }

    rapor = {
        **toplam_ozet,
        "toplam": toplam_ozet,
        "oturumlar": oturum_raporlari,
    }

    cikti_dir.mkdir(parents=True, exist_ok=True)
    kayit_yolu = cikti_dir / "insan_demo_kayit.jsonl"
    rapor_yolu = cikti_dir / "rapor.json"

    with kayit_yolu.open("w", encoding="utf-8") as f:
        for kayit in tum_kayitlar:
            f.write(json.dumps(kayit, ensure_ascii=False) + "\n")

    rapor_yolu.write_text(
        json.dumps(rapor, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rapor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="İnsan gösterim kayıtlarını System 1 eğitim kaydına çevirir."
    )
    parser.add_argument(
        "--girdi",
        type=Path,
        default=DEFAULT_IN_DIR,
        help="Girdi demo kayıt klasörü (varsayılan: artifacts/demolar)",
    )
    parser.add_argument(
        "--cikti",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Çıktı klasörü (varsayılan: artifacts/demolar_cikti)",
    )
    parser.add_argument(
        "--haric",
        type=Path,
        default=None,
        help="haric.json dosya yolu (varsayılan: artifacts/demolar/haric.json)",
    )
    parser.add_argument(
        "--basarisizlari-dahil-et",
        action="store_true",
        help="success=false olan oturumları da dönüştürmeye dahil et",
    )
    args = parser.parse_args(argv)

    rapor = klasor_donustur(
        args.girdi,
        args.cikti,
        basarisizlari_dahil_et=args.basarisizlari_dahil_et,
        haric_yolu=args.haric,
    )
    print(
        f"Oturum: {rapor['islenen_oturum']}/{rapor['toplam_oturum']} | "
        f"Ham adım: {rapor['ham_adim']} -> Birleştirme sonrası: {rapor['birlestirme_sonrasi_adim']} -> "
        f"Tutulan: {rapor['tutulan_adim']} (Atılan: {rapor['atilan_adim']}: {rapor['atilan_nedenleri']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
