"""Unit tests for System 1 request builder refactoring and human demo converter (scripts/kayit/donustur.py).

Uses strictly synthetic recordings (no real user demonstration data).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from reflex.browser.cdp import fingerprint
from reflex.config import ReflexConfig
from reflex.context import DualTierContext
from reflex.system1 import build_system1_request, choose_action
from scripts.kayit.donustur import (
    kaydirma_birlestir,
    kayit_dogrula,
    klasor_donustur,
    main,
    olasilik_dagilimi,
    oturum_satirlarini_donustur,
)


def _sentetik_state(
    *,
    url: str = "https://ornek.test/sayfa",
    title: str = "Örnek Sayfa",
    text: str = "Birim testi için sentetik sayfa içeriği.",
    scroll_y: int = 0,
    scroll_h: int = 2400,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if actions is None:
        actions = [
            {"id": "e1", "node": 10, "kind": "fill", "role": "searchbox", "label": "Arama Kutusu", "value": ""},
            {"id": "e2", "node": 11, "kind": "click", "role": "button", "label": "Ara Butonu", "value": ""},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
        ]
    return {
        "url": url,
        "title": title,
        "text": text,
        "full_text": text,
        "headings": ["[h1] Örnek Başlık"],
        "scroll": {"y": scroll_y, "height": scroll_h},
        "actions": actions,
    }


def _sentetik_oturum(
    adimlar: list[dict[str, Any]],
    *,
    goal: str = "Sentetik test görevini tamamla",
    success: bool = True,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "session",
            "goal": goal,
            "start_url": "https://ornek.test",
            "started": 1700000000.0,
            "viewport": {"width": 1120, "height": 780},
        },
        *adimlar,
        {
            "type": "session_end",
            "success": success,
            "note": None,
            "steps": len(adimlar),
            "duration_s": 12.5,
        },
    ]


def test_choose_action_http_body_equals_build_system1_request():
    """Madde 4: Aynı girdide choose_action'ın HTTP'ye gönderdiği gövde == build_system1_request çıktısı."""
    raw_state = _sentetik_state(
        actions=[
            {"id": "e1", "node": 10, "kind": "fill", "role": "textbox", "label": "Kullanıcı Adı", "value": ""},
            {"id": "e2", "node": 11, "kind": "click", "role": "button", "label": "Giriş Yap", "value": ""},
            {
                "id": "e3",
                "node": 12,
                "kind": "select",
                "role": "combobox",
                "label": "Dil → Türkçe",
                "value": "tr",
                "current_value": "Seçiniz",
            },
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ]
    )
    page = {**raw_state, "fingerprint": fingerprint(raw_state)}
    compiled_goal = DualTierContext(primary_goal="Sentetik forma giriş yap").compile_system1_goal("")
    history = [
        {"action": "Çerezleri Kabul Et", "kind": "click", "text": None, "page_changed": True},
        {"action": "Scroll down", "kind": "scroll", "text": None, "page_changed": True},
    ]
    config = ReflexConfig(decision_backend="local_turnike", jev_model="~typesafe/jev-latest")

    beklenen_govde, beklenen_targets, beklenen_controls = build_system1_request(
        page,
        compiled_goal,
        history,
        config,
        suppress_done=False,
    )
    assert "CLICK" in beklenen_targets
    assert "TYPE_TEXT" in beklenen_targets
    assert "SELECT" in beklenen_targets
    assert "SCROLL_DOWN" in beklenen_controls

    sahte_yanit = MagicMock()
    sahte_yanit.status_code = 200
    sahte_yanit.json.return_value = {
        "model": config.jev_model,
        "answers": {
            "operation": {
                "choice": "CLICK",
                "probabilities": {
                    "CLICK": 0.9,
                    "TYPE_TEXT": 0.02,
                    "SELECT": 0.02,
                    "SCROLL_DOWN": 0.02,
                    "WAIT": 0.02,
                    "DONE": 0.01,
                    "BLOCKED": 0.01,
                },
            },
            "click_target": {
                "choice": "2",
                "probabilities": {"2": 1.0},
            },
        },
        "usage": {"prompt_tokens": 100, "completion_tokens": 5},
    }

    with patch("reflex.system1._HTTP_CLIENT.post", return_value=sahte_yanit) as mock_post:
        sonuc = choose_action(page, compiled_goal, history, config, suppress_done=False)

    mock_post.assert_called_once()
    gonderilen_govde = mock_post.call_args.kwargs["json"]
    assert gonderilen_govde == beklenen_govde
    assert sonuc["operation"] == "CLICK"
    assert sonuc["target"] == "2"
    assert sonuc["choice"] == "e2"


def test_kaydirma_birlestirme_315px_ve_yon_degisimi():
    """Madde 2: Ardışık aynı yönlü kaydırmalarda ilkini ve son TUTULAN adımdan >=315px uzak olanları tut."""
    adimlar = [
        {"type": "step", "step": 1, "op": "SCROLL_DOWN", "scroll_y": 0, "state": _sentetik_state(scroll_y=0)},
        {"type": "step", "step": 2, "op": "SCROLL_DOWN", "scroll_y": 150, "state": _sentetik_state(scroll_y=150)},
        {"type": "step", "step": 3, "op": "SCROLL_DOWN", "scroll_y": 314, "state": _sentetik_state(scroll_y=314)},
        # 315 - 0 = 315 >= 315 -> TUTULUR (son tutulan scroll_y = 315 olur)
        {"type": "step", "step": 4, "op": "SCROLL_DOWN", "scroll_y": 315, "state": _sentetik_state(scroll_y=315)},
        # 500 - 315 = 185 < 315 -> ATILIR (ilk adımdan >315 olsa bile son tutulandan <315)
        {"type": "step", "step": 5, "op": "SCROLL_DOWN", "scroll_y": 500, "state": _sentetik_state(scroll_y=500)},
        # 630 - 315 = 315 >= 315 -> TUTULUR (son tutulan scroll_y = 630 olur)
        {"type": "step", "step": 6, "op": "SCROLL_DOWN", "scroll_y": 630, "state": _sentetik_state(scroll_y=630)},
        # Yön değişti (SCROLL_UP) -> ilk adım olduğu için TUTULUR (son tutulan = 700)
        {"type": "step", "step": 7, "op": "SCROLL_UP", "scroll_y": 700, "state": _sentetik_state(scroll_y=700)},
        # |500 - 700| = 200 < 315 -> ATILIR
        {"type": "step", "step": 8, "op": "SCROLL_UP", "scroll_y": 500, "state": _sentetik_state(scroll_y=500)},
        # |350 - 700| = 350 >= 315 -> TUTULUR
        {"type": "step", "step": 9, "op": "SCROLL_UP", "scroll_y": 350, "state": _sentetik_state(scroll_y=350)},
        # Araya CLICK girdi
        {"type": "step", "step": 10, "op": "CLICK", "node": 11, "label": "Ara Butonu", "state": _sentetik_state(scroll_y=350)},
        # CLICK sonrası yeni SCROLL_DOWN serisi: mesafe küçük olsa bile ilk adım olduğu için TUTULUR
        {"type": "step", "step": 11, "op": "SCROLL_DOWN", "scroll_y": 360, "state": _sentetik_state(scroll_y=360)},
    ]

    tutulan, atilan = kaydirma_birlestir(adimlar)
    assert atilan == 4
    assert [a["step"] for a in tutulan] == [1, 4, 6, 7, 9, 10, 11]


def test_select_esleme_ve_olasiliklar():
    """Madde 5 & 6: SELECT adımında node ve value eşleşen seçenek altın etiket olur."""
    select_actions = [
        {"id": "e1", "node": 5, "kind": "click", "role": "button", "label": "İptal"},
        {
            "id": "e2",
            "node": 20,
            "kind": "select",
            "role": "combobox",
            "label": "Kategori → Kitap",
            "value": "book",
            "current_value": "Seçiniz",
        },
        {
            "id": "e3",
            "node": 20,
            "kind": "select",
            "role": "combobox",
            "label": "Kategori → Elektronik",
            "value": "elec",
            "current_value": "Seçiniz",
        },
        {
            "id": "e4",
            "node": 21,
            "kind": "select",
            "role": "combobox",
            "label": "Şehir → Ankara",
            "value": "elec",
            "current_value": "Seçiniz",
        },
    ]
    adimlar = [
        {
            "type": "step",
            "step": 1,
            "op": "SELECT",
            "node": 20,
            "label": "Kategori",
            "value": "elec",
            "option": "Elektronik",
            "state": _sentetik_state(actions=select_actions),
        },
        {
            "type": "step",
            "step": 2,
            "op": "DONE",
            "state": _sentetik_state(text="Seçim tamamlandı", actions=select_actions),
        },
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_select.jsonl")
    assert len(kayitlar) == 2
    assert rapor["atilan_adim"] == 0

    sec_kayit = kayitlar[0]
    kayit_dogrula(sec_kayit)
    cevaplar = sec_kayit["yanit"]["answers"]
    assert set(cevaplar.keys()) == {"operation", "select_target"}
    assert cevaplar["operation"]["choice"] == "SELECT"
    # node=20 element index "2", "elec" onun 2. seçeneği -> "2:2"
    assert cevaplar["select_target"]["choice"] == "2:2"
    assert abs(cevaplar["select_target"]["probabilities"]["2:2"] - 0.9) < 1e-6
    assert abs(cevaplar["select_target"]["probabilities"]["2:1"] - 0.05) < 1e-6
    assert abs(cevaplar["select_target"]["probabilities"]["3:1"] - 0.05) < 1e-6

    done_kayit = kayitlar[1]
    kayit_dogrula(done_kayit)
    assert set(done_kayit["yanit"]["answers"].keys()) == {"operation"}
    assert done_kayit["yanit"]["answers"]["operation"]["choice"] == "DONE"


def test_eslesmeyen_adimlarin_atilmasi_ve_rapor_nedenleri():
    """Madde 5: S1 listesinde olmayan node, işaretli radio, 60 sınırı ve kontrol yok adımları atılır."""
    # 1) İşaretli radio (checked="true") -> CLICK hedeflerinden filtrelenir
    radio_actions = [
        {"id": "e1", "node": 1, "kind": "click", "role": "radio", "label": "Katılım → Hayır", "checked": "true"},
        {"id": "e2", "node": 2, "kind": "click", "role": "radio", "label": "Katılım → Evet", "checked": "false"},
    ]
    # 2) 65 öğeli sayfa -> 61..65 arası 60 sınırına takılır
    cok_oge_actions = [
        {"id": f"e{i}", "node": i, "kind": "click", "role": "button", "label": f"Buton {i}"}
        for i in range(1, 66)
    ]
    # 3) scroll_up kontrolü olmayan sayfa (scrollY=0)
    kontrolsuz_actions = [
        {"id": "e1", "node": 10, "kind": "click", "role": "button", "label": "Tamam"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]

    adimlar = [
        # Adım 1: node=999 listede yok -> oge_s1_listesinde_degil
        {
            "type": "step",
            "step": 1,
            "op": "CLICK",
            "node": 999,
            "label": "Hayalet Buton",
            "state": _sentetik_state(actions=radio_actions),
        },
        # Adım 2: node=1 checked="true" radio -> isaretli_radio_filtresi
        {
            "type": "step",
            "step": 2,
            "op": "CLICK",
            "node": 1,
            "label": "Katılım → Hayır",
            "state": _sentetik_state(actions=radio_actions),
        },
        # Adım 3: node=64 (>60) -> 60_siniri
        {
            "type": "step",
            "step": 3,
            "op": "CLICK",
            "node": 64,
            "label": "Buton 64",
            "state": _sentetik_state(actions=cok_oge_actions),
        },
        # Adım 4: SCROLL_UP ama scroll_up kontrolü yok -> kontrol_yok
        {
            "type": "step",
            "step": 4,
            "op": "SCROLL_UP",
            "scroll_y": 0,
            "state": _sentetik_state(actions=kontrolsuz_actions),
        },
        # Adım 5: Geçerli CLICK (node=2, checked="false") -> TUTULUR (tek seçenekli click_target -> olasılık 1.0)
        {
            "type": "step",
            "step": 5,
            "op": "CLICK",
            "node": 2,
            "label": "Katılım → Evet",
            "state": _sentetik_state(actions=radio_actions),
        },
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_atilan.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 5
    # Tek seçenekli click_target sorusunda olasılık 1.0 olmalı
    assert kayitlar[0]["yanit"]["answers"]["click_target"]["probabilities"] == {"2": 1.0}
    assert rapor["tutulan_adim"] == 1
    assert rapor["atilan_adim"] == 4
    assert rapor["atilan_nedenleri"] == {
        "oge_s1_listesinde_degil": 1,
        "isaretli_radio_filtresi": 1,
        "60_siniri": 1,
        "kontrol_yok": 1,
    }


def test_history_ve_page_changed_kurulumu(tmp_path: Path):
    """Madde 3: history önceki TUTULAN adımlardan kurulur; page_changed sonraki tutulan adımın fingerprint farkıdır."""
    state_1 = _sentetik_state(url="https://ornek.test/1", text="Sayfa 1 başlangıç", scroll_y=0)
    scroll_initial_actions = [
        {"id": "e1", "node": 10, "kind": "fill", "role": "searchbox", "label": "Arama Kutusu", "value": ""},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    state_2_scroll_a = _sentetik_state(url="https://ornek.test/1", text="Sayfa 1 arama yazıldı", scroll_y=0, actions=scroll_initial_actions)
    state_2_scroll_b = _sentetik_state(url="https://ornek.test/1", text="Sayfa 1 biraz kaydı", scroll_y=180, actions=scroll_initial_actions)
    state_3_ayni = _sentetik_state(url="https://ornek.test/1", text="Sayfa 1 aşağı kaydı", scroll_y=450)
    state_4_degismedi = copy.deepcopy(state_3_ayni)
    state_5_son = _sentetik_state(url="https://ornek.test/2", text="Sayfa 2 sonuçlar", scroll_y=0)

    adimlar = [
        # Adım 1 (TUTULUR): TYPE_TEXT (autofocus gibi label=None verildiğinde aksiyon etiketinden tamamlanır)
        # Sonraki tutulan adım (Adım 2) farklı fingerprint'e sahip -> page_changed=True
        {
            "type": "step",
            "step": 1,
            "op": "TYPE_TEXT",
            "node": 10,
            "label": None,
            "text": "test sorgusu",
            "state": state_1,
        },
        # Adım 2 (TUTULUR): SCROLL_DOWN (y=0). Sonraki tutulan adım (Adım 4, y=450) farklı fingerprint -> page_changed=True
        {
            "type": "step",
            "step": 2,
            "op": "SCROLL_DOWN",
            "scroll_y": 0,
            "state": state_2_scroll_a,
        },
        # Adım 3 (ATILIR - kaydırma birleştirme, 180 - 0 < 315): history'ye GİRMEMELİ
        {
            "type": "step",
            "step": 3,
            "op": "SCROLL_DOWN",
            "scroll_y": 180,
            "state": state_2_scroll_b,
        },
        # Adım 4 (TUTULUR): CLICK ama sonraki tutulan adım (Adım 5) ile aynı fingerprint -> page_changed=False
        {
            "type": "step",
            "step": 4,
            "op": "CLICK",
            "node": 11,
            "label": "Ara Butonu",
            "state": state_3_ayni,
        },
        # Adım 5 (TUTULUR): CLICK ve sonraki tutulan adım (Adım 6 DONE) farklı fingerprint -> page_changed=True
        {
            "type": "step",
            "step": 5,
            "op": "CLICK",
            "node": 11,
            "label": "Ara Butonu",
            "state": state_4_degismedi,
        },
        # Adım 6 (TUTULUR): DONE
        {
            "type": "step",
            "step": 6,
            "op": "DONE",
            "state": state_5_son,
        },
    ]

    girdi_dir = tmp_path / "girdi"
    cikti_dir = tmp_path / "cikti"
    girdi_dir.mkdir()

    # Başarılı oturum
    basarili_dosya = girdi_dir / "01_basarili.jsonl"
    basarili_dosya.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _sentetik_oturum(adimlar, success=True)) + "\n",
        encoding="utf-8",
    )
    # Başarısız oturum (varsayılan olarak atlanmalı)
    basarisiz_dosya = girdi_dir / "02_basarisiz.jsonl"
    basarisiz_dosya.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _sentetik_oturum(adimlar[:2], success=False)) + "\n",
        encoding="utf-8",
    )

    rapor = klasor_donustur(girdi_dir, cikti_dir, basarisizlari_dahil_et=False)
    assert rapor["toplam_oturum"] == 2
    assert rapor["islenen_oturum"] == 1
    assert rapor["atlanan_basarisiz_oturum"] == 1
    assert rapor["tutulan_adim"] == 5

    cikti_satirlar = [
        json.loads(line)
        for line in (cikti_dir / "insan_demo_kayit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(cikti_satirlar) == 5
    for k in cikti_satirlar:
        kayit_dogrula(k)
        assert k["kaynak"] == "insan_demo"
        assert k["oturum"] == "01_basarili.jsonl"
        assert set(k["insan"].keys()) == {"op", "node", "label", "text", "option"}

    # Son adımın (Adım 6 DONE) recent_actions geçmişini kontrol et
    son_gecmis = cikti_satirlar[-1]["istek"]["state"]["recent_actions"]
    assert son_gecmis == [
        {"action": "Arama Kutusu", "kind": "fill", "text": "test sorgusu", "page_changed": True},
        {"action": "Scroll down", "kind": "scroll", "text": None, "page_changed": True},
        {"action": "Ara Butonu", "kind": "click", "text": None, "page_changed": False},
        {"action": "Ara Butonu", "kind": "click", "text": None, "page_changed": True},
    ]


def test_olasilik_dagilimi_tek_ve_cok_secenek():
    """Madde 6: Tek seçenekte 1.0, çok seçenekte altın=0.9 ve kalanlar eşit bölünür."""
    assert olasilik_dagilimi(["1"], "1") == {"1": 1.0}
    dag = olasilik_dagilimi(["CLICK", "TYPE_TEXT", "SCROLL_DOWN", "DONE", "BLOCKED"], "CLICK")
    assert dag["CLICK"] == 0.9
    for k in ("TYPE_TEXT", "SCROLL_DOWN", "DONE", "BLOCKED"):
        assert abs(dag[k] - 0.025) < 1e-8
    assert abs(sum(dag.values()) - 1.0) <= 1e-6


def test_type_text_matching_and_target_probabilities():
    """Multiple text input elements: gold gets 0.9, remaining 0.1 split equally, sum == 1.0."""
    text_actions = [
        {"id": "e1", "node": 10, "kind": "fill", "role": "textbox", "label": "Ad", "value": ""},
        {"id": "e2", "node": 11, "kind": "fill", "role": "textbox", "label": "Soyad", "value": ""},
        {"id": "e3", "node": 12, "kind": "fill", "role": "textbox", "label": "E-posta", "value": ""},
        {"id": "btn", "node": 13, "kind": "click", "role": "button", "label": "Kaydet"},
    ]
    adimlar = [
        {
            "type": "step",
            "step": 1,
            "op": "TYPE_TEXT",
            "node": 11,
            "label": "Soyad",
            "text": "Demir",
            "state": _sentetik_state(actions=text_actions),
        },
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_type_text.jsonl")
    assert len(kayitlar) == 1
    assert rapor["tutulan_adim"] == 1
    assert rapor["atilan_adim"] == 0

    kayit = kayitlar[0]
    kayit_dogrula(kayit)
    answers = kayit["yanit"]["answers"]
    assert set(answers.keys()) == {"operation", "type_text_target"}
    assert answers["operation"]["choice"] == "TYPE_TEXT"

    # node=11 corresponds to element index "2" among targets
    target_ans = answers["type_text_target"]
    assert target_ans["choice"] == "2"
    probs = target_ans["probabilities"]
    assert len(probs) == 3
    assert set(probs.keys()) == {"1", "2", "3"}
    assert abs(probs["2"] - 0.9) < 1e-6
    assert abs(probs["1"] - 0.05) < 1e-6
    assert abs(probs["3"] - 0.05) < 1e-6
    assert abs(sum(probs.values()) - 1.0) <= 1e-6


def test_scroll_answers_have_no_target_question():
    """Verify yanit['answers'] contains ONLY 'operation' and no target question for scroll."""
    scroll_actions = [
        {"id": "e1", "node": 1, "kind": "click", "role": "button", "label": "Detaylar"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
        {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -420},
    ]
    revealed_actions = [
        {"id": "e1", "node": 1, "kind": "click", "role": "button", "label": "Detaylar"},
        {"id": "e2", "node": 2, "kind": "click", "role": "button", "label": "Yeni Buton"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
        {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -420},
    ]
    adimlar = [
        {
            "type": "step",
            "step": 1,
            "op": "SCROLL_DOWN",
            "scroll_y": 0,
            "state": _sentetik_state(scroll_y=0, actions=scroll_actions),
        },
        {
            "type": "step",
            "step": 2,
            "op": "SCROLL_UP",
            "scroll_y": 420,
            "state": _sentetik_state(scroll_y=420, actions=scroll_actions),
        },
        {
            "type": "step",
            "step": 3,
            "op": "CLICK",
            "node": 2,
            "label": "Yeni Buton",
            "state": _sentetik_state(scroll_y=420, actions=revealed_actions),
        },
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_scroll_answers.jsonl")
    assert len(kayitlar) == 3
    assert rapor["tutulan_adim"] == 3

    for k in kayitlar[:2]:
        kayit_dogrula(k)
        answers = k["yanit"]["answers"]
        assert list(answers.keys()) == ["operation"]
        assert "scroll_target" not in answers
        assert "scroll_down_target" not in answers
        assert "scroll_up_target" not in answers
        assert not any("target" in key for key in answers)
        assert abs(sum(answers["operation"]["probabilities"].values()) - 1.0) <= 1e-6

    assert kayitlar[0]["yanit"]["answers"]["operation"]["choice"] == "SCROLL_DOWN"
    assert kayitlar[1]["yanit"]["answers"]["operation"]["choice"] == "SCROLL_UP"


def test_basarisizlari_dahil_et_flag(tmp_path: Path):
    """Verify sessions with success=False are included when flag is True and skipped when False."""
    adimlar = [
        {
            "type": "step",
            "step": 1,
            "op": "CLICK",
            "node": 11,
            "label": "Ara Butonu",
            "state": _sentetik_state(),
        },
    ]
    oturum = _sentetik_oturum(adimlar, success=False)

    # 1. oturum_satirlarini_donustur testing
    kayitlar_skip, rapor_skip = oturum_satirlarini_donustur(
        oturum, "test_skip.jsonl", basarisizlari_dahil_et=False
    )
    assert len(kayitlar_skip) == 0
    assert rapor_skip["dahil_edildi"] is False
    assert rapor_skip["tutulan_adim"] == 0
    assert rapor_skip["atilan_adim"] == 1
    assert rapor_skip["atilan_nedenleri"] == {"basarisiz_oturum": 1}

    kayitlar_inc, rapor_inc = oturum_satirlarini_donustur(
        oturum, "test_inc.jsonl", basarisizlari_dahil_et=True
    )
    assert len(kayitlar_inc) == 1
    assert rapor_inc["dahil_edildi"] is True
    assert rapor_inc["tutulan_adim"] == 1
    assert rapor_inc["atilan_adim"] == 0
    kayit_dogrula(kayitlar_inc[0])

    # 2. klasor_donustur testing
    girdi_dir = tmp_path / "girdi"
    cikti_dir_skip = tmp_path / "cikti_skip"
    cikti_dir_inc = tmp_path / "cikti_inc"
    girdi_dir.mkdir()

    fail_file = girdi_dir / "01_failed.jsonl"
    fail_file.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in oturum) + "\n",
        encoding="utf-8",
    )

    rapor_dir_skip = klasor_donustur(girdi_dir, cikti_dir_skip, basarisizlari_dahil_et=False)
    assert rapor_dir_skip["toplam_oturum"] == 1
    assert rapor_dir_skip["islenen_oturum"] == 0
    assert rapor_dir_skip["atlanan_basarisiz_oturum"] == 1
    assert rapor_dir_skip["tutulan_adim"] == 0
    assert (cikti_dir_skip / "insan_demo_kayit.jsonl").read_text(encoding="utf-8").strip() == ""

    rapor_dir_inc = klasor_donustur(girdi_dir, cikti_dir_inc, basarisizlari_dahil_et=True)
    assert rapor_dir_inc["toplam_oturum"] == 1
    assert rapor_dir_inc["islenen_oturum"] == 1
    assert rapor_dir_inc["atlanan_basarisiz_oturum"] == 0
    assert rapor_dir_inc["tutulan_adim"] == 1
    inc_lines = [
        json.loads(line)
        for line in (cikti_dir_inc / "insan_demo_kayit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(inc_lines) == 1


def test_non_s1_operations_dropped():
    """Synthetic steps with GO_BACK, GO_FORWARD, RELOAD dropped with reason s1_disi_op."""
    adimlar = [
        {"type": "step", "step": 1, "time": 100.0, "op": "GO_BACK", "target_url": "https://ornek.test/1", "state": _sentetik_state()},
        {"type": "step", "step": 2, "time": 105.0, "op": "GO_FORWARD", "target_url": "https://ornek.test/2", "state": _sentetik_state()},
        {"type": "step", "step": 3, "time": 110.0, "op": "RELOAD", "target_url": "https://ornek.test/2", "state": _sentetik_state()},
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_non_s1.jsonl")
    assert len(kayitlar) == 0
    assert rapor["tutulan_adim"] == 0
    assert rapor["atilan_adim"] == 3
    assert rapor["atilan_nedenleri"] == {"s1_disi_op": 3}
    assert rapor["birlestirme_oncesi_op"] == {"GO_BACK": 1, "GO_FORWARD": 1, "RELOAD": 1}
    assert rapor["tutulan_op"] == {}


def test_select_option_not_found():
    """Synthetic step with SELECT where value is not in dropdown options, dropped with reason secenek_bulunamadi."""
    select_actions = [
        {
            "id": "e1",
            "node": 20,
            "kind": "select",
            "role": "combobox",
            "label": "Dil → Türkçe",
            "value": "tr",
            "current_value": "Seçiniz",
        },
        {
            "id": "e2",
            "node": 20,
            "kind": "select",
            "role": "combobox",
            "label": "Dil → İngilizce",
            "value": "en",
            "current_value": "Seçiniz",
        },
    ]
    adimlar = [
        {
            "type": "step",
            "step": 1,
            "op": "SELECT",
            "node": 20,
            "label": "Dil",
            "value": "de",  # "de" is not in options!
            "option": "Almanca",
            "state": _sentetik_state(actions=select_actions),
        },
        {
            "type": "step",
            "step": 2,
            "op": "SELECT",
            "node": 20,
            "label": "Dil",
            "value": "tr",  # Valid option!
            "option": "Türkçe",
            "state": _sentetik_state(actions=select_actions),
        },
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_select_missing.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 2
    kayit_dogrula(kayitlar[0])
    assert rapor["tutulan_adim"] == 1
    assert rapor["atilan_adim"] == 1
    assert rapor["atilan_nedenleri"] == {"secenek_bulunamadi": 1}


def test_invalid_state_and_snapshot_error():
    """Step with invalid state dropped with reason gecersiz_state."""
    adimlar = [
        # 1. state is None
        {"type": "step", "step": 1, "op": "CLICK", "node": 10, "state": None},
        # 2. state is not a dict
        {"type": "step", "step": 2, "op": "CLICK", "node": 10, "state": "gecersiz_string"},
        # 3. state has snapshot_error
        {
            "type": "step",
            "step": 3,
            "op": "CLICK",
            "node": 10,
            "state": {"snapshot_error": "CDP execution context was destroyed", "actions": []},
        },
        # 4. Valid step
        {"type": "step", "step": 4, "op": "DONE", "state": _sentetik_state()},
    ]

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_invalid_state.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 4
    kayit_dogrula(kayitlar[0])
    assert rapor["tutulan_adim"] == 1
    assert rapor["atilan_adim"] == 3
    assert rapor["atilan_nedenleri"] == {"gecersiz_state": 3}


def test_cli_main_entrypoint(tmp_path: Path):
    """Invoke main(['--girdi', ..., '--cikti', ...]), verifying exit code 0 and output files created."""
    girdi_dir = tmp_path / "cli_girdi"
    cikti_dir = tmp_path / "cli_cikti"
    girdi_dir.mkdir()

    session_data = _sentetik_oturum(
        [
            {
                "type": "step",
                "step": 1,
                "op": "CLICK",
                "node": 11,
                "label": "Ara Butonu",
                "state": _sentetik_state(),
            },
            {
                "type": "step",
                "step": 2,
                "op": "DONE",
                "state": _sentetik_state(text="Bitti"),
            },
        ],
        success=True,
    )
    (girdi_dir / "01_test.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in session_data) + "\n",
        encoding="utf-8",
    )

    exit_code = main(["--girdi", str(girdi_dir), "--cikti", str(cikti_dir)])
    assert exit_code == 0

    jsonl_path = cikti_dir / "insan_demo_kayit.jsonl"
    rapor_path = cikti_dir / "rapor.json"
    assert jsonl_path.exists()
    assert rapor_path.exists()

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    for r in records:
        kayit_dogrula(r)
        assert r["kaynak"] == "insan_demo"
        assert r["oturum"] == "01_test.jsonl"

    rapor_data = json.loads(rapor_path.read_text(encoding="utf-8"))
    assert rapor_data["toplam_oturum"] == 1
    assert rapor_data["islenen_oturum"] == 1
    assert rapor_data["atlanan_basarisiz_oturum"] == 0
    assert rapor_data["tutulan_adim"] == 2
    assert rapor_data["atilan_adim"] == 0
    assert "toplam" in rapor_data
    assert "oturumlar" in rapor_data
    assert len(rapor_data["oturumlar"]) == 1


# ==============================================================================
# D1, D2, D3, D4 Comprehensive Synthetic Test Suites
# ==============================================================================


def test_d1_non_label_actions_in_history():
    """D1: GO_BACK, GO_FORWARD, RELOAD, UNMAPPED_CLICK are added to history but produce no labels."""
    adimlar = [
        {"type": "step", "step": 1, "time": 10.0, "op": "CLICK", "node": 11, "label": "Ara Butonu", "state": _sentetik_state()},
        {"type": "step", "step": 2, "time": 20.0, "op": "GO_BACK", "target_url": "https://ornek.test/1", "state": _sentetik_state()},
        {"type": "step", "step": 3, "time": 30.0, "op": "GO_FORWARD", "target_url": "https://ornek.test/2", "state": _sentetik_state()},
        {"type": "step", "step": 4, "time": 40.0, "op": "RELOAD", "target_url": "https://ornek.test/2", "state": _sentetik_state()},
        {"type": "step", "step": 5, "time": 50.0, "op": "UNMAPPED_CLICK", "unmapped": {"text": "Açılır Menü Önerisi"}, "state": _sentetik_state()},
        {"type": "step", "step": 6, "time": 60.0, "op": "CLICK", "node": 11, "label": "Ara Butonu", "state": _sentetik_state()},
    ]
    kayitlar, _rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_history.jsonl")
    assert len(kayitlar) == 2  # Only step 1 and step 6 produce labels
    assert [k["adim"] for k in kayitlar] == [1, 6]

    # Verify recent_actions in step 6 contains all non-label actions
    gecmis = kayitlar[1]["istek"]["state"]["recent_actions"]
    assert len(gecmis) == 5
    assert gecmis[0]["kind"] == "click"
    assert gecmis[1] == {"action": "Go back", "kind": "go_back", "text": None, "page_changed": False}
    assert gecmis[2] == {"action": "Go forward", "kind": "go_forward", "text": None, "page_changed": False}
    assert gecmis[3] == {"action": "Reload", "kind": "reload", "text": None, "page_changed": False}
    assert gecmis[4] == {"action": "Açılır Menü Önerisi", "kind": "click", "text": None, "page_changed": False}


def test_d1_page_changed_based_on_next_real_step():
    """D1: page_changed is computed from the next real step fingerprint, even if next step is non-label."""
    state_a = _sentetik_state(url="https://ornek.test/a", text="Sayfa A")
    state_b = _sentetik_state(url="https://ornek.test/b", text="Sayfa B")
    adimlar = [
        {"type": "step", "step": 1, "time": 10.0, "op": "CLICK", "node": 11, "label": "Git", "state": state_a},
        {"type": "step", "step": 2, "time": 20.0, "op": "GO_BACK", "target_url": "https://ornek.test/a", "state": state_b},
        {"type": "step", "step": 3, "time": 30.0, "op": "DONE", "state": state_b},
    ]
    kayitlar, _ = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_page_changed.jsonl")
    assert len(kayitlar) == 2  # Step 1 and Step 3
    # Step 1 page_changed must be True because step 2 transitioned to state_b
    done_gecmis = kayitlar[1]["istek"]["state"]["recent_actions"]
    assert done_gecmis[0]["page_changed"] is True
    # Step 2 page_changed must be False because step 3 remained at state_b
    assert done_gecmis[1]["page_changed"] is False


def test_d1_synthetic_go_back_inserted_on_url_return():
    """D1: Insert synthetic go_back history item when returning to earlier visited URL without explicit GO_BACK."""
    state_url1 = _sentetik_state(url="https://ornek.test/sayfa1")
    state_url2 = _sentetik_state(url="https://ornek.test/sayfa2")
    adimlar = [
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "Link 2", "state": state_url1},
        {"type": "step", "step": 2, "op": "SCROLL_DOWN", "scroll_y": 0, "state": state_url2},
        # Step 3 returns to URL 1 without a link click in Step 2:
        {"type": "step", "step": 3, "op": "CLICK", "node": 11, "label": "Buton", "state": state_url1},
    ]
    kayitlar, _ = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_url_return.jsonl")
    # Step 3's history should contain synthetic go_back
    son_gecmis = kayitlar[-1]["istek"]["state"]["recent_actions"]
    assert any(h.get("kind") == "go_back" and h.get("action") == "Go back" and h.get("page_changed") is True for h in son_gecmis)
    # Synthetic go_back must NOT produce a training label
    assert not any(k.get("insan", {}).get("op") == "GO_BACK" for k in kayitlar)


def test_d1_legacy_url_return_skipped_on_link_click():
    """D1: When previous step was a link click to cur_url, synthetic go_back is NOT added."""
    state_url1 = _sentetik_state(url="https://ornek.test/sayfa1")
    link_actions = [
        {"id": "e1", "node": 50, "kind": "click", "role": "link", "label": "Ana Sayfa", "href": "https://ornek.test/sayfa1"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    state_url2 = _sentetik_state(url="https://ornek.test/sayfa2", actions=link_actions)
    adimlar = [
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "Sayfa 2", "state": state_url1},
        {"type": "step", "step": 2, "op": "CLICK", "node": 50, "label": "Ana Sayfa", "state": state_url2},
        {"type": "step", "step": 3, "op": "CLICK", "node": 11, "label": "Buton", "state": state_url1},
    ]
    kayitlar, _ = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_link_click.jsonl")
    son_gecmis = kayitlar[-1]["istek"]["state"]["recent_actions"]
    assert not any(h.get("kind") == "go_back" for h in son_gecmis)


def test_d1_legacy_url_return_skipped_when_go_back_present():
    """D1: When session already contains explicit GO_BACK, synthetic go_back is NOT added."""
    state_url1 = _sentetik_state(url="https://ornek.test/sayfa1")
    state_url2 = _sentetik_state(url="https://ornek.test/sayfa2")
    adimlar = [
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "Sayfa 2", "state": state_url1},
        {"type": "step", "step": 2, "op": "GO_BACK", "target_url": "https://ornek.test/sayfa1", "state": state_url2},
        {"type": "step", "step": 3, "op": "CLICK", "node": 11, "label": "Buton", "state": state_url1},
    ]
    kayitlar, _ = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_explicit_back.jsonl")
    son_gecmis = kayitlar[-1]["istek"]["state"]["recent_actions"]
    # Only 1 go_back (the explicit step 2), no synthetic duplicate
    go_backs = [h for h in son_gecmis if h.get("kind") == "go_back"]
    assert len(go_backs) == 1


def test_d1_non_matching_steps_enter_history_with_action_and_kind():
    """D1: 60_siniri and kontrol_yok steps enter history with their action and kind."""
    cok_actions = [
        {"id": f"e{i}", "node": i, "kind": "click", "role": "button", "label": f"B{i}"}
        for i in range(1, 65)
    ]
    kontrolsuz = [
        {"id": "e1", "node": 1, "kind": "click", "role": "button", "label": "Tamam"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    revealed = [
        {"id": "e99", "node": 99, "kind": "click", "role": "button", "label": "Yeni Buton"},
    ]
    adimlar = [
        # Step 1: 60_siniri CLICK (node 63)
        {"type": "step", "step": 1, "op": "CLICK", "node": 63, "label": "B63", "state": _sentetik_state(actions=cok_actions)},
        # Step 2: kontrol_yok SCROLL_UP (followed by interaction on new node 99)
        {"type": "step", "step": 2, "op": "SCROLL_UP", "scroll_y": 0, "state": _sentetik_state(actions=kontrolsuz)},
        # Step 3: CLICK on newly revealed node 99
        {"type": "step", "step": 3, "op": "CLICK", "node": 99, "label": "Yeni Buton", "state": _sentetik_state(actions=revealed)},
        # Step 4: DONE
        {"type": "step", "step": 4, "op": "DONE", "state": _sentetik_state()},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d1_non_matching.jsonl")
    assert len(kayitlar) == 2  # Step 3 and Step 4 produce labels
    assert [k["adim"] for k in kayitlar] == [3, 4]
    assert rapor["atilan_nedenleri"]["60_siniri"] == 1
    assert rapor["atilan_nedenleri"]["kontrol_yok"] == 1

    gecmis = kayitlar[0]["istek"]["state"]["recent_actions"]
    assert len(gecmis) == 2
    assert gecmis[0] == {"action": "B63", "kind": "click", "text": None, "page_changed": True}
    assert gecmis[1] == {"action": "Scroll up", "kind": "scroll", "text": None, "page_changed": True}


def test_d2_reading_scroll_discarded_when_next_action_on_known_element():
    """D2: Reading scroll run discarded ('okuma_kaydirmasi') if next action touches already visible node."""
    initial_actions = [
        {"id": "e1", "node": 10, "kind": "click", "role": "button", "label": "Mevcut Buton"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    adimlar = [
        {"type": "step", "step": 1, "op": "SCROLL_DOWN", "scroll_y": 0, "state": _sentetik_state(actions=initial_actions)},
        {"type": "step", "step": 2, "op": "SCROLL_DOWN", "scroll_y": 420, "state": _sentetik_state(actions=initial_actions)},
        {"type": "step", "step": 3, "op": "CLICK", "node": 10, "label": "Mevcut Buton", "state": _sentetik_state(actions=initial_actions)},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d2_okuma.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 3
    assert rapor["atilan_nedenleri"]["okuma_kaydirmasi"] == 2
    # But scrolls must remain in history:
    gecmis = kayitlar[0]["istek"]["state"]["recent_actions"]
    assert len(gecmis) == 2


def test_d2_reading_scroll_discarded_when_followed_by_done():
    """D2: Reading scroll run discarded ('okuma_kaydirmasi') if followed by DONE or session ends."""
    initial_actions = [
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    adimlar = [
        {"type": "step", "step": 1, "op": "SCROLL_DOWN", "scroll_y": 0, "state": _sentetik_state(actions=initial_actions)},
        {"type": "step", "step": 2, "op": "DONE", "state": _sentetik_state(actions=initial_actions)},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d2_done.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 2
    assert rapor["atilan_nedenleri"]["okuma_kaydirmasi"] == 1


def test_d2_reading_scroll_kept_when_next_action_on_new_element():
    """D2: Reading scroll run kept as labels if next action touches an element NOT in start-of-run actions."""
    initial_actions = [
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    revealed_actions = [
        {"id": "e5", "node": 55, "kind": "click", "role": "button", "label": "Yeni Buton"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    adimlar = [
        {"type": "step", "step": 1, "op": "SCROLL_DOWN", "scroll_y": 0, "state": _sentetik_state(actions=initial_actions)},
        {"type": "step", "step": 2, "op": "SCROLL_DOWN", "scroll_y": 420, "state": _sentetik_state(actions=revealed_actions)},
        {"type": "step", "step": 3, "op": "CLICK", "node": 55, "label": "Yeni Buton", "state": _sentetik_state(actions=revealed_actions)},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d2_yeni_oge.jsonl")
    assert len(kayitlar) == 3
    assert rapor["atilan_nedenleri"].get("okuma_kaydirmasi", 0) == 0


def test_d2_long_scroll_run_clipped_to_first_and_last_three():
    """D2: Scroll runs > 4 keep only the first step and last 3 steps as labels; intermediate discarded as 'uzun_kosu_kirpma'."""
    actions = [{"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420}]
    adimlar = [
        {"type": "step", "step": i, "op": "SCROLL_DOWN", "scroll_y": i * 420, "state": _sentetik_state(actions=actions)}
        for i in range(1, 7)  # 6 scrolls
    ]
    revealed = [
        {"id": "e99", "node": 99, "kind": "click", "role": "button", "label": "Yeni"},
        {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 420},
    ]
    adimlar.append({"type": "step", "step": 7, "op": "CLICK", "node": 99, "label": "Yeni", "state": _sentetik_state(actions=revealed)})

    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d2_uzun_kosu.jsonl")
    # Kept: step 1, steps 4, 5, 6, and step 7 -> 5 labels
    tutulan_adimlar = [k["adim"] for k in kayitlar]
    assert tutulan_adimlar == [1, 4, 5, 6, 7]
    assert rapor["atilan_nedenleri"]["uzun_kosu_kirpma"] == 2
    # All 6 scrolls must be in step 7's history
    gecmis = kayitlar[-1]["istek"]["state"]["recent_actions"]
    scroll_gecmis = [h for h in gecmis if h.get("kind") == "scroll"]
    assert len(scroll_gecmis) == 6


def test_d3_auto_exclusion_scroll_run_adjacent_to_unmapped_click():
    """D3: Scroll runs containing or adjacent to UNMAPPED_CLICK do not produce labels."""
    adimlar = [
        {"type": "step", "step": 1, "op": "SCROLL_DOWN", "scroll_y": 0, "state": _sentetik_state()},
        {"type": "step", "step": 2, "op": "UNMAPPED_CLICK", "unmapped": {"text": "Görünmeyen Buton"}, "state": _sentetik_state()},
        {"type": "step", "step": 3, "op": "SCROLL_DOWN", "scroll_y": 420, "state": _sentetik_state()},
        {"type": "step", "step": 4, "op": "DONE", "state": _sentetik_state()},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d3_unmapped.jsonl")
    assert len(kayitlar) == 1
    assert kayitlar[0]["adim"] == 4
    assert rapor["atilan_nedenleri"]["unmapped_click_bitisigi"] == 2
    assert rapor["atilan_nedenleri"]["unmapped_click"] == 1
    # All steps remain in history
    gecmis = kayitlar[0]["istek"]["state"]["recent_actions"]
    assert len(gecmis) == 3


def test_d3_manual_exclusion_haric_json():
    """D3: Steps matching ranges in haric.json do not produce labels, stay in history, and record drop reason."""
    haric_kurallari = [
        {"oturum": "*sentetik_haric*", "adimlar": [2, 3], "neden": "cevap_tiklamalari_kaydedilmedi"}
    ]
    adimlar = [
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "Ara", "state": _sentetik_state()},
        {"type": "step", "step": 2, "op": "CLICK", "node": 11, "label": "Ara", "state": _sentetik_state()},
        {"type": "step", "step": 3, "op": "CLICK", "node": 11, "label": "Ara", "state": _sentetik_state()},
        {"type": "step", "step": 4, "op": "DONE", "state": _sentetik_state()},
    ]
    kayitlar, rapor = oturum_satirlarini_donustur(
        _sentetik_oturum(adimlar),
        "sentetik_haric.jsonl",
        haric_listesi=haric_kurallari,
    )
    assert [k["adim"] for k in kayitlar] == [1, 4]
    assert rapor["atilan_nedenleri"]["cevap_tiklamalari_kaydedilmedi"] == 2
    gecmis = kayitlar[1]["istek"]["state"]["recent_actions"]
    assert len(gecmis) == 3


def test_d3_haric_json_file_exists_and_matches_schema():
    """D3: artifacts/demolar/haric.json exists, contains initial OKB rule."""
    haric_yolu = Path(__file__).resolve().parents[1] / "artifacts" / "demolar" / "haric.json"
    assert haric_yolu.is_file()
    veri = json.loads(haric_yolu.read_text(encoding="utf-8"))
    assert isinstance(veri, list)
    assert len(veri) >= 1
    kural = veri[0]
    assert kural["oturum"] == "*okb-bozuluk-testi*"
    assert kural["adimlar"] == [12, 38]
    assert kural["neden"] == "cevap_tiklamalari_kaydedilmedi"


def test_d4_submitted_with_enter_and_unmapped_click_counting():
    """D4: TYPE_TEXT with submitted_with_enter=true and UNMAPPED_CLICK are separately reported."""
    adimlar = [
        {"type": "step", "step": 1, "op": "TYPE_TEXT", "node": 10, "text": "test 1", "submitted_with_enter": True, "state": _sentetik_state()},
        {"type": "step", "step": 2, "op": "TYPE_TEXT", "node": 10, "text": "test 2", "submitted_with_enter": False, "state": _sentetik_state()},
        {"type": "step", "step": 3, "op": "TYPE_TEXT", "node": 10, "text": "test 3", "submitted_with_enter": True, "state": _sentetik_state()},
        {"type": "step", "step": 4, "op": "UNMAPPED_CLICK", "unmapped": {"text": "U1"}, "state": _sentetik_state()},
        {"type": "step", "step": 5, "op": "UNMAPPED_CLICK", "unmapped": {"text": "U2"}, "state": _sentetik_state()},
        {"type": "step", "step": 6, "op": "DONE", "state": _sentetik_state()},
    ]
    _kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d4_sayac.jsonl")
    assert rapor["type_text_enter_sayisi"] == 2
    assert rapor["unmapped_click_sayisi"] == 2


def test_d4_op_distribution_percentages():
    """D4: Operation distribution includes exact percentages for kept labels."""
    adimlar = [
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "B1", "state": _sentetik_state()},
        {"type": "step", "step": 2, "op": "CLICK", "node": 11, "label": "B2", "state": _sentetik_state()},
        {"type": "step", "step": 3, "op": "CLICK", "node": 11, "label": "B3", "state": _sentetik_state()},
        {"type": "step", "step": 4, "op": "DONE", "state": _sentetik_state()},
    ]
    _kayitlar, rapor = oturum_satirlarini_donustur(_sentetik_oturum(adimlar), "sentetik_d4_yuzde.jsonl")
    assert rapor["tutulan_op"] == {"CLICK": 3, "DONE": 1}
    assert rapor["tutulan_op_yuzde"]["CLICK"] == 75.0
    assert rapor["tutulan_op_yuzde"]["DONE"] == 25.0
    assert abs(sum(rapor["tutulan_op_yuzde"].values()) - 100.0) < 1e-4
    assert "birlestirme_oncesi_op_yuzde" in rapor
    assert "birlestirme_sonrasi_op_yuzde" in rapor


def test_d4_insan_demo_kayit_schema_validation(tmp_path: Path):
    """D4: Generated jsonl records strictly validate against schema, criteria keys, and prob sums."""
    girdi_dir = tmp_path / "girdi"
    cikti_dir = tmp_path / "cikti"
    girdi_dir.mkdir()
    oturum = _sentetik_oturum([
        {"type": "step", "step": 1, "op": "CLICK", "node": 11, "label": "Ara Butonu", "state": _sentetik_state()},
        {"type": "step", "step": 2, "op": "DONE", "state": _sentetik_state()},
    ])
    (girdi_dir / "01.jsonl").write_text("\n".join(json.dumps(r) for r in oturum) + "\n", encoding="utf-8")

    klasor_donustur(girdi_dir, cikti_dir)
    satirlar = [json.loads(line) for line in (cikti_dir / "insan_demo_kayit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(satirlar) == 2
    for s in satirlar:
        kayit_dogrula(s)
        assert set(s.keys()) == {"istek", "yanit", "kaynak", "oturum", "adim", "insan"}
        assert s["kaynak"] == "insan_demo"
        assert set(s["insan"].keys()) == {"op", "node", "label", "text", "option"}


def test_d5_sahte_gecmis_non_http_and_within_2s_filtered(tmp_path: Path):
    """D5: target_url non-http/https GO_BACK/GO_FORWARD/RELOAD and RELOAD/GO_FORWARD within 2s of a history step are dropped."""
    adimlar = [
        {"type": "step", "step": 1, "time": 100.0, "op": "CLICK", "node": 11, "label": "Ara Butonu", "state": _sentetik_state()},
        # 1. Real GO_BACK (http/https target_url) -> kept in history (s1_disi_op)
        {"type": "step", "step": 2, "time": 110.0, "op": "GO_BACK", "target_url": "https://ornek.test/onceki", "state": _sentetik_state()},
        # 2. Fake RELOAD with target_url=about:blank at 110.04s -> dropped as sahte_gecmis
        {"type": "step", "step": 3, "time": 110.04, "op": "RELOAD", "target_url": "about:blank", "state": _sentetik_state()},
        # 3. Fake RELOAD with https target_url but 0.38s after history step (<= 2.0s) -> dropped as sahte_gecmis
        {"type": "step", "step": 4, "time": 110.38, "op": "RELOAD", "target_url": "https://ornek.test/onceki", "state": _sentetik_state()},
        # 4. Fake GO_FORWARD within 2s (1.5s after step 4, 1.88s after step 2) -> dropped as sahte_gecmis
        {"type": "step", "step": 5, "time": 111.88, "op": "GO_FORWARD", "target_url": "https://ornek.test/sonraki", "state": _sentetik_state()},
        # 5. Fake GO_BACK with non-http target_url -> dropped as sahte_gecmis
        {"type": "step", "step": 6, "time": 120.0, "op": "GO_BACK", "target_url": "about:blank", "state": _sentetik_state()},
        # 6. Real RELOAD > 2s after any history step (time=125.0) -> kept in history (s1_disi_op)
        {"type": "step", "step": 7, "time": 125.0, "op": "RELOAD", "target_url": "https://ornek.test/onceki", "state": _sentetik_state()},
        # 7. Real GO_BACK within 2s of RELOAD is NOT dropped by the 2s rule (only RELOAD/GO_FORWARD are)
        {"type": "step", "step": 8, "time": 126.0, "op": "GO_BACK", "target_url": "http://ornek.test/ilk", "state": _sentetik_state()},
        {"type": "step", "step": 9, "time": 130.0, "op": "DONE", "state": _sentetik_state()},
    ]

    girdi_dir = tmp_path / "d5_girdi"
    cikti_dir = tmp_path / "d5_cikti"
    girdi_dir.mkdir()
    (girdi_dir / "01_d5.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in _sentetik_oturum(adimlar)) + "\n",
        encoding="utf-8",
    )

    rapor = klasor_donustur(girdi_dir, cikti_dir)
    assert rapor["sahte_gecmis"] == 4
    assert rapor["atilan_nedenleri"]["sahte_gecmis"] == 4
    assert rapor["atilan_nedenleri"]["s1_disi_op"] == 3
    assert rapor["oturumlar"][0]["sahte_gecmis"] == 4

    rapor_dosya = json.loads((cikti_dir / "rapor.json").read_text(encoding="utf-8"))
    assert rapor_dosya["sahte_gecmis"] == 4
    assert rapor_dosya["atilan_nedenleri"]["sahte_gecmis"] == 4

    kayitlar = [
        json.loads(line)
        for line in (cikti_dir / "insan_demo_kayit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(kayitlar) == 2
    gecmis = kayitlar[-1]["istek"]["state"]["recent_actions"]
    # History must contain ONLY: step 1 (click), step 2 (go_back), step 7 (reload), step 8 (go_back)
    assert [h["kind"] for h in gecmis] == ["click", "go_back", "reload", "go_back"]


