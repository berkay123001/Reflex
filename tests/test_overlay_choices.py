"""Tests for F11: invisible overlay radio/checkbox support in snapshot.js and Browser.act()."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from websockets.sync.client import connect as ws_connect

from reflex.browser.cdp import Browser, TargetCoveredError
from reflex.config import CalibrationThresholds
from reflex.gate import evaluate_decision, missing_form_labels
from tests.test_form_submit_gate import _click, _free_port, _snapshot_url, chromium_port

FIXTURES = Path(__file__).resolve().parent / "fixtures"
OVERLAY_FIXTURE = FIXTURES / "overlay_radios.html"
ETIKET_RISK_FIXTURE = FIXTURES / "etiket_risk.html"
KISA_KELIME_FIXTURE = FIXTURES / "kisa_kelime.html"


@contextmanager
def _chromium_browser(port: int, url: str) -> Iterator[Browser]:
    """Attach a real `Browser` controller to the isolated test Chromium on `port`."""
    deadline = time.monotonic() + 10.0
    ws_url = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                ws_url = str(data.get("webSocketDebuggerUrl") or "")
                if ws_url:
                    break
        except Exception:
            time.sleep(0.05)
    assert ws_url, f"Could not obtain webSocketDebuggerUrl from port {port}"

    with ws_connect(ws_url, max_size=50 * 1024 * 1024) as ws:
        msg_id = 0

        def local_cdp(method: str, session_id: str | None = None, **params: Any) -> dict[str, Any]:
            nonlocal msg_id
            msg_id += 1
            cur_id = msg_id
            payload: dict[str, Any] = {"id": cur_id, "method": method, "params": params}
            if session_id is not None:
                payload["sessionId"] = session_id
            ws.send(json.dumps(payload))
            while True:
                raw = ws.recv(timeout=15.0)
                msg = json.loads(raw)
                if msg.get("id") == cur_id:
                    if "error" in msg:
                        raise RuntimeError(json.dumps(msg["error"]))
                    return dict(msg.get("result") or {})

        with (
            patch("reflex.browser.cdp.ensure_chromium_cdp"),
            patch("reflex.browser.cdp.ensure_daemon"),
            patch("reflex.browser.cdp.cdp", side_effect=local_cdp),
        ):
            browser = Browser(url)
            try:
                yield browser
            finally:
                browser.close()


def test_overlay_radios_fixture_lists_all_nine_options_and_finish_button(chromium_port: int) -> None:
    """9 options are listed with 'Soru → Şık' labels, role=radio, non-empty group, plus 'Sonucu gör'."""
    snap = _snapshot_url(chromium_port, OVERLAY_FIXTURE.resolve().as_uri())
    actions = snap["actions"]

    radios = [a for a in actions if a.get("role") == "radio"]
    assert len(radios) == 9

    expected_questions = {
        "q1": "1. Uyku düzeniniz nasıl?",
        "q2": "2. Gün içinde enerjiniz nasıl?",
        "q3": "3. Karar verme süreciniz nasıl?",
    }
    for r in radios:
        grp = r.get("group")
        assert grp in expected_questions
        q_text = expected_questions[grp]
        assert r.get("group_label") == q_text
        assert r.get("accessible_name")
        assert r["label"] == f"{q_text} → {r['accessible_name']}"
        assert r.get("checked") == "false"
        assert r.get("form_key") == "loose"
        assert str(r["node"]) in snap["guards"] and snap["guards"][str(r["node"])] is not None

    submit = next(a for a in actions if a.get("accessible_name") == "Sonucu gör")
    assert submit["role"] == "button"
    assert submit.get("submits_form") is True
    assert submit.get("form_key") == "loose"


def test_overlay_radios_incomplete_form_gate_reports_question_headings(chromium_port: int) -> None:
    """Clicking 'Sonucu gör' with unanswered questions triggers incomplete_form with question titles."""
    snap = _snapshot_url(chromium_port, OVERLAY_FIXTURE.resolve().as_uri())
    actions = snap["actions"]
    submit = next(a for a in actions if a.get("accessible_name") == "Sonucu gör")

    missing = missing_form_labels(submit, actions)
    assert missing == [
        "1. Uyku düzeniniz nasıl?",
        "2. Gün içinde enerjiniz nasıl?",
        "3. Karar verme süreciniz nasıl?",
    ]
    for label in missing:
        assert not re.fullmatch(r"rg\d+|q\d+", label)

    gate = evaluate_decision(
        _click(submit["id"]),
        CalibrationThresholds(),
        action=submit,
        page_actions=actions,
    )
    assert gate.should_escalate is True
    assert gate.trigger_type == "incomplete_form"
    assert "1. Uyku düzeniniz nasıl?" in gate.reason
    assert "2. Gün içinde enerjiniz nasıl?" in gate.reason
    assert "3. Karar verme süreciniz nasıl?" in gate.reason
    assert not re.search(r"\brg\d+\b", gate.reason)


def test_act_clicks_overlay_radio_and_sets_checked_true(chromium_port: int) -> None:
    """Browser.act() clicks an opacity:0 overlay radio by coordinate and sets input.checked === true."""
    with _chromium_browser(chromium_port, OVERLAY_FIXTURE.resolve().as_uri()) as browser:
        page = browser.observe(screenshot=False)
        target_action = next(
            a
            for a in page["actions"]
            if a.get("role") == "radio"
            and a.get("label") == "3. Karar verme süreciniz nasıl? → Karar verirken güçlük çekmiyorum."
        )
        assert browser.fresh(page, target_action) is True

        res = browser.act(target_action, page)
        assert res["executed"] == target_action["id"]

        checked = browser.evaluate(
            "document.querySelector('input[name=\"q3\"][value=\"0\"]').checked"
        )
        assert checked is True

        page_after = browser.observe(screenshot=False)
        updated = next(a for a in page_after["actions"] if a.get("node") == target_action["node"])
        assert updated.get("checked") == "true"


_COVERED_AND_FALLBACK_HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <style>
    body { font: 16px sans-serif; margin: 24px; }
    .row { position: relative; padding: 8px 12px; margin: 8px 0; height: 28px; }
    .overlay-input { opacity: 0; position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; }
    .blocker { position: absolute; inset: 0; z-index: 5; background: rgba(255, 255, 255, 0.9); }
  </style>
</head>
<body>
  <div class="mb-4">
    <h4>Negatif ve etiket sırası</h4>
    <!-- 1. Covered opacity:0 radio: blocker div sits above input at center -->
    <div class="row">
      <input id="covered_radio" type="radio" name="g_cov" value="covered" class="overlay-input">
      <label>Kaplanmış şık</label>
      <div class="blocker">Üstteki engel</div>
    </div>
    <!-- 2. Uncovered parent short text fallback (no label element, <= 80 chars) -->
    <div class="row">
      <input id="parent_short" type="checkbox" name="c_short" class="overlay-input">
      <span>Kısa ebeveyn metni</span>
    </div>
    <!-- 3. Uncovered parent long text (> 80 chars) does not use parent text as choice fallback -->
    <div class="row">
      <input id="parent_long" type="checkbox" name="c_long" class="overlay-input">
      <span>Bu ebeveyn metni seksen karakter sınırını aşacak kadar uzun yazılmıştır ve doğrudan etiket olarak alınmamalıdır.</span>
    </div>
  </div>
</body>
</html>
"""


def test_covered_opacity_zero_input_is_not_listed_and_rejected_by_act(
    chromium_port: int, tmp_path: Path
) -> None:
    """An opacity:0 input covered by another element at its center is excluded from snapshot and act()."""
    html_file = tmp_path / "covered.html"
    html_file.write_text(_COVERED_AND_FALLBACK_HTML, encoding="utf-8")

    snap = _snapshot_url(chromium_port, html_file.resolve().as_uri())
    actions = snap["actions"]
    labels = [a.get("label") for a in actions]
    roles = [a.get("role") for a in actions]

    # 1. Covered radio must NOT be listed
    assert "radio" not in roles
    assert not any("Kaplanmış şık" in str(lbl) for lbl in labels)

    # 2. Uncovered checkbox with short parent text (<= 80 chars) is listed with that text
    short_cb = next(a for a in actions if a.get("role") == "checkbox" and "Kısa ebeveyn metni" in a["label"])
    assert short_cb["accessible_name"] == "Kısa ebeveyn metni"

    # 3. Uncovered checkbox with parent text > 80 chars falls back to role name ('checkbox')
    long_cb = next(a for a in actions if a.get("role") == "checkbox" and a["node"] != short_cb["node"])
    assert long_cb["accessible_name"] == ""

    # 4. If an overlay input becomes covered after observation, act() refuses to click it
    with _chromium_browser(chromium_port, html_file.resolve().as_uri()) as browser:
        page = browser.observe(screenshot=False)
        cb_action = next(
            a for a in page["actions"] if a.get("role") == "checkbox" and a.get("accessible_name") == "Kısa ebeveyn metni"
        )
        # Cover it with an element that does not change its guard attributes, then test act() target check
        browser.evaluate(
            """(() => {
              const inp = document.getElementById('parent_short');
              const cover = document.createElement('div');
              cover.style.cssText = 'position:fixed;inset:0;z-index:9999;background:transparent';
              document.body.appendChild(cover);
            })()"""
        )
        # Bypass fresh() to directly test F11b's act() target evaluation on a covered opacity:0 input
        with patch.object(browser, "fresh", return_value=True):
            with pytest.raises(TargetCoveredError):
                browser.act(cb_action, page)
        assert browser.evaluate("document.getElementById('parent_short').checked") is False


_CUSTOM_QUIZ_H5_HTML = """<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <style>
    body { font: 16px sans-serif; margin: 24px; }
    .quiz-q { margin-bottom: 24px; max-width: 720px; }
    h5.flex.items-start {
      display: flex;
      align-items: flex-start;
      cursor: pointer;
      padding: 12px;
      margin: 8px 0;
      min-height: 48px;
      border: 1px solid #ddd;
    }
    .circle-wrap { margin-right: 8px; }
    .circle { font-family: 'Material Symbols'; display: inline-block; width: 20px; height: 20px; }
    .circle-lig { display: inline-block; width: 20px; height: 20px; letter-spacing: -0.35em; }
    .circle-dot { display: none; }
  </style>
</head>
<body>
  <div class="quiz-q">
    <h2>1. İş görüşmesinde işler yolunda gitmedi, ne yaparsın?</h2>
    <h5 class="flex items-start">
      <span class="circle-wrap"><span class="circle">circle</span><span class="circle-dot">circle-dot</span></span>
      Hatamı kabul eder ve bu deneyimi gelişme fırsatına dönüştürmeye çalışırım.
    </h5>
    <h5 class="flex items-start">
      <span class="circle-wrap"><span class="circle-lig">circle</span><span class="circle-dot">circle-dot</span></span>
      Öfkeyi hızlı bir şekilde kontrol ederim ve pozitif bir bakış açısı geliştiririm.
    </h5>
    <h5 class="flex items-start">
      <span class="circle-wrap"><i class="circle">circle</i><i class="circle-dot">circle-dot</i></span>
      Değişikliklere hızlı bir şekilde uyum sağlarım ve yeni durumlara adapte olurum.
    </h5>
    <h5 class="flex items-start">
      <span class="circle-wrap"><span class="circle">circle</span><span class="circle-dot">circle-dot</span></span>
      Ekiple fikir alışverişi yapar ve herkesin karekterine uygun görevlendirmeyle başarıya ulaşırız.
    </h5>
  </div>
  <a href="https://example.test/more">Daha Fazla Sana Özel İçerikler <span class="circle">arrow_forward</span></a>
</body>
</html>
"""


def test_f12_custom_h5_quiz_options_strip_icon_class_residue_and_are_listed(
    chromium_port: int, tmp_path: Path
) -> None:
    """F12: h5 (cursor:pointer) quiz options with 'circle circle-dot' icons are listed with clean labels."""
    html_file = tmp_path / "custom_quiz_h5.html"
    html_file.write_text(_CUSTOM_QUIZ_H5_HTML, encoding="utf-8")

    snap = _snapshot_url(chromium_port, html_file.resolve().as_uri())
    actions = [a for a in snap["actions"] if a.get("kind") == "click"]
    labels = [a["label"] for a in actions]

    expected_options = [
        "Hatamı kabul eder ve bu deneyimi gelişme fırsatına dönüştürmeye çalışırım.",
        "Öfkeyi hızlı bir şekilde kontrol ederim ve pozitif bir bakış açısı geliştiririm.",
        "Değişikliklere hızlı bir şekilde uyum sağlarım ve yeni durumlara adapte olurum.",
        "Ekiple fikir alışverişi yapar ve herkesin karekterine uygun görevlendirmeyle başarıya ulaşırız.",
    ]
    for opt in expected_options:
        assert opt in labels, f"Missing option {opt!r} in {labels}"

    for lbl in labels:
        assert "circle" not in lbl
        assert "circle-dot" not in lbl
        assert "arrow_forward" not in lbl

    assert "Daha Fazla Sana Özel İçerikler" in labels


def test_f12b_etiket_risk_fixture_preserves_all_labels(chromium_port: int) -> None:
    """F12b: tests/fixtures/etiket_risk.html keeps real words matching class names or years/hyphens."""
    snap = _snapshot_url(chromium_port, ETIKET_RISK_FIXTURE.resolve().as_uri())
    labels = [a["label"] for a in snap["actions"] if a.get("kind") == "click"]
    assert labels == [
        "Add to cart",
        "Fiyat 1.250 TL price",
        "2025-2026 ücretleri",
        "ücretler 2025-2026",
        "e-posta ile giriş",
        "covid-19 testi",
        "Haberler title",
    ]


def test_f12b_material_symbols_icon_fixture_stripped(chromium_port: int, tmp_path: Path) -> None:
    """F12b: <h5 style='cursor:pointer'><span style='font-family:Material Symbols;...'>circle</span> Şık metni</h5> -> 'Şık metni'."""
    html_file = tmp_path / "icon_fixture.html"
    html_file.write_text(
        '<!doctype html><html><body>'
        '<h5 style="cursor:pointer"><span style="font-family:\'Material Symbols\';'
        'display:inline-block;width:24px;height:24px">circle</span> Şık metni</h5>'
        '</body></html>',
        encoding="utf-8",
    )
    snap = _snapshot_url(chromium_port, html_file.resolve().as_uri())
    labels = [a["label"] for a in snap["actions"] if a.get("kind") == "click"]
    assert labels == ["Şık metni"]


def test_f12c_kisa_kelime_fixture_preserves_short_words(chromium_port: int) -> None:
    """F12c: tests/fixtures/kisa_kelime.html keeps short 3+ letter words while stripping ligature icons."""
    snap = _snapshot_url(chromium_port, KISA_KELIME_FIXTURE.resolve().as_uri())
    labels = [a["label"] for a in snap["actions"] if a.get("kind") == "click"]
    assert labels == [
        "Ara",
        "Yeni Sezon",
        "Sale Kulaklık",
        "Tek yön",
        "Gidiş Dönüş",
        "Oyna",
        "Evet Hayır",
    ]



