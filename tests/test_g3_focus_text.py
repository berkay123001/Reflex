"""Tests for Milestone G3: Focused DOM text truncation in safety.wrap_untrusted_dom and supervisor escalation."""

from unittest.mock import patch

from reflex.context import DualTierContext
from reflex.safety import extract_focused_full_text, wrap_untrusted_dom
from reflex.supervisor import resolve_escalation


def test_20k_char_middle_paragraph_extracted_when_focused():
    """G3 AC: 20.000 karakterlik metinde, ilgili paragraf ortada olsa bile focus sağlandığında çıktıda yer almalıdır."""
    filler_para = "Genel bilgilendirme ve standart sayfa içeriği dolgu metni. " * 10 + "\n\n"
    target_para = "Gizli Kampanya Kodu: REFLEX-G3-SPECIAL. Bu kupon ödeme adımında kullanılabilir.\n\n"

    # Construct ~20,000 char full_text with target located at offset ~10,000
    pre_text = (filler_para * 25)[:9500]
    post_text = (filler_para * 25)[:9500]
    full_text = pre_text + target_para + post_text
    assert len(full_text) >= 19000

    page = {"url": "https://shop.example", "title": "Store", "text": "Viewport", "full_text": full_text}

    # 1. With focus: target paragraph in the middle must be retained
    focused_out = wrap_untrusted_dom(page, [], focus="kampanya kodu kupon")
    assert "REFLEX-G3-SPECIAL" in focused_out
    assert "[... omitted ...]" in focused_out
    assert len(focused_out) < len(full_text)

    # 2. Without focus (legacy behavior): target paragraph in the middle must be omitted
    unfocused_out = wrap_untrusted_dom(page, [], focus="")
    assert "REFLEX-G3-SPECIAL" not in unfocused_out
    assert "...[middle section omitted]..." in unfocused_out


def test_empty_or_whitespace_or_none_focus_preserves_legacy_output():
    """G3 AC: focus boş bırakıldığında önceki çıktıyla (baştan 4.200 + sondan 3.000 karakter) aynı sonuç üretilmelidir."""
    long_text = ("Paragraf " + "A" * 80 + "\n\n") * 120  # ~10,000 chars
    page = {"url": "https://example.com", "title": "Legacy", "text": "Home", "full_text": long_text}

    expected_raw = long_text[:4200] + "\n...[middle section omitted]...\n" + long_text[-3000:]

    out_empty = wrap_untrusted_dom(page, [], focus="")
    out_whitespace = wrap_untrusted_dom(page, [], focus="   \t\n  ")
    out_none = wrap_untrusted_dom(page, [], focus=None)

    assert expected_raw in out_empty
    assert expected_raw in out_whitespace
    assert expected_raw in out_none
    assert "...[middle section omitted]..." in out_empty


def test_short_full_text_not_truncated():
    """G3 AC: Document <= 7500 chars -> returned fully without truncation regardless of focus."""
    short_text = "Short text content " * 150  # ~2,850 chars
    page = {"url": "https://example.com", "title": "Short", "text": "Short", "full_text": short_text}

    out = wrap_untrusted_dom(page, [], focus="content")
    assert "[... omitted ...]" not in out
    assert "...[middle section omitted]..." not in out
    assert short_text in out


def test_total_text_limit_does_not_exceed_7500_chars():
    """G3 AC: Toplam metin karakter sınırı 7.500 karakteri aşmamalıdır ve atlanan yerlerde '[... omitted ...]' yer almalıdır."""
    # Massive document with 40,000 characters where all paragraphs match
    matching_para = "Sipariş teslimat detayları ve fatura adresi bilgileri buradadır. " * 8 + "\n\n"
    huge_text = (matching_para * 60)[:40000]
    page = {"url": "https://example.com", "title": "Huge", "text": "Viewport", "full_text": huge_text}

    wrapped = wrap_untrusted_dom(page, [], focus="sipariş teslimat fatura")
    assert "[... omitted ...]" in wrapped

    # Extract the full text section from wrapped output
    full_marker = "Full Page Text (Entire Document From Top to Bottom):\n"
    assert full_marker in wrapped
    extracted_full = wrapped.split(full_marker)[1].split("</untrusted_page_data>")[0].strip()
    assert len(extracted_full) <= 7500


def test_turkish_suffix_first_5_letter_prefix_matching():
    """G3 AC: Turkish suffix matching on 5-character prefix (e.g. 'fiyatları' matches 'fiyat', 'siparişiniz' matches 'sipar')."""
    text = (
        ("Giriş bölümü " * 50 + "\n\n") * 10  # ~1,500 chars
        + "Müşteri siparişinizi aldık. Tüm süreç detaylandırılmış olarak takip edilebilir.\n\n"
        + ("Gereksiz detay dolgusu " * 50 + "\n\n") * 60  # ~15,000 chars
        + "Bitiş bölümü " * 40  # ~500 chars
    )
    page = {"url": "https://shop.tr", "title": "TR", "text": "Home", "full_text": text}

    # Focus in uppercase Turkish: SİPARİŞLER (prefix 'sipar') and DETAYLAR (prefix 'detay')
    out = wrap_untrusted_dom(page, [], focus="SİPARİŞLER VE DETAYLAR")
    assert "siparişinizi" in out
    assert "detaylandırılmış" in out

    # Test 'fiyatları' focus matching 'fiyat' in middle chunk
    text_fiyat = (
        ("Giriş dolgu " * 40 + "\n\n") * 20  # ~10,000 chars
        + "Özel indirimli ürün fiyat listesi burada mevcuttur.\n\n"
        + ("Çıkış dolgu " * 40 + "\n\n") * 20  # ~10,000 chars
    )
    page_fiyat = {"url": "https://fiyat.tr", "title": "Fiyat", "text": "H", "full_text": text_fiyat}
    out_fiyat = wrap_untrusted_dom(page_fiyat, [], focus="ürün fiyatları")
    assert "fiyat listesi" in out_fiyat


def test_focus_terms_under_4_chars_ignored():
    """Focus terms with length < 4 characters must be ignored, falling back to legacy truncation."""
    filler = ("Genel dolgu metni " * 40 + "\n\n") * 50
    page = {"url": "https://example.com", "title": "Test", "text": "Test", "full_text": filler}

    # 've', 'de', 'bir', 'in' are all < 4 chars
    out = wrap_untrusted_dom(page, [], focus="ve de bir in")
    assert "...[middle section omitted]..." in out
    assert "[... omitted ...]" not in out


def test_multiple_chunks_sorted_in_original_document_order():
    """Selected middle chunks must appear in their original document appearance order."""
    section_1 = "ÖNEMLİ BÖLÜM 1: Kargo takip numarası TR-998811.\n\n"
    section_2 = "ÖNEMLİ BÖLÜM 2: İade kabul şartları ve süreçleri.\n\n"
    section_3 = "ÖNEMLİ BÖLÜM 3: Müşteri hizmetleri iletişim kanalları.\n\n"

    filler = ("Dolgu metni " * 40 + "\n\n") * 15  # ~2,500 chars each

    doc = filler + section_1 + filler + section_2 + filler + section_3 + filler
    page = {"url": "https://doc.example", "title": "Doc", "text": "Doc", "full_text": doc}

    out = wrap_untrusted_dom(page, [], focus="kargo iade iletişim")
    pos_1 = out.find("ÖNEMLİ BÖLÜM 1")
    pos_2 = out.find("ÖNEMLİ BÖLÜM 2")
    pos_3 = out.find("ÖNEMLİ BÖLÜM 3")

    assert pos_1 != -1
    assert pos_2 != -1
    assert pos_3 != -1
    assert pos_1 < pos_2 < pos_3


def test_resolve_escalation_wires_focus_from_active_subgoal_or_primary_goal():
    """resolve_escalation must pass ctx.active_subgoal or ctx.primary_goal as focus to wrap_untrusted_dom."""
    target_text = "Kritik güvenlik kodu: ALPHA-OMEGA-999.\n\n"
    filler = ("Standart güvenlik kuralları dolgusu " * 30 + "\n\n") * 50
    full_text = filler[:9000] + target_text + filler[:9000]
    page = {"url": "https://sec.example", "title": "Sec", "text": "Sec", "full_text": full_text}

    # Case A: active_subgoal is set -> should extract target_text into supervisor user prompt
    ctx_a = DualTierContext(primary_goal="Generic Goal", active_subgoal="güvenlik kodu bul")
    with patch("reflex.supervisor._chat_completion") as mock_comp:
        mock_comp.return_value = ({"action_type": "OVERRIDE_ACTION"}, 100, 0.001, "fake-model")
        resolve_escalation(
            ctx=ctx_a,
            page=page,
            elements=[],
            history=[],
            escalation_reason="Stuck",
            trigger_type="stuck",
            s1_decision=None,
            supervisor_model="fake-model",
        )
        prompt_a = mock_comp.call_args[0][2]
        assert "ALPHA-OMEGA-999" in prompt_a

    # Case B: active_subgoal is empty -> fallback to primary_goal
    ctx_b = DualTierContext(primary_goal="güvenlik kodu bul", active_subgoal="")
    with patch("reflex.supervisor._chat_completion") as mock_comp:
        mock_comp.return_value = ({"action_type": "OVERRIDE_ACTION"}, 100, 0.001, "fake-model")
        resolve_escalation(
            ctx=ctx_b,
            page=page,
            elements=[],
            history=[],
            escalation_reason="Stuck",
            trigger_type="stuck",
            s1_decision=None,
            supervisor_model="fake-model",
        )
        prompt_b = mock_comp.call_args[0][2]
        assert "ALPHA-OMEGA-999" in prompt_b

    # Case C: neither has relevant keywords -> middle section omitted
    ctx_c = DualTierContext(primary_goal="Alakasız hedef", active_subgoal="")
    with patch("reflex.supervisor._chat_completion") as mock_comp:
        mock_comp.return_value = ({"action_type": "OVERRIDE_ACTION"}, 100, 0.001, "fake-model")
        resolve_escalation(
            ctx=ctx_c,
            page=page,
            elements=[],
            history=[],
            escalation_reason="Stuck",
            trigger_type="stuck",
            s1_decision=None,
            supervisor_model="fake-model",
        )
        prompt_c = mock_comp.call_args[0][2]
        assert "ALPHA-OMEGA-999" not in prompt_c


def test_extract_focused_full_text_direct_unit_tests():
    """Verify extract_focused_full_text directly on short documents, empty focus, and no match."""
    # Short document <= 7500
    assert extract_focused_full_text("hello world", focus="world") == "hello world"

    # Empty focus on long text
    long_doc = "A" * 10000
    legacy = extract_focused_full_text(long_doc, focus="")
    assert legacy == long_doc[:4200] + "\n...[middle section omitted]...\n" + long_doc[-3000:]

    # Focus terms with no matches in middle
    long_unmatched = ("Dolgu metni " * 40 + "\n\n") * 50
    out = extract_focused_full_text(long_unmatched, focus="tamamenfarklibirterim")
    assert "[... omitted ...]" in out
    assert len(out) <= 7500

