"""Unit tests for Task G20: two-stage target scoring clean prompt path & regression test."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest

from reflex.turnike_server import (
    TurnikeWeb2Engine,
    _clone_cache,
    DEFAULT_MAX_LENGTH,
    MAX_OPTIONS,
    SCHEMA_MARKER,
)


def test_clone_cache_none():
    assert _clone_cache(None) is None


def test_clone_cache_tuple():
    mock_tensor = MagicMock()
    mock_tensor.clone.return_value = "cloned_tensor"
    cache = ((mock_tensor, mock_tensor),)
    cloned = _clone_cache(cache)
    assert cloned == (("cloned_tensor", "cloned_tensor"),)


def test_turnike_script_ab_preserved():
    """scripts/ab_turnike_target.py must exist in the repository."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "ab_turnike_target.py"
    assert script_path.exists(), "scripts/ab_turnike_target.py does not exist"
    content = script_path.read_text(encoding="utf-8")
    assert "clean" in content
    assert "reflex_path" in content


def test_score_question_with_cache_prefix_match_and_fallback():
    """Verify that _score_question_with_cache truncates ids when prefix matches, and falls back to full pass on mismatch."""
    engine = TurnikeWeb2Engine.__new__(TurnikeWeb2Engine)
    engine.device = "cpu"
    engine.sembol_agirlik = MagicMock()
    engine.max_length = DEFAULT_MAX_LENGTH
    engine.max_options = MAX_OPTIONS
    engine.done_engeli = 0.0

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [42]
    mock_tok.side_effect = lambda txt, add_special_tokens=False: {"input_ids": [10, 20, 30, 40, 50]}
    engine.tok = mock_tok
    engine.yer_id = 999

    # Setup mock govde forward pass
    mock_govde = MagicMock()
    fake_out = MagicMock()
    fake_out.last_hidden_state = MagicMock()
    # Mock hidden state slice [0, -1]
    fake_hidden = MagicMock()
    fake_hidden.float.return_value = fake_hidden
    fake_out.last_hidden_state.__getitem__.return_value = fake_hidden
    mock_govde.return_value = fake_out
    engine.govde = mock_govde

    # Mock torch module in turnike_server
    mock_torch = MagicMock()
    mock_torch.tensor.side_effect = lambda x, **kw: MagicMock(tolist=lambda: x)
    mock_tensor = MagicMock()
    mock_tensor.cpu.return_value.tolist.return_value = [0.8, 0.2]
    mock_torch.softmax.return_value = mock_tensor
    mock_torch.nn.functional.linear.return_value.__getitem__.return_value = mock_tensor

    with patch("reflex.turnike_server.torch", mock_torch):

            state = "Test State"
            ad = "click_target"
            soru = {"instructions": "Click it", "criteria": {"1": "Button 1", "2": "Button 2"}}

            # Case A: prefix matches
            # Let _split_and_tokenize return ids = [1, 2, 3, 4, 5], konumlar = [3]
            with patch.object(engine, "_split_and_tokenize", return_value=([1, 2, 3, 4, 5], [3])):
                res = engine._score_question_with_cache(
                    state=state,
                    ad=ad,
                    soru=soru,
                    pre_ids=[1, 2],
                    past_key_values=((MagicMock(),),),
                )
                assert res["choice"] == "1"
                # Govde must be called with truncated suffix: ids[len(pre_ids) : konumlar[0] + 1] -> [1, 2, 3, 4, 5][2 : 4] -> [3, 4]
                call_args = mock_govde.call_args
                passed_ids = call_args.kwargs["input_ids"].tolist()[0]
                assert passed_ids == [3, 4], f"Expected truncated suffix [3, 4], got {passed_ids}"
                assert call_args.kwargs["use_cache"] is False

            mock_govde.reset_mock()

            # Case B: prefix mismatch (pre_ids [99, 99] != [1, 2]) -> fallback to full clean pass
            with patch.object(engine, "_split_and_tokenize", return_value=([1, 2, 3, 4, 5], [3])):
                res = engine._score_question_with_cache(
                    state=state,
                    ad=ad,
                    soru=soru,
                    pre_ids=[99, 99],
                    past_key_values=((MagicMock(),),),
                )
                assert res["choice"] == "1"
                # Govde must be called with full clean pass: ids[: konumlar[0] + 1] -> [1, 2, 3, 4]
                call_args = mock_govde.call_args
                passed_ids = call_args.kwargs["input_ids"].tolist()[0]
                assert passed_ids == [1, 2, 3, 4], f"Expected full pass [1, 2, 3, 4], got {passed_ids}"
                assert call_args.kwargs["use_cache"] is False
                assert "past_key_values" not in call_args.kwargs or call_args.kwargs["past_key_values"] is None


def test_karar_never_concatenates_operation_and_target():
    """Verify that karar() calls _score_question_with_cache separately for operation and target without concatenated prompt."""
    engine = TurnikeWeb2Engine.__new__(TurnikeWeb2Engine)
    engine.device = "cpu"
    engine.max_length = DEFAULT_MAX_LENGTH
    engine.max_options = MAX_OPTIONS
    engine.done_engeli = 0.0
    engine.ad = "turnike-mock"

    govde_req = {
        "state": {"page": {"url": "https://example.com"}},
        "questions": {
            "operation": {
                "criteria": {"CLICK": "Click"},
                "instructions": "Which op?",
            },
            "click_target": {
                "criteria": {"1": "Button 1", "2": "Button 2"},
                "instructions": "Which button?",
            },
        },
    }

    with patch.object(engine, "_fit_state", return_value="State"):
        with patch.object(engine, "_prefill_state", return_value=([1, 2], "mock_pkv")):
            with patch.object(engine, "_score_question_with_cache") as mock_score:
                mock_score.side_effect = [
                    {"choice": "CLICK", "confidence": 0.95, "probabilities": {"CLICK": 0.95}},
                    {"choice": "1", "confidence": 0.90, "probabilities": {"1": 0.90, "2": 0.10}},
                ]

                out = engine.karar(govde_req)
                assert mock_score.call_count == 2
                # Verify first call is operation
                first_call = mock_score.call_args_list[0]
                assert first_call.kwargs["ad"] == "operation"
                # Verify second call is click_target as standalone question
                second_call = mock_score.call_args_list[1]
                assert second_call.kwargs["ad"] == "click_target"
                assert out["answers"]["operation"]["choice"] == "CLICK"
                assert out["answers"]["click_target"]["choice"] == "1"
