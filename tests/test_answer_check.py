"""Final-answer check: Muse Spark Open sends S2 back once when the answer misses task requirements (offline)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from reflex.config import DEFAULT_ANSWER_CHECK_MODEL, ReflexConfig, get_default_answer_check_model
from reflex.context import DualTierContext
from reflex.supervisor import AnswerCheckOutcome, SupervisorDecision, check_final_answer
from tests.test_stuck_finish import _make_mock_session

OPEN_MODEL = "meta-models/muse-spark-open-1026"


def _completion(parsed: dict) -> tuple:
    return (parsed, 120, 0.002, OPEN_MODEL)


def test_check_parses_fail_with_unmet_requirements() -> None:
    ctx = DualTierContext(primary_goal="SO2 for Maine North, County Cork", active_subgoal="g")
    with patch("reflex.supervisor._chat_completion", return_value=_completion(
        {"verdict": "fail", "unmet": ["location is County Kerry, not County Cork"]}
    )) as call:
        out = check_final_answer(ctx=ctx, answer="Maine Bridge, Kerry: 1 ug/m3", page={"url": "u", "text": "t"}, model=OPEN_MODEL)
    assert out.verdict == "fail"
    assert out.unmet == ["location is County Kerry, not County Cork"]
    assert out.cost_usd == 0.002
    model, system_prompt, user_prompt = call.call_args.args[:3]
    assert model == OPEN_MODEL
    assert "replaces something the user specified" in system_prompt
    assert "User Task: SO2 for Maine North, County Cork" in user_prompt
    assert "<untrusted_page_data>" in user_prompt


def test_fail_without_unmet_items_counts_as_pass() -> None:
    ctx = DualTierContext(primary_goal="g", active_subgoal="g")
    with patch("reflex.supervisor._chat_completion", return_value=_completion({"verdict": "fail", "unmet": []})):
        assert check_final_answer(ctx=ctx, answer="a", page={}, model=OPEN_MODEL).verdict == "pass"


def _session(model: str = OPEN_MODEL):
    sess = _make_mock_session(goal="Find SO2 for Maine North, County Cork")
    sess.config.answer_check_model = model
    sess.answer_checks = []
    sess._answer_check_rejections = 0
    return sess


def _complete(sess, summary: str = "Maine Bridge, Kerry: 1 ug/m3") -> None:
    decision = SupervisorDecision(action_type="COMPLETE_TASK", summary_for_user=summary, reasoning="done")
    with patch("reflex.agent.resolve_escalation", return_value=decision):
        sess._handle_escalation(trigger_type="s2_every_step", reason="r", s1_decision=None, step_t0=time.perf_counter())


def test_failed_check_sends_s2_back_once_then_accepts() -> None:
    sess = _session()
    fail = AnswerCheckOutcome(verdict="fail", unmet=["County Kerry is not County Cork"], cost_usd=0.002)
    with patch("reflex.agent.check_final_answer", return_value=fail) as check:
        _complete(sess)
        assert sess.status == "ready"
        trigger, reason = sess.pending_escalation_reason
        assert trigger == "answer_check"
        assert "County Kerry is not County Cork" in reason
        assert "do not present a substitute" in reason
        sess.pending_escalation_reason = None
        _complete(sess, "Maine North, County Cork is not listed on AccuWeather.")
    assert sess.status == "done"
    assert check.call_count == 1
    assert sess._answer_check_rejections == 1
    assert len(sess.answer_checks) == 1


def test_passing_check_completes_immediately() -> None:
    sess = _session()
    with patch("reflex.agent.check_final_answer", return_value=AnswerCheckOutcome(verdict="pass")):
        _complete(sess)
    assert sess.status == "done"


def test_check_error_never_blocks_the_answer() -> None:
    sess = _session()
    with patch("reflex.agent.check_final_answer", side_effect=RuntimeError("BASETEN_API_KEY not configured")):
        _complete(sess)
    assert sess.status == "done"


def test_no_check_when_disabled_or_out_of_steps() -> None:
    sess = _session(model="")
    with patch("reflex.agent.check_final_answer") as check:
        _complete(sess)
    assert check.call_count == 0 and sess.status == "done"

    late = _session()
    late.history = [{"step": i} for i in range(late.config.max_steps - 4)]
    with patch("reflex.agent.check_final_answer") as check:
        _complete(late)
    assert check.call_count == 0 and late.status == "done"


@pytest.mark.parametrize(
    ("env", "key", "expected"),
    [("off", "bt", ""), ("", "bt", ""), (None, "bt", DEFAULT_ANSWER_CHECK_MODEL), (None, "", ""), ("custom-model", "", "custom-model")],
)
def test_default_answer_check_model(monkeypatch, env, key, expected) -> None:
    if env is None:
        monkeypatch.delenv("REFLEX_ANSWER_CHECK_MODEL", raising=False)
    else:
        monkeypatch.setenv("REFLEX_ANSWER_CHECK_MODEL", env)
    monkeypatch.setattr("reflex.config.load_env", lambda: None)
    monkeypatch.setattr("reflex.config.get_baseten_api_key", lambda: key)
    assert get_default_answer_check_model() == expected


def test_tests_run_with_the_check_off() -> None:
    assert ReflexConfig().answer_check_model == ""


def test_check_uses_its_own_short_timeout() -> None:
    sess = _session()
    sess.config.answer_check_timeout_s = 7.0
    with patch("reflex.agent.check_final_answer", return_value=AnswerCheckOutcome(verdict="pass")) as check:
        _complete(sess)
    assert check.call_args.kwargs["supervisor_timeout_s"] == 7.0
