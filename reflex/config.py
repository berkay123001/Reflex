"""Configuration and environment loader for Reflex."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_env_path() -> Path:
    """Return the path to the .env file, respecting REFLEX_ENV_PATH or defaulting to cwd/.env."""
    custom = os.environ.get("REFLEX_ENV_PATH")
    if custom:
        return Path(custom)
    return Path.cwd() / ".env"


ENV_PATH = get_env_path()

MAX_SYSTEM1_ELEMENTS = 60
MAX_SUPERVISOR_ELEMENTS = 150

OPENROUTER_BASE_URL = "https://openrouter.ai"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
META_API_CHAT_URL = "https://api.meta.ai/v1/chat/completions"
LOCAL_TURNIKE_URL = "http://127.0.0.1:8794/v1/systemone"

DEFAULT_DECISION_BACKEND = "openrouter_jev"
DEFAULT_JEV_MODEL = "~typesafe/jev-latest"
DEFAULT_SUPERVISOR_MODEL = "muse-spark-1.3-contributor"
DEFAULT_SUPERVISOR_TIMEOUT_S = 60.0
# Meta API reasoning effort for System 2. 2026-10-30, 19 real-site tasks x2 (contributor):
# "minimal" cut the median S2 call from 5.2 s to 3.4 s without lowering the completion rate.
DEFAULT_SUPERVISOR_REASONING_EFFORT = "minimal"

SUPERVISOR_PRESETS = [
    {"id": "muse-spark-1.3-contributor", "label": "Muse Spark 1.3 Contributor (Meta API)"},
    {"id": "muse-wonder-1.0", "label": "Muse Wonder 1.0 Fast (Meta API)"},
    {"id": "muse-spark-1.3", "label": "Muse Spark 1.3 (Meta API)"},
    {"id": "qwen/qwen3.8-flash", "label": "Qwen 3.8 Flash (OpenRouter)"},
    {"id": "qwen/qwen3.8-omni-flash", "label": "Qwen 3.8 Omni Flash (OpenRouter)"},
    {"id": "anthropic/claude-haiku-4.5", "label": "Claude Haiku 4.5 (OpenRouter)"},
]

DECISION_PRESETS = [
    {"id": "local_turnike", "label": "Local · Turnike-4B-Web2-FP8 (8794)"},
    {"id": "openrouter_jev", "label": "OpenRouter · Jev (~typesafe/jev-latest)"},
]

# Run modes offered by the entry points (Studio, MCP, eval CLI). The library default
# (`ReflexConfig()`) stays hybrid; entry points apply DEFAULT_RUN_MODE explicitly.
# 2026-10-29 ablation on 19 real-site tasks x2: s2_only 36/38, strict 33/38, hybrid 14/19.
RUN_MODES: dict[str, dict[str, bool]] = {
    "s2_only": {"s2_every_step": True, "s1_strict": False},
    "strict": {"s2_every_step": False, "s1_strict": True},
    "hybrid": {"s2_every_step": False, "s1_strict": False},
    "package": {"s2_every_step": False, "s1_strict": False, "s2_action_packages": True},
}
DEFAULT_RUN_MODE = "s2_only"


def run_mode_flags(mode: str | None) -> dict[str, bool]:
    """Return the ReflexConfig flags for a run mode; unknown or empty modes fall back to DEFAULT_RUN_MODE."""
    key = (mode or "").strip().lower()
    return dict(RUN_MODES.get(key, RUN_MODES[DEFAULT_RUN_MODE]))


def run_mode_of(s2_every_step: bool, s1_strict: bool, s2_action_packages: bool = False) -> str:
    """Return the run-mode id matching the given flags (s2_action_packages wins, then s2_every_step over s1_strict)."""
    if s2_action_packages:
        return "package"
    if s2_every_step:
        return "s2_only"
    return "strict" if s1_strict else "hybrid"

# USD per 1M tokens (Source: https://dev.meta.ai/docs/pricing-rate-limits)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "muse-spark-1.3-contributor": {
        "input": 0.10,
        "cached": 0.002,
        "output": 0.20,
    },
    "muse-spark-1.3": {
        "input": 1.25,
        "cached": 0.15,
        "output": 4.25,
    },
    "muse-wonder-1.0": {
        "input": 4.00,
        "cached": 0.20,
        "output": 20.00,
    },
}


def calculate_usage_cost(
    model: str,
    usage: dict[str, Any] | None,
    pricing_table: dict[str, dict[str, float]] | None = None,
    response_model: str | None = None,
) -> tuple[float | str, dict[str, Any]]:
    """Compute cost in USD from usage tokens or explicit usage.cost; return 'bilinmiyor' if model price is unknown.

    Looks up `model` (the requested model ID) FIRST in `pricing_table`, falling back to `response_model` only if needed.
    """
    raw = dict(usage or {})
    prompt_tokens = int(raw.get("prompt_tokens", 0) or 0)
    completion_tokens = int(raw.get("completion_tokens", 0) or 0)
    details = raw.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached_tokens = int(details.get("cached_tokens", 0) or 0)
    else:
        cached_tokens = int(raw.get("cached_tokens", 0) or 0)
    total_tokens = int(raw.get("total_tokens") or (prompt_tokens + completion_tokens))

    table = pricing_table or MODEL_PRICING
    clean_model = (model or "").strip()
    short_model = clean_model.removeprefix("meta/")
    rates = table.get(clean_model) or table.get(short_model)
    if rates is None and response_model:
        clean_resp = response_model.strip()
        short_resp = clean_resp.removeprefix("meta/")
        rates = table.get(clean_resp) or table.get(short_resp)

    if rates is not None:
        uncached_prompt = max(0, prompt_tokens - cached_tokens) if cached_tokens <= prompt_tokens else prompt_tokens
        in_rate = float(rates.get("input", 0.0))
        cached_rate = float(rates.get("cached", rates.get("cached_input", 0.0)))
        out_rate = float(rates.get("output", 0.0))
        cost_val: float | str = round(
            (uncached_prompt * in_rate + cached_tokens * cached_rate + completion_tokens * out_rate) / 1_000_000,
            8,
        )
    elif "cost" in raw and raw["cost"] is not None and not isinstance(raw["cost"], str):
        cost_val = float(raw["cost"])
    else:
        cost_val = "bilinmiyor"

    norm_usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
        "cost": cost_val,
    }
    if isinstance(details, dict) or cached_tokens > 0:
        norm_usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return cost_val, norm_usage


def load_env() -> None:
    """Load environment variables from .env without overwriting existing shell variables."""
    target_path = get_env_path()
    if not target_path.exists():
        return
    for raw_line in target_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key from environment or .env."""
    load_env()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        env_file = get_env_path()
        raise RuntimeError(
            f"OPENROUTER_API_KEY not found. Set it in {env_file} or in the shell environment."
        )
    return key


def get_meta_api_key() -> str:
    """Return the official Meta Model API key (`META_API_KEY` or `LLAMA_API_KEY`) if configured."""
    load_env()
    return (
        os.environ.get("META_API_KEY", "").strip()
        or os.environ.get("LLAMA_API_KEY", "").strip()
    )


def get_default_trace_path() -> str | None:
    """Return the default trace log path from REFLEX_TRACE_PATH, or None if unset."""
    load_env()
    val = os.environ.get("REFLEX_TRACE_PATH", "").strip()
    return val or None


def get_default_ax_supplement() -> bool:
    """Return whether AXTree supplementation is active from REFLEX_AX_SUPPLEMENT."""
    load_env()
    val = os.environ.get("REFLEX_AX_SUPPLEMENT", "1").strip().lower()
    return val not in {"0", "false", "no", "off"}


def get_default_light_stealth() -> bool:
    """Return whether light stealth mode is active from REFLEX_LIGHT_STEALTH."""
    load_env()
    val = os.environ.get("REFLEX_LIGHT_STEALTH", "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def save_env_key(key_name: str, key_value: str) -> None:
    """Persist an API key (`META_API_KEY`, `LLAMA_API_KEY`, `OPENROUTER_API_KEY`) into `.env` and `os.environ`."""
    if "\n" in key_name or "\r" in key_name:
        raise ValueError("key_name must not contain newline characters")
    if "\n" in key_value or "\r" in key_value:
        raise ValueError("key_value must not contain newline characters")
    clean_val = key_value.strip()
    if not clean_val:
        return
    os.environ[key_name] = clean_val
    target_path = get_env_path()
    lines: list[str] = []
    replaced = False
    if target_path.exists():
        for raw_line in target_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith(f"{key_name}="):
                lines.append(f"{key_name}={clean_val}")
                replaced = True
            else:
                lines.append(raw_line)
    if not replaced:
        lines.append(f"{key_name}={clean_val}")
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CalibrationThresholds(BaseModel):
    """Per-operation and joint probability thresholds for System 1 -> System 2 escalation.

    Addresses calibration differences across action types (e.g. DONE vs CLICK vs TYPE_TEXT)
    and catches target ambiguity via top1-top2 probability margin.
    """

    op_min_conf: dict[str, float] = Field(
        default_factory=lambda: {
            "CLICK": 0.42,
            "TYPE_TEXT": 0.42,
            "SELECT": 0.42,
            "SCROLL_DOWN": 0.15,
            "SCROLL_UP": 0.15,
            "WAIT": 0.15,
            "DONE": 0.72,
        }
    )
    target_min_conf: float = 0.32
    joint_min_conf: float = 0.20
    target_min_margin: float = 0.04  # Top-1 vs Top-2 probability margin (duplicate card links deduplicated in snapshot.js)
    max_no_progress_steps: int = 2   # Consecutive steps with zero element/page progress before escalation
    max_oscillation_cycles: int = 2  # A -> B -> A -> B state ping-pong limit


class ReflexConfig(BaseModel):
    """Runtime configuration for a Reflex session."""

    decision_backend: str = DEFAULT_DECISION_BACKEND
    jev_model: str = DEFAULT_JEV_MODEL
    local_turnike_url: str = LOCAL_TURNIKE_URL
    supervisor_model: str = DEFAULT_SUPERVISOR_MODEL
    supervisor_timeout_s: float = DEFAULT_SUPERVISOR_TIMEOUT_S
    supervisor_reasoning_effort: str = DEFAULT_SUPERVISOR_REASONING_EFFORT  # Meta API: minimal | low | medium | high
    allow_model_fallback: bool = False
    interactive: bool = False
    s2_every_step: bool = False  # evaluation baseline: no System 1, System 2 decides every step
    s1_strict: bool = False  # System 1 only for confident clicks/typing; scrolling and arbitrary choices go to System 2
    s2_action_packages: bool = False
    max_steps: int = 80
    max_escalations: int = 40  # System 2 call budget; see escalation_limit
    observe_deadline_s: float = 8.0
    require_confirmation_for_destructive: bool = True
    verify_done_with_supervisor: bool = True
    enable_site_memory: bool = False
    screenshots: bool = True
    thresholds: CalibrationThresholds = Field(default_factory=CalibrationThresholds)
    model_pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {k: dict(v) for k, v in MODEL_PRICING.items()}
    )
    trace_path: str | None = Field(default_factory=get_default_trace_path)
    ax_supplement: bool = Field(default_factory=get_default_ax_supplement)
    light_stealth: bool = Field(default_factory=get_default_light_stealth)

    @field_validator("trace_path", mode="before")
    @classmethod
    def _normalize_trace_path(cls, v: Any) -> str | None:
        if v is None:
            return None
        val = str(v).strip()
        return val or None

    @property
    def escalation_limit(self) -> int:
        """Effective System 2 call budget.

        In s2_only and strict modes System 2 decides every or most steps, so a budget below
        max_steps would stop the run early; max_steps bounds those runs instead.
        """
        if self.s2_every_step or self.s1_strict or self.s2_action_packages:
            return max(self.max_escalations, self.max_steps)
        return self.max_escalations
