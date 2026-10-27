"""Configuration and environment loader for Reflex."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_env_path() -> Path:
    """Return the path to the .env file, respecting REFLEX_ENV_PATH or defaulting to cwd/.env."""
    custom = os.environ.get("REFLEX_ENV_PATH")
    if custom:
        return Path(custom)
    return Path.cwd() / ".env"


ENV_PATH = get_env_path()

OPENROUTER_BASE_URL = "https://openrouter.ai"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
META_API_CHAT_URL = "https://api.meta.ai/v1/chat/completions"
LOCAL_TURNIKE_URL = "http://127.0.0.1:8794/v1/systemone"

DEFAULT_DECISION_BACKEND = "local_turnike"
DEFAULT_JEV_MODEL = "~typesafe/jev-latest"
DEFAULT_SUPERVISOR_MODEL = "muse-spark-1.3-contributor"

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
    allow_model_fallback: bool = False
    max_steps: int = 80
    max_escalations: int = 40
    require_confirmation_for_destructive: bool = True
    verify_done_with_supervisor: bool = True
    enable_site_memory: bool = False
    screenshots: bool = True
    thresholds: CalibrationThresholds = Field(default_factory=CalibrationThresholds)
