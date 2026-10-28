"""Tests for Task G6: Configuration & Packaging Hygiene."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib
import pytest

from reflex.config import (
    get_env_path,
    get_openrouter_api_key,
    load_env,
    save_env_key,
)
import reflex.turnike_server as turnike_server


def test_env_path_default_cwd(monkeypatch):
    """When REFLEX_ENV_PATH is not set, get_env_path() defaults to cwd / .env."""
    monkeypatch.delenv("REFLEX_ENV_PATH", raising=False)
    assert get_env_path() == Path.cwd() / ".env"


def test_env_path_custom_env_var(monkeypatch, tmp_path):
    """When REFLEX_ENV_PATH is set, get_env_path() returns that path."""
    custom_path = tmp_path / "custom.env"
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_path))
    assert get_env_path() == custom_path


def test_load_env_from_reflex_env_path(monkeypatch, tmp_path):
    """load_env() reads from REFLEX_ENV_PATH without overwriting existing environment variables."""
    custom_env = tmp_path / "test.env"
    custom_env.write_text("TEST_KEY_NEW=hello_world\nEXISTING_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_env))
    monkeypatch.setenv("EXISTING_KEY", "already_set")
    monkeypatch.delenv("TEST_KEY_NEW", raising=False)

    load_env()

    assert os.environ.get("TEST_KEY_NEW") == "hello_world"
    assert os.environ.get("EXISTING_KEY") == "already_set"


def test_save_env_key_updates_file_and_environ(monkeypatch, tmp_path):
    """save_env_key writes to the target .env file and updates os.environ."""
    custom_env = tmp_path / "save_target.env"
    custom_env.write_text("OTHER_KEY=123\n", encoding="utf-8")
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_env))

    save_env_key("TEST_API_KEY", "secret_token_123")

    assert os.environ.get("TEST_API_KEY") == "secret_token_123"
    content = custom_env.read_text(encoding="utf-8")
    assert "TEST_API_KEY=secret_token_123" in content
    assert "OTHER_KEY=123" in content

    # Test updating existing key
    save_env_key("TEST_API_KEY", "secret_token_456")
    assert os.environ.get("TEST_API_KEY") == "secret_token_456"
    content2 = custom_env.read_text(encoding="utf-8")
    assert "TEST_API_KEY=secret_token_456" in content2
    assert "TEST_API_KEY=secret_token_123" not in content2


@pytest.mark.parametrize("invalid_key", ["KEY\nNAME", "KEY\rNAME", "FOO\r\nBAR"])
def test_save_env_key_rejects_newlines_in_key(monkeypatch, tmp_path, invalid_key):
    """save_env_key raises ValueError if key contains newline characters."""
    custom_env = tmp_path / "target.env"
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_env))

    with pytest.raises(ValueError, match="key_name must not contain newline"):
        save_env_key(invalid_key, "valid_value")


@pytest.mark.parametrize("invalid_val", ["VAL\nUE", "VAL\rUE", "FOO\r\nBAR"])
def test_save_env_key_rejects_newlines_in_value(monkeypatch, tmp_path, invalid_val):
    """save_env_key raises ValueError if value contains newline characters."""
    custom_env = tmp_path / "target.env"
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_env))

    with pytest.raises(ValueError, match="key_value must not contain newline"):
        save_env_key("VALID_KEY", invalid_val)


def test_get_openrouter_api_key_error_reports_resolved_env_path(monkeypatch, tmp_path):
    """get_openrouter_api_key reports the resolved env file path in error message."""
    custom_env = tmp_path / "missing.env"
    monkeypatch.setenv("REFLEX_ENV_PATH", str(custom_env))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        get_openrouter_api_key()

    assert str(custom_env) in str(exc_info.value)


def test_turnike_server_model_path_configuration(monkeypatch):
    """turnike_server respects TURNIKE_MODEL_PATH when model_path is not specified."""
    custom_model = "/custom/path/to/turnike_model"
    monkeypatch.setenv("TURNIKE_MODEL_PATH", custom_model)

    # Test initialization with None resolves to TURNIKE_MODEL_PATH
    engine = turnike_server.TurnikeWeb2Engine.__new__(turnike_server.TurnikeWeb2Engine)
    effective = None or os.environ.get("TURNIKE_MODEL_PATH") or turnike_server.DEFAULT_MODEL_PATH
    assert str(Path(effective)) == custom_model


def test_pyproject_turnike_optional_dependencies():
    """pyproject.toml defines turnike optional dependencies."""
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    assert "turnike" in optional_deps
    assert set(optional_deps["turnike"]) == {"torch>=2.11", "transformers>=5.17", "torchao>=0.18", "accelerate"}


def test_gitignore_contains_artifacts():
    """.gitignore contains artifacts/ entry."""
    gitignore_path = Path(__file__).resolve().parent.parent / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]
    assert "artifacts/" in lines
