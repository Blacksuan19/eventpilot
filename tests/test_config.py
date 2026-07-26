"""Verify environment-backed runtime policy settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eventpilot.config import Settings


def test_max_wait_defaults_to_one_hour(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep unattended source discovery active at least once per hour."""
    monkeypatch.delenv("EVENTPILOT_MAX_WAIT_SECONDS", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings()

    assert settings.max_wait_seconds == 3_600


def test_max_wait_can_be_reduced_through_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Allow deployments to select a shorter polling ceiling."""
    monkeypatch.setenv("EVENTPILOT_MAX_WAIT_SECONDS", "900")
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.max_wait_seconds == 900


def test_max_wait_rejects_values_beyond_the_tool_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject configuration that the typed wait action cannot represent."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(max_wait_seconds=86_401)


def test_runtime_and_llm_environment_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Load process, graph, dashboard, and provider settings from their public variables."""
    values = {
        "EVENTPILOT_NOTIFICATION_DESTINATION": "ops-room",
        "EVENTPILOT_DATABASE_PATH": "/data/custom.sqlite",
        "EVENTPILOT_MOCK_LLM": "true",
        "EVENTPILOT_TIME_ACCELERATION": "60",
        "EVENTPILOT_MAX_PHYSICAL_WAIT_SECONDS": "2.5",
        "EVENTPILOT_MAX_WAIT_SECONDS": "900",
        "EVENTPILOT_RECURSION_LIMIT": "99",
        "EVENTPILOT_DASHBOARD_HOST": "127.0.0.1",
        "EVENTPILOT_DASHBOARD_PORT": "9000",
        "EVENTPILOT_EXTERNAL_CALL_TIMEOUT_SECONDS": "15",
        "EVENTPILOT_RETRY_MAX_ATTEMPTS": "4",
        "EVENTPILOT_RETRY_INITIAL_INTERVAL_SECONDS": "0.25",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-test",
        "LLM_API_KEY": "secret",
        "LLM_API_BASE": "https://llm.example/v1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.notification_destination == "ops-room"
    assert settings.database_path == Path("/data/custom.sqlite")
    assert settings.mock_llm is True
    assert settings.time_acceleration == 60
    assert settings.max_physical_wait_seconds == 2.5
    assert settings.max_wait_seconds == 900
    assert settings.recursion_limit == 99
    assert settings.dashboard_host == "127.0.0.1"
    assert settings.dashboard_port == 9000
    assert settings.external_call_timeout_seconds == 15
    assert settings.retry_max_attempts == 4
    assert settings.retry_initial_interval_seconds == 0.25
    assert settings.instructor_model == "openai/gpt-test"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "secret"
    assert settings.llm_api_base == "https://llm.example/v1"
