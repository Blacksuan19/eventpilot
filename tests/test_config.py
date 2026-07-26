"""Verify environment-backed runtime policy settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from eventpilot.config import Settings


def test_max_wait_defaults_to_one_hour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
