"""Define environment-backed runtime configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load autonomous runtime and LLM configuration from the environment."""

    model_config = SettingsConfigDict(env_prefix="EVENTPILOT_", env_file=".env", extra="ignore")

    notification_destination: str = "local-console"
    database_path: Path = Path(".eventpilot/checkpoints.sqlite")
    mock_llm: bool = False
    llm_provider: str | None = Field(default=None, validation_alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, validation_alias="LLM_MODEL")
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_api_base: str | None = Field(default=None, validation_alias="LLM_API_BASE")

    @property
    def instructor_model(self) -> str | None:
        """Build Instructor's provider/model identifier when both values are configured."""
        if not self.llm_provider or not self.llm_model:
            return None
        return f"{self.llm_provider}/{self.llm_model}"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached application settings."""
    return Settings()
