"""Application configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All application settings.

    Values are read from environment variables and the project-level .env file.
    All paths use pathlib.Path to avoid platform-specific separator issues.

    Attributes:
        DATA_DIR: Root data directory.
        FACTORIES_CSV: Path to factories dataset.
        RAW_POLLUTION_CSV: Path to raw pollution readings dataset.
        PROCESSED_POLLUTION_FILE: Path to processed pollution dataset.
        POLLUTION_CSV: Optional explicit override path for API pollution source.
        PROCESSED_DIR: Path to processed data directory.
        RECOMMENDATIONS_CSV: Path to recommendations dataset.
        API_HOST: Host to bind the API server to.
        API_PORT: Port to bind the API server to.
        CORS_ORIGINS: Allowed CORS origins.
        LOG_LEVEL: Python logging level name.
        CACHE_TTL_SECONDS: Data cache time-to-live in seconds.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    DATA_DIR: Path = _PROJECT_ROOT / "data"
    FACTORIES_CSV: Path = _PROJECT_ROOT / "data" / "raw" / "factories" / "factories.csv"
    RAW_POLLUTION_CSV: Path = (
        _PROJECT_ROOT / "data" / "raw" / "pollution" / "pollution_readings.csv"
    )
    PROCESSED_POLLUTION_FILE: Path = _PROJECT_ROOT / "data" / "processed" / "ml_dataset.parquet"
    POLLUTION_CSV: Optional[Path] = None
    PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"
    RECOMMENDATIONS_CSV: Path = _PROJECT_ROOT / "data" / "output" / "recommendations.csv"

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    LOG_LEVEL: str = "INFO"
    CACHE_TTL_SECONDS: int = 300


def get_settings() -> Settings:
    """Return application settings, reading from .env automatically.

    Returns:
        Settings: Populated settings instance.
    """
    return Settings()
