"""CSV/Parquet data loader with in-memory caching and graceful fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_CONFIG_PATH: Optional[Path] = None


def _get_config() -> Dict[str, Any]:
    global _CONFIG_CACHE, _CONFIG_PATH
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    env_path = os.getenv("APP_CONFIG_PATH")
    config_path = Path(env_path) if env_path else Path(__file__).resolve().parent.parent.parent / "config.yaml"
    _CONFIG_PATH = config_path

    try:
        with config_path.open("r", encoding="utf-8") as file:
            _CONFIG_CACHE = yaml.safe_load(file) or {}
    except FileNotFoundError:
        logger.warning("Config not found at %s — using empty config", config_path)
        _CONFIG_CACHE = {}
    except Exception as exc:
        logger.exception("Failed to load config: %s", exc)
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE

# ---------------------------------------------------------------------------
# Canonical empty schemas — every column that downstream code may reference.
# Returned when a file is missing so the server never crashes on empty data.
# ---------------------------------------------------------------------------
_FACTORIES_SCHEMA: Dict[str, str] = {
    "factory_id": "object",
    "factory_name": "object",
    "industry_type": "object",
    "latitude": "float64",
    "longitude": "float64",
    "city": "object",
    "state": "object",
    "country": "object",
    "source": "object",
    "osm_id": "Int64",
    "last_updated": "object",
}

_POLLUTION_SCHEMA: Dict[str, str] = {
    "pm25": "float64",
    "pm10": "float64",
    "co": "float64",
    "no2": "float64",
    "so2": "float64",
    "o3": "float64",
    "aqi_index": "float64",
    "timestamp": "object",
    "station_name": "object",
    "station_lat": "float64",
    "station_lon": "float64",
    "city": "object",
    "country": "object",
    "source": "object",
    "nearest_factory_distance_km": "float64",
}

_RECOMMENDATIONS_SCHEMA: Dict[str, str] = {
    "factory_id": "object",
    "factory_name": "object",
    "industry_type": "object",
    "latitude": "float64",
    "longitude": "float64",
    "city": "object",
    "state": "object",
    "country": "object",
    "pollution_impact_score": "float64",
    "latest_pm25": "float64",
    "latest_pm10": "float64",
    "risk_level": "object",
    "recommendation": "object",
}


def _empty_frame(schema: Dict[str, str]) -> pd.DataFrame:
    """Build an empty DataFrame whose columns match the given schema.

    Args:
        schema: Mapping of column name to dtype string.

    Returns:
        Empty DataFrame with typed columns.
    """
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})


class DataLoader:
    """Cached data loader for all pipeline datasets.

    Loads factory, pollution, and recommendation datasets once from disk and
    serves them from an in-memory cache.  Cache is invalidated after
    ``CACHE_TTL_SECONDS`` or when :meth:`refresh` is called.

    If a file is not found or cannot be read, a correctly schema'd empty
    DataFrame is returned — the server never crashes on missing data.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialise the loader; does not read any files yet.

        Args:
            settings: Optional Settings override (useful in tests).
        """
        self._settings: Settings = settings or get_settings()
        self._factories: Optional[pd.DataFrame] = None
        self._pollution: Optional[pd.DataFrame] = None
        self._recommendations: Optional[pd.DataFrame] = None
        self._recommendation_reports: Optional[List[Dict[str, Any]]] = None
        self._loaded_at: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_stale(self) -> bool:
        """Return True if cache TTL has expired."""
        return (time.monotonic() - self._loaded_at) > self._settings.CACHE_TTL_SECONDS

    def _load_file(
        self,
        path: Path,
        empty_schema: Dict[str, str],
        label: str,
    ) -> pd.DataFrame:
        """Read a single data file, auto-detecting format.

        Args:
            path: Absolute path to the data file.
            empty_schema: Fallback column→dtype map used when file is absent.
            label: Human-readable name for log messages.

        Returns:
            Loaded DataFrame, or correctly schema'd empty DataFrame on error.
        """
        if not path.exists():
            logger.warning(
                "%s file not found at %s — returning empty DataFrame", label, path
            )
            return _empty_frame(empty_schema)

        try:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                df = pd.read_parquet(path)
            elif suffix == ".json":
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
            logger.info("Loaded %s: %d rows from %s", label, len(df), path)
            return df
        except Exception as exc:
            logger.exception(
                "Failed to load %s from %s — returning empty DataFrame: %s",
                label,
                path,
                exc,
            )
            return _empty_frame(empty_schema)

    def _resolve_pollution_path(self) -> Path:
        """Resolve which pollution dataset path the API should load.

        Priority:
        1) Explicit POLLUTION_CSV override from environment.
        2) Processed pollution dataset.
        3) Raw pollution readings.
        """
        if self._settings.POLLUTION_CSV is not None:
            return self._settings.POLLUTION_CSV
        if self._settings.PROCESSED_POLLUTION_FILE.exists():
            return self._settings.PROCESSED_POLLUTION_FILE
        return self._settings.RAW_POLLUTION_CSV

    def _load_all(self) -> None:
        """Load all three datasets and refresh the cache timestamp."""
        self._factories = self._load_file(
            self._settings.FACTORIES_CSV, _FACTORIES_SCHEMA, "factories"
        )
        self._pollution = self._load_file(
            self._resolve_pollution_path(), _POLLUTION_SCHEMA, "pollution"
        )
        self._recommendations = self._load_file(
            self._settings.RECOMMENDATIONS_CSV, _RECOMMENDATIONS_SCHEMA, "recommendations"
        )
        self._recommendation_reports = self._load_recommendation_reports()
        self._loaded_at = time.monotonic()

    def _resolve_recommendations_json_path(self) -> Path:
        """Resolve recommendations JSON path from config.yaml.

        Returns:
            Path: Absolute recommendations JSON path.
        """
        config = _get_config()
        base_dir = _CONFIG_PATH.parent if _CONFIG_PATH else Path.cwd()

        output_json = config.get("recommendations", {}).get("output_json")
        if output_json:
            return (base_dir / str(output_json)).resolve()

        recommendations_csv = config.get("paths", {}).get("recommendations")
        if recommendations_csv:
            return (base_dir / str(recommendations_csv)).with_suffix(".json").resolve()
        return self._settings.RECOMMENDATIONS_CSV.with_suffix(".json")

    def _load_recommendation_reports(self) -> List[Dict[str, Any]]:
        """Load full recommendation reports from JSON output.

        Returns:
            List[Dict[str, Any]]: Report dictionaries.
        """
        path = self._resolve_recommendations_json_path()
        if not path.exists():
            logger.warning("recommendations_json file not found at %s — returning empty list", path)
            return []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            reports = payload.get("reports", [])
            if isinstance(reports, list):
                logger.info("Loaded recommendations_json: %d reports from %s", len(reports), path)
                return reports
            logger.warning("recommendations_json payload malformed at %s — returning empty list", path)
            return []
        except Exception as exc:
            logger.exception(
                "Failed to load recommendations_json from %s — returning empty list: %s",
                path,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_factories(self) -> pd.DataFrame:
        """Return the factories DataFrame, reloading from disk if stale.

        Returns:
            pd.DataFrame: Factory records with guaranteed column schema.
        """
        if self._factories is None or self._is_stale():
            self._load_all()
        return self._factories.copy()  # type: ignore[return-value]

    def load_pollution(self) -> pd.DataFrame:
        """Return the pollution DataFrame, reloading from disk if stale.

        Returns:
            pd.DataFrame: Pollution readings with guaranteed column schema.
        """
        if self._pollution is None or self._is_stale():
            self._load_all()
        return self._pollution.copy()  # type: ignore[return-value]

    def load_recommendations(self) -> pd.DataFrame:
        """Return the recommendations DataFrame, reloading from disk if stale.

        Returns:
            pd.DataFrame: Factory recommendations with guaranteed column schema.
        """
        if self._recommendations is None or self._is_stale():
            self._load_all()
        return self._recommendations.copy()  # type: ignore[return-value]

    def load_recommendation_reports(self) -> List[Dict[str, Any]]:
        """Return recommendations JSON reports, reloading from disk if stale.

        Returns:
            List[Dict[str, Any]]: Full recommendation reports.
        """
        if self._recommendation_reports is None or self._is_stale():
            self._load_all()
        return list(self._recommendation_reports or [])

    def refresh(self) -> None:
        """Force reload all datasets from disk, ignoring cache TTL."""
        logger.info("Forced data refresh triggered")
        self._loaded_at = 0.0
        self._load_all()

    def dataset_info(self) -> Dict[str, int]:
        """Return row counts for each dataset (used by /health endpoint).

        Returns:
            Dict mapping dataset name to row count.
        """
        return {
            "factories": len(self.load_factories()),
            "pollution": len(self.load_pollution()),
            "recommendations": len(self.load_recommendations()),
        }
