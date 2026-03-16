"""Factory cleaning and normalization pipeline."""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict

import pandas as pd

from src.ingestion.factory_collector import CITY_TO_STATE

LOGGER = logging.getLogger(__name__)


class FactoryDataCleaner:
    """Apply quality checks and normalization on collected factory data."""

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps in required order."""
        cleaned = df.copy()
        cleaned = self.remove_duplicates(cleaned)
        cleaned = self.validate_coordinates(cleaned)
        cleaned = self.normalize_factory_names(cleaned)
        cleaned = self.normalize_industry_types(cleaned)
        cleaned = self.add_derived_fields(cleaned)
        return cleaned

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates by OSM id and near-identical coordinates."""
        if df.empty:
            return df.copy()

        before = len(df)
        deduped = df.drop_duplicates(subset=["osm_id"], keep="first").copy()

        lat_round = pd.to_numeric(deduped["latitude"], errors="coerce").round(3)
        lon_round = pd.to_numeric(deduped["longitude"], errors="coerce").round(3)
        deduped["_lat_round"] = lat_round
        deduped["_lon_round"] = lon_round
        deduped = deduped.drop_duplicates(subset=["_lat_round", "_lon_round"], keep="first")
        deduped = deduped.drop(columns=["_lat_round", "_lon_round"])

        removed = before - len(deduped)
        LOGGER.info("Removed %s duplicates", removed)
        return deduped.reset_index(drop=True)

    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop invalid or out-of-India coordinates."""
        if df.empty:
            return df.copy()

        validated = df.copy()
        validated["latitude"] = pd.to_numeric(validated["latitude"], errors="coerce")
        validated["longitude"] = pd.to_numeric(validated["longitude"], errors="coerce")

        before = len(validated)
        missing_mask = validated["latitude"].isna() | validated["longitude"].isna()
        missing_count = int(missing_mask.sum())
        validated = validated[~missing_mask]

        in_india = (
            (validated["latitude"] >= 6)
            & (validated["latitude"] <= 38)
            & (validated["longitude"] >= 67)
            & (validated["longitude"] <= 98)
        )
        bounds_dropped = int((~in_india).sum())
        validated = validated[in_india]

        LOGGER.info(
            "Coordinate validation dropped %s null-coordinate rows and %s out-of-India rows",
            missing_count,
            bounds_dropped,
        )
        LOGGER.debug("Coordinate validation total dropped=%s", before - len(validated))
        return validated.reset_index(drop=True)

    def normalize_factory_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize factory names to clean title-cased values."""
        normalized = df.copy()

        def _normalize_name(row: pd.Series) -> str:
            raw = row.get("factory_name")
            value = "" if raw is None else str(raw)
            value = re.sub(r"[^A-Za-z0-9\s\-&]", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
            if value:
                return value.title()
            fallback_id = str(row.get("osm_id") or row.get("factory_id") or "unknown")
            return f"Industrial_Facility_{fallback_id}"

        normalized["factory_name"] = normalized.apply(_normalize_name, axis=1)
        return normalized

    def normalize_industry_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize industry labels into a standard taxonomy."""
        normalized = df.copy()

        mapping = {
            # Food-related industries
            "food": "food_processing",
            "brewery": "food_processing",
            "dairy": "food_processing",
            # Automotive-related industries
            "automobile": "automotive",
            "auto": "automotive",
            "vehicle": "automotive",
            # General / industrial categories - normalize to canonical labels
            "general industry": "general_industry",
            "general_industry": "general_industry",
            "general_industrial": "general_industry",
            "industrial": "industrial",
            # Works / manufacturing categories - normalize to canonical label
            "works": "works",
            "manufacturing": "works",
        }

        def _map_type(value: Any) -> str:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return "unknown"
            token = str(value).strip().lower()
            if not token:
                return "unknown"
            return mapping.get(token, token)

        normalized["industry_type"] = normalized.get("industry_type", pd.Series(dtype=str)).map(_map_type)
        normalized["industry_type"] = normalized["industry_type"].fillna("unknown")
        return normalized

    def _normalize_osm_id(self, osm_id: Any) -> str:
        # Treat explicit None, NaN, and empty/whitespace-only strings as unknown.
        if osm_id is None:
            return "unknown"
        if isinstance(osm_id, float) and pd.isna(osm_id):
            return "unknown"
        text = str(osm_id).strip()
        if not text or text.lower() in {"nan", "none"}:
            return "unknown"
        sanitized = re.sub(r"[^A-Za-z0-9]", "_", text)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized or "unknown"

    def add_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required schema fields and metadata."""
        enriched = df.copy()
        # Use raw osm_id values and let _normalize_osm_id handle missing/NaN cases.
        osm_series = enriched.get("osm_id")
        if osm_series is None:
            osm_series = pd.Series([None] * len(enriched), index=enriched.index)
        enriched["factory_id"] = osm_series.map(
            lambda osm_id: f"OSM_{self._normalize_osm_id(osm_id)}"
        )
        enriched["source"] = "OpenStreetMap"
        enriched["state"] = enriched.get("city", pd.Series(dtype=str)).map(lambda city: CITY_TO_STATE.get(str(city), "Unknown"))
        enriched["country"] = "India"
        enriched["last_updated"] = date.today().isoformat()
        return enriched

    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate aggregate quality summary for cleaned data."""
        if df.empty:
            return {
                "total_factories": 0,
                "by_city": {},
                "by_industry_type": {},
                "by_state": {},
                "coordinate_coverage": 0.0,
            }

        coord_ok = (
            pd.to_numeric(df.get("latitude"), errors="coerce").notna()
            & pd.to_numeric(df.get("longitude"), errors="coerce").notna()
        )
        return {
            "total_factories": int(len(df)),
            "by_city": df.get("city", pd.Series(dtype=str)).value_counts().to_dict(),
            "by_industry_type": df.get("industry_type", pd.Series(dtype=str)).value_counts().to_dict(),
            "by_state": df.get("state", pd.Series(dtype=str)).value_counts().to_dict(),
            "coordinate_coverage": round((float(coord_ok.mean()) * 100.0), 2),
        }
