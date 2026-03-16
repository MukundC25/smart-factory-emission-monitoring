"""Factory feature processing for ML-ready datasets."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.cluster import DBSCAN

from src.common import haversine_distance_km
from src.ingestion.factory_collector import CITY_COORDINATES

LOGGER = logging.getLogger(__name__)


class FactoryProcessor:
    """Build derived features and enforce final schema."""

    HIGH_RISK = {"chemical", "steel", "power", "pharmaceutical", "cement"}
    MEDIUM_RISK = {"manufacturing", "automotive", "paper", "food_processing"}

    def __init__(self, dbscan_eps: float = 0.05, dbscan_min_samples: int = 2) -> None:
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Run full processing pipeline for ML-ready output."""
        processed = raw_df.copy()
        processed = self.add_urban_rural_flag(processed)
        processed = self.add_pollution_risk_category(processed)
        processed = self.add_cluster_id(processed)
        processed = self.final_schema(processed)
        return processed

    def add_urban_rural_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each factory by distance to city center."""
        enriched = df.copy()

        def _classify(row: pd.Series) -> str:
            city = str(row.get("city", ""))
            center: Tuple[float, float] | None = CITY_COORDINATES.get(city)
            if center is None:
                return "unknown"

            lat = float(row.get("latitude", 0.0))
            lon = float(row.get("longitude", 0.0))
            distance_km = haversine_distance_km(lat, lon, center[0], center[1])
            if distance_km < 15:
                return "urban"
            if distance_km <= 40:
                return "peri-urban"
            return "rural"

        enriched["urban_rural"] = enriched.apply(_classify, axis=1)
        return enriched

    def add_pollution_risk_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map industry types to baseline pollution risk categories."""
        enriched = df.copy()

        def _risk(industry_type: str) -> str:
            token = str(industry_type or "").lower()
            if token in self.HIGH_RISK:
                return "High"
            if token in self.MEDIUM_RISK:
                return "Medium"
            return "Low"

        enriched["pollution_risk_category"] = enriched.get("industry_type", pd.Series(dtype=str)).map(_risk)
        return enriched

    def add_cluster_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign DBSCAN cluster ids based on lat/lon proximity."""
        enriched = df.copy()
        if enriched.empty:
            enriched["cluster_id"] = pd.Series(dtype="int64")
            return enriched

        coords = enriched[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
        valid_mask = coords.notna().all(axis=1)
        cluster_ids = pd.Series([-1] * len(enriched), index=enriched.index)

        if int(valid_mask.sum()) >= self.dbscan_min_samples:
            model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            labels = model.fit_predict(coords[valid_mask].to_numpy())
            cluster_ids.loc[valid_mask] = labels

        enriched["cluster_id"] = cluster_ids.astype(int)
        return enriched

    def final_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce exact final output columns, order, and dtypes."""
        ordered_columns = [
            "factory_id",
            "factory_name",
            "industry_type",
            "latitude",
            "longitude",
            "city",
            "state",
            "country",
            "source",
            "osm_id",
            "last_updated",
            "urban_rural",
            "pollution_risk_category",
            "cluster_id",
        ]

        normalized = df.copy()
        for column in ordered_columns:
            if column not in normalized.columns:
                default_value = -1 if column == "cluster_id" else ""
                normalized[column] = default_value

        normalized = normalized[ordered_columns]
        dtype_map: Dict[str, str] = {
            "factory_id": "string",
            "factory_name": "string",
            "industry_type": "string",
            "latitude": "float64",
            "longitude": "float64",
            "city": "string",
            "state": "string",
            "country": "string",
            "source": "string",
            "osm_id": "string",
            "last_updated": "string",
            "urban_rural": "string",
            "pollution_risk_category": "string",
            "cluster_id": "int64",
        }

        for column, dtype in dtype_map.items():
            if dtype == "float64":
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
            elif dtype == "int64":
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(-1).astype("int64")
            else:
                normalized[column] = normalized[column].astype(dtype)

        return normalized
