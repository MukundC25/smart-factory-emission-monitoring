"""
PollutionRiskScorer: Computes risk scores for factories based on pollution data.

- Uses weighted pollutant thresholds
- Maps pollutant values to risk scores (0-10)
- Assigns composite risk and dominant pollutant
- Handles spatial join using nearest_factory_distance_km (precomputed)
- Falls back to city/national averages if needed

Type hints and Google-style docstrings throughout.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)

class PollutionRiskScorer:
    """Computes pollution risk scores for factories using weighted pollutant metrics."""

    risk_weights: Dict[str, Dict[str, Any]] = {
        "pm25":  {"weight": 0.30, "thresholds": {"good": 12,  "moderate": 35,  "poor": 55,  "severe": 150}},
        "pm10":  {"weight": 0.20, "thresholds": {"good": 54,  "moderate": 154, "poor": 254, "severe": 354}},
        "so2":   {"weight": 0.20, "thresholds": {"good": 40,  "moderate": 80,  "poor": 380, "severe": 800}},
        "no2":   {"weight": 0.15, "thresholds": {"good": 40,  "moderate": 80,  "poor": 180, "severe": 280}},
        "co":    {"weight": 0.10, "thresholds": {"good": 1,   "moderate": 2,   "poor": 10,  "severe": 17}},
        "o3":    {"weight": 0.05, "thresholds": {"good": 50,  "moderate": 100, "poor": 168, "severe": 208}},
    }

    @staticmethod
    def score_parameter(value: float, thresholds: Dict[str, float]) -> float:
        """Map pollutant value to a risk score (0-10) using linear interpolation within bands.

        Args:
            value: Pollutant value.
            thresholds: Dict with keys good, moderate, poor, severe.

        Returns:
            float: Risk score (0.0–10.0)
        """
        bands = [
            (0, thresholds["good"], 0, 2),
            (thresholds["good"], thresholds["moderate"], 2, 5),
            (thresholds["moderate"], thresholds["poor"], 5, 8),
            (thresholds["poor"], thresholds["severe"], 8, 10),
            (thresholds["severe"], float("inf"), 10, 10),
        ]
        for low, high, score_low, score_high in bands:
            if value <= high:
                if high == low:
                    return float(score_high)
                return score_low + (score_high - score_low) * (value - low) / (high - low)
        return 10.0

    def compute_factory_risk(
        self,
        factory_row: pd.Series,
        pollution_row: pd.Series
    ) -> Dict[str, Any]:
        """Compute risk scores for a single factory and pollution row.

        Args:
            factory_row: Row from factories DataFrame.
            pollution_row: Row from pollution DataFrame.

        Returns:
            dict: Risk scores and metadata.
        """
        scores = {}
        for pollutant, meta in self.risk_weights.items():
            value = pollution_row.get(pollutant, np.nan)
            if pd.isna(value):
                scores[f"{pollutant}_score"] = np.nan
            else:
                scores[f"{pollutant}_score"] = self.score_parameter(value, meta["thresholds"])
        # Composite score (ignore NaNs)
        composite = 0.0
        total_weight = 0.0
        for pollutant, meta in self.risk_weights.items():
            s = scores.get(f"{pollutant}_score", np.nan)
            if not pd.isna(s):
                composite += meta["weight"] * s
                total_weight += meta["weight"]
        composite_score = composite / total_weight if total_weight > 0 else np.nan
        # Risk level
        if composite_score < 3:
            risk_level = "Low"
        elif composite_score < 6:
            risk_level = "Medium"
        elif composite_score < 8:
            risk_level = "High"
        else:
            risk_level = "Critical"
        # Dominant pollutant
        dom_pollutant = max(
            ((p, scores.get(f"{p}_score", -1)) for p in self.risk_weights),
            key=lambda x: x[1] if not pd.isna(x[1]) else -1
        )[0]
        return {
            "factory_id": factory_row.get("factory_id"),
            "factory_name": factory_row.get("factory_name"),
            "industry_type": factory_row.get("industry_type"),
            "city": factory_row.get("city"),
            **scores,
            "composite_score": composite_score,
            "risk_level": risk_level,
            "dominant_pollutant": dom_pollutant,
        }

    def score_all_factories(
        self,
        factories_df: pd.DataFrame,
        pollution_df: pd.DataFrame,
        max_station_distance_km: float = 100.0
    ) -> pd.DataFrame:
        """Score all factories by joining with nearest pollution station or fallback.

        Args:
            factories_df: DataFrame of factories.
            pollution_df: DataFrame of pollution readings (must have nearest_factory_distance_km).
            max_station_distance_km: Max distance for direct match.

        Returns:
            pd.DataFrame: Risk scores for all factories.
        """
        results = []
        fallback_city = 0
        fallback_national = 0
        for _, factory in factories_df.iterrows():
            # Find pollution rows within max_station_distance_km
            mask = (
                (pollution_df["nearest_factory_distance_km"] <= max_station_distance_km) &
                (pollution_df["city"] == factory["city"])
            )
            candidates = pollution_df[mask]
            if not candidates.empty:
                pollution_row = candidates.iloc[0]
            else:
                # Fallback: city average
                city_rows = pollution_df[pollution_df["city"] == factory["city"]]
                if not city_rows.empty:
                    pollution_row = city_rows.mean(numeric_only=True)
                    fallback_city += 1
                else:
                    # Fallback: national average
                    pollution_row = pollution_df.mean(numeric_only=True)
                    fallback_national += 1
            result = self.compute_factory_risk(factory, pollution_row)
            results.append(result)
        LOGGER.info(
            "Risk scoring complete: %d factories, %d city fallback, %d national fallback",
            len(factories_df), fallback_city, fallback_national
        )
        return pd.DataFrame(results)
