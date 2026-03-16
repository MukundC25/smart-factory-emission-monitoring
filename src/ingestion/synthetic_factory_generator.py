"""Synthetic fallback factory generator for API outage scenarios."""

from __future__ import annotations

import logging
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.ingestion.factory_collector import CITY_COORDINATES, CITY_TO_STATE, TARGET_CITIES

LOGGER = logging.getLogger(__name__)

CITY_INDUSTRY_DISTRIBUTION: Dict[str, List[Tuple[str, float]]] = {
    "Pune": [("automotive", 0.30), ("it_hardware", 0.20), ("manufacturing", 0.50)],
    "Mumbai": [("chemical", 0.25), ("pharmaceutical", 0.20), ("textile", 0.30), ("food_processing", 0.25)],
    "Surat": [("textile", 0.60), ("chemical", 0.20), ("diamond", 0.20)],
    "Ahmedabad": [("textile", 0.30), ("chemical", 0.30), ("pharmaceutical", 0.20), ("automotive", 0.20)],
    "Chennai": [("automotive", 0.35), ("manufacturing", 0.35), ("electronics", 0.30)],
    "Hyderabad": [("pharmaceutical", 0.35), ("electronics", 0.20), ("chemical", 0.20), ("manufacturing", 0.25)],
    "Bengaluru": [("electronics", 0.35), ("it_hardware", 0.35), ("manufacturing", 0.30)],
    "Delhi": [("manufacturing", 0.35), ("automotive", 0.25), ("food_processing", 0.20), ("chemical", 0.20)],
    "Kolkata": [("steel", 0.25), ("chemical", 0.25), ("textile", 0.25), ("manufacturing", 0.25)],
    "Nagpur": [("power", 0.30), ("cement", 0.25), ("manufacturing", 0.45)],
}

DEFAULT_MIX: List[Tuple[str, float]] = [
    ("manufacturing", 0.35),
    ("general_industrial", 0.25),
    ("automotive", 0.15),
    ("chemical", 0.10),
    ("textile", 0.10),
    ("food_processing", 0.05),
]


class SyntheticFactoryGenerator:
    """Generate realistic synthetic factory records for fallback mode."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def _sample_industry_type(self, city: str) -> str:
        distribution = CITY_INDUSTRY_DISTRIBUTION.get(city, DEFAULT_MIX)
        labels = [item[0] for item in distribution]
        weights = np.array([item[1] for item in distribution], dtype=float)
        weights = weights / weights.sum()
        return str(self.rng.choice(labels, p=weights))

    def generate(self, n_per_city: int = 20) -> pd.DataFrame:
        """Generate synthetic factories for all target cities."""
        rows: List[Dict[str, object]] = []
        today = date.today().isoformat()

        synthetic_index = 1
        for city in TARGET_CITIES:
            center = CITY_COORDINATES.get(city)
            if center is None:
                continue

            for _ in range(n_per_city):
                lat_offset = float(self.rng.uniform(-0.05, 0.05))
                lon_offset = float(self.rng.uniform(-0.05, 0.05))
                lat = float(center[0] + lat_offset)
                lon = float(center[1] + lon_offset)
                industry = self._sample_industry_type(city)
                osm_id = f"SYN_{city[:3].upper()}_{synthetic_index:06d}"

                rows.append(
                    {
                        "factory_id": f"OSM_{osm_id}",
                        "factory_name": f"{city} {industry.replace('_', ' ').title()} Facility {synthetic_index}",
                        "industry_type": industry,
                        "latitude": lat,
                        "longitude": lon,
                        "city": city,
                        "state": CITY_TO_STATE.get(city, "Unknown"),
                        "country": "India",
                        "source": "OpenStreetMap",
                        "osm_id": osm_id,
                        "last_updated": today,
                        "urban_rural": "peri-urban",
                        "pollution_risk_category": "Medium",
                        "cluster_id": -1,
                    }
                )
                synthetic_index += 1

        df = pd.DataFrame(rows)
        LOGGER.warning("Generated %s synthetic factories as API fallback", len(df))
        return df
