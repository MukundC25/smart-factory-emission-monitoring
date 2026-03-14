"""Collect factory data from OpenStreetMap and optional Google Places."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.common import get_project_root, initialize_environment, safe_request_json

LOGGER = logging.getLogger(__name__)

CITY_METADATA: Dict[str, Dict[str, Any]] = {
    "Pune": {"state": "Maharashtra", "country": "India", "lat": 18.5204, "lon": 73.8567},
    "Mumbai": {"state": "Maharashtra", "country": "India", "lat": 19.0760, "lon": 72.8777},
    "Surat": {"state": "Gujarat", "country": "India", "lat": 21.1702, "lon": 72.8311},
    "Ahmedabad": {"state": "Gujarat", "country": "India", "lat": 23.0225, "lon": 72.5714},
    "Chennai": {"state": "Tamil Nadu", "country": "India", "lat": 13.0827, "lon": 80.2707},
    "Hyderabad": {"state": "Telangana", "country": "India", "lat": 17.3850, "lon": 78.4867},
    "Bengaluru": {"state": "Karnataka", "country": "India", "lat": 12.9716, "lon": 77.5946},
    "Delhi NCR": {"state": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
}

INDUSTRY_TYPES = [
    "steel",
    "cement",
    "chemical",
    "textile",
    "automotive",
    "electronics",
    "pharmaceutical",
    "food_processing",
]


def _build_overpass_query(city: str) -> str:
    """Build Overpass query for industrial entities in a city.

    Args:
        city: City name.

    Returns:
        str: Overpass query string.
    """
    city_token = city.replace('"', "")
    return f"""
[out:json][timeout:30];
area["name"="{city_token}"]["boundary"="administrative"]->.searchArea;
(
  node["amenity"="factory"](area.searchArea);
  way["amenity"="factory"](area.searchArea);
  relation["amenity"="factory"](area.searchArea);
  node["landuse"="industrial"](area.searchArea);
  way["landuse"="industrial"](area.searchArea);
  relation["landuse"="industrial"](area.searchArea);
  node["man_made"="works"](area.searchArea);
  way["man_made"="works"](area.searchArea);
  relation["man_made"="works"](area.searchArea);
);
out center tags;
""".strip()


def _extract_industry_type(tags: Dict[str, str]) -> str:
    """Infer industry type from OSM tags.

    Args:
        tags: OSM tag dictionary.

    Returns:
        str: Industry type label.
    """
    return (
        tags.get("industrial")
        or tags.get("man_made")
        or tags.get("amenity")
        or tags.get("landuse")
        or "general_industry"
    )


def _parse_osm_elements(city: str, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform OSM elements into normalized factory records.

    Args:
        city: City name.
        elements: Raw OSM elements.

    Returns:
        List[Dict[str, Any]]: Factory records.
    """
    records: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    meta = CITY_METADATA[city]
    for element in elements:
        tags = element.get("tags", {})
        latitude = element.get("lat") or element.get("center", {}).get("lat")
        longitude = element.get("lon") or element.get("center", {}).get("lon")
        if latitude is None or longitude is None:
            continue

        osm_id = str(element.get("id", ""))
        factory_name = tags.get("name") or f"{city} Industrial Site {osm_id}"
        records.append(
            {
                "factory_id": f"osm_{osm_id}",
                "factory_name": factory_name,
                "industry_type": _extract_industry_type(tags),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "city": city,
                "state": meta["state"],
                "country": meta["country"],
                "source": "overpass",
                "osm_id": osm_id,
                "last_updated": now,
            }
        )
    return records


def _fetch_factories_from_overpass(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch factory records from Overpass API for configured cities.

    Args:
        config: Runtime configuration.

    Returns:
        List[Dict[str, Any]]: Collected records.
    """
    ingestion_cfg = config["ingestion"]
    url = config["apis"]["overpass_url"]
    records: List[Dict[str, Any]] = []
    for city in ingestion_cfg["cities"]:
        query = _build_overpass_query(city)
        response = safe_request_json(
            method="POST",
            url=url,
            timeout=ingestion_cfg["timeout_seconds"],
            max_retries=ingestion_cfg["max_retries"],
            backoff_base_seconds=ingestion_cfg["backoff_base_seconds"],
            rate_limit_seconds=ingestion_cfg["rate_limit_seconds"],
            data={"data": query},
        )
        if not response:
            LOGGER.warning("No Overpass response for city: %s", city)
            continue
        elements = response.get("elements", [])
        city_records = _parse_osm_elements(city, elements)
        LOGGER.info("Collected %s OSM factory records for %s", len(city_records), city)
        records.extend(city_records)
    return records


def _fetch_factories_from_google(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch factory records from Google Places API if key is configured.

    Args:
        config: Runtime configuration.

    Returns:
        List[Dict[str, Any]]: Google-derived records.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        LOGGER.info("GOOGLE_PLACES_API_KEY not set, skipping Google Places fallback")
        return []

    ingestion_cfg = config["ingestion"]
    records: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    for city, meta in CITY_METADATA.items():
        query = f"factories in {city}"
        params = {"query": query, "key": api_key}
        response = safe_request_json(
            method="GET",
            url=base_url,
            timeout=ingestion_cfg["timeout_seconds"],
            max_retries=ingestion_cfg["max_retries"],
            backoff_base_seconds=ingestion_cfg["backoff_base_seconds"],
            rate_limit_seconds=ingestion_cfg["rate_limit_seconds"],
            params=params,
        )
        if not response:
            continue
        for place in response.get("results", []):
            geometry = place.get("geometry", {}).get("location", {})
            if "lat" not in geometry or "lng" not in geometry:
                continue
            place_id = place.get("place_id", "")
            records.append(
                {
                    "factory_id": f"gplaces_{place_id}",
                    "factory_name": place.get("name", f"{city} Factory"),
                    "industry_type": "general_industry",
                    "latitude": float(geometry["lat"]),
                    "longitude": float(geometry["lng"]),
                    "city": city,
                    "state": meta["state"],
                    "country": meta["country"],
                    "source": "google_places",
                    "osm_id": "",
                    "last_updated": now,
                }
            )
    LOGGER.info("Collected %s Google Places factory records", len(records))
    return records


def _generate_synthetic_factories(required_count: int) -> List[Dict[str, Any]]:
    """Generate synthetic factory records when real coverage is low.

    Args:
        required_count: Number of synthetic rows to generate.

    Returns:
        List[Dict[str, Any]]: Synthetic factory records.
    """
    rng = np.random.default_rng(42)
    records: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    city_names = list(CITY_METADATA.keys())
    for index in range(required_count):
        city = city_names[index % len(city_names)]
        meta = CITY_METADATA[city]
        latitude = meta["lat"] + rng.normal(0, 0.08)
        longitude = meta["lon"] + rng.normal(0, 0.08)
        industry_type = INDUSTRY_TYPES[index % len(INDUSTRY_TYPES)]
        records.append(
            {
                "factory_id": f"synthetic_{index + 1}",
                "factory_name": f"{city} {industry_type.title()} Plant {index + 1}",
                "industry_type": industry_type,
                "latitude": float(latitude),
                "longitude": float(longitude),
                "city": city,
                "state": meta["state"],
                "country": meta["country"],
                "source": "synthetic",
                "osm_id": "",
                "last_updated": now,
            }
        )
    return records


def collect_factory_data(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Collect factory data from APIs and synthetic fallback.

    Args:
        config: Optional pre-loaded config dictionary.

    Returns:
        pd.DataFrame: Factory dataset.
    """
    runtime_config = config or initialize_environment()
    records = _fetch_factories_from_overpass(runtime_config)

    min_required = int(runtime_config["ingestion"]["min_factory_records"])
    if len(records) < min_required:
        google_records = _fetch_factories_from_google(runtime_config)
        records.extend(google_records)

    if len(records) < min_required:
        synthetic_needed = min_required - len(records)
        LOGGER.warning(
            "Only %s real factory records available. Generating %s synthetic records.",
            len(records),
            synthetic_needed,
        )
        records.extend(_generate_synthetic_factories(synthetic_needed))

    factories_df = pd.DataFrame(records)
    factories_df = factories_df.drop_duplicates(subset=["factory_id"]).reset_index(drop=True)

    output_path = get_project_root() / runtime_config["paths"]["factories_raw"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factories_df.to_csv(output_path, index=False)
    LOGGER.info("Factory dataset written to %s with %s rows", output_path, len(factories_df))
    return factories_df


def main() -> None:
    """Run factory data collection module standalone."""
    config = initialize_environment()
    collect_factory_data(config)


if __name__ == "__main__":
    main()
