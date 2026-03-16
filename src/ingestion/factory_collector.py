"""Factory collection from OpenStreetMap Overpass API."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

from src.common import get_project_root, initialize_environment

LOGGER = logging.getLogger(__name__)

TARGET_CITIES: List[str] = [
    "Pune",
    "Mumbai",
    "Nagpur",
    "Nashik",
    "Aurangabad",
    "Surat",
    "Ahmedabad",
    "Vadodara",
    "Rajkot",
    "Chennai",
    "Coimbatore",
    "Madurai",
    "Hyderabad",
    "Visakhapatnam",
    "Bengaluru",
    "Mangaluru",
    "Delhi",
    "Noida",
    "Gurgaon",
    "Faridabad",
    "Kolkata",
    "Howrah",
    "Jaipur",
    "Jodhpur",
    "Bhopal",
    "Indore",
    "Lucknow",
    "Kanpur",
]

CITY_COORDINATES: Dict[str, Tuple[float, float]] = {
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Surat": (21.1702, 72.8311),
    "Ahmedabad": (23.0225, 72.5714),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Hyderabad": (17.3850, 78.4867),
    "Visakhapatnam": (17.6868, 83.2185),
    "Bengaluru": (12.9716, 77.5946),
    "Mangaluru": (12.9141, 74.8560),
    "Delhi": (28.6139, 77.2090),
    "Noida": (28.5355, 77.3910),
    "Gurgaon": (28.4595, 77.0266),
    "Faridabad": (28.4089, 77.3178),
    "Kolkata": (22.5726, 88.3639),
    "Howrah": (22.5958, 88.2636),
    "Jaipur": (26.9124, 75.7873),
    "Jodhpur": (26.2389, 73.0243),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
}


@dataclass(frozen=True)
class QuerySpec:
    name: str
    filter_expr: str


QUERY_SPECS: List[QuerySpec] = [
    QuerySpec("man_made_works", '["man_made"="works"]'),
    QuerySpec("landuse_industrial", '["landuse"="industrial"]'),
    QuerySpec("building_industrial", '["building"="industrial"]'),
    QuerySpec("amenity_factory", '["amenity"="factory"]'),
    QuerySpec("industrial_factory", '["industrial"="factory"]'),
]


class OverpassFactoryCollector:
    """Collect and parse factory-like OSM entities from Overpass."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        runtime_config = config or initialize_environment()
        pipeline_cfg = runtime_config.get("factory_pipeline", {})
        self.config = runtime_config
        self.overpass_url = pipeline_cfg.get(
            "overpass_url", runtime_config.get("apis", {}).get("overpass_url", "https://overpass-api.de/api/interpreter")
        )
        self.timeout = int(pipeline_cfg.get("overpass_timeout", 60))
        self.retries = int(pipeline_cfg.get("overpass_retries", 3))
        self.city_delay_seconds = float(pipeline_cfg.get("city_delay_seconds", 2))
        self.city_cache: Dict[str, Tuple[float, float]] = {}
        self._last_geocode_time = 0.0
        self._last_context: Tuple[str, str] = ("", "")
        self.user_agent = (
            pipeline_cfg.get("overpass_user_agent")
            or runtime_config.get("apis", {}).get("overpass_user_agent")
        )
        if not self.user_agent:
            raise ValueError(
                "Overpass user agent must be set via factory_pipeline.overpass_user_agent "
                "in config.yaml with a valid contact email or URL to comply with "
                "Overpass/Nominatim usage policies."
            )
        nominatim_user_agent = pipeline_cfg.get("nominatim_user_agent", self.user_agent)
        self._geocoder = Nominatim(user_agent=nominatim_user_agent)

    @staticmethod
    def _build_osm_id(osm_type: Any, osm_local_id: Any) -> str:
        local_id_str = "" if osm_local_id is None else str(osm_local_id)
        if not local_id_str:
            return local_id_str
        type_str = "" if osm_type is None else str(osm_type).strip()
        if not type_str:
            return local_id_str
        safe_type = type_str.replace("/", "_")
        return f"{safe_type}_{local_id_str}"

    def build_overpass_query(self, city: str, query_type: str, radius_km: int = 25) -> str:
        """Build Overpass QL query for one city and one query type."""
        coords = self.geocode_city(city)
        if coords is None:
            LOGGER.error("Skipping query — no coordinates for city: %s", city)
            return ""
        lat, lon = coords
        radius_m = radius_km * 1000
        return (
            f"[out:json][timeout:{self.timeout}];\n"
            "(\n"
            f"  node{query_type}(around:{radius_m},{lat},{lon});\n"
            f"  way{query_type}(around:{radius_m},{lat},{lon});\n"
            f"  relation{query_type}(around:{radius_m},{lat},{lon});\n"
            ");\n"
            "out center tags;"
        )

    def geocode_city(self, city: str) -> Optional[Tuple[float, float]]:
        """Resolve city coordinates via cache, Nominatim, then hardcoded fallback."""
        if city in self.city_cache:
            return self.city_cache[city]

        # Check hardcoded map first to avoid unnecessary external API calls.
        coords = CITY_COORDINATES.get(city)
        if coords is not None:
            self.city_cache[city] = coords
            return coords

        now = time.time()
        elapsed = now - self._last_geocode_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        try:
            location = self._geocoder.geocode(f"{city}, India", timeout=10)
            self._last_geocode_time = time.time()
            if location is not None:
                coords = (float(location.latitude), float(location.longitude))
                self.city_cache[city] = coords
                return coords
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            LOGGER.warning("Geocode failed for %s: %s", city, exc)

        LOGGER.error("No coordinates available for city: %s", city)
        return None

    def fetch_overpass(self, query: str, retries: int = 3) -> Dict[str, Any]:
        """Execute Overpass query with retry/backoff behavior."""
        if not query or not query.strip():
            LOGGER.warning("Empty query received — skipping Overpass request")
            return {"elements": []}
        city, query_name = self._last_context
        max_retries = max(1, retries)

        for attempt in range(max_retries):
            attempt_num = attempt + 1
            LOGGER.info(
                "Overpass request attempt %s/%s for city=%s query=%s",
                attempt_num,
                max_retries,
                city,
                query_name,
            )
            try:
                response = requests.post(
                    self.overpass_url,
                    data={"data": query},
                    timeout=self.timeout,
                    headers={"User-Agent": self.user_agent},
                )
                if response.status_code == 429:
                    retry_after_header = response.headers.get("Retry-After")
                    try:
                        retry_after = int(retry_after_header) if retry_after_header else 30
                    except (TypeError, ValueError):
                        retry_after = 30
                    LOGGER.warning(
                        "Overpass 429 rate limit — waiting %ss before retry", retry_after
                    )
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    return {"elements": []}
                return payload
            except (requests.RequestException, ValueError) as exc:
                if attempt_num >= max_retries:
                    LOGGER.error("Overpass failed for city=%s query=%s: %s", city, query_name, exc)
                    break
                backoff_seconds = 2 ** attempt_num
                LOGGER.warning(
                    "Overpass retry for city=%s query=%s after %.1fs due to %s",
                    city,
                    query_name,
                    backoff_seconds,
                    exc,
                )
                time.sleep(backoff_seconds)

        return {"elements": []}

    def collect_city(self, city: str) -> List[Dict[str, Any]]:
        """Run all query types for one city and deduplicate on OSM id."""
        collected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for spec in QUERY_SPECS:
            query = self.build_overpass_query(city=city, query_type=spec.filter_expr)
            self._last_context = (city, spec.name)
            payload = self.fetch_overpass(query=query, retries=self.retries)
            for element in payload.get("elements", []):
                osm_type = element.get("type", "")
                osm_local_id = element.get("id")
                if osm_local_id is None:
                    continue
                composite_id = self._build_osm_id(osm_type, osm_local_id)
                if composite_id in seen_ids:
                    continue
                seen_ids.add(composite_id)
                enriched = dict(element)
                enriched["_query_type"] = spec.name
                collected.append(enriched)

        LOGGER.info("City %s: found %s raw elements", city, len(collected))
        return collected

    def collect_all(self, cities: List[str]) -> pd.DataFrame:
        """Collect and parse factory-like records for a list of cities."""
        parsed_records: List[Dict[str, Any]] = []
        total = len(cities)
        for index, city in enumerate(cities, start=1):
            LOGGER.info("Processing city %s/%s: %s", index, total, city)
            raw_elements = self.collect_city(city)
            for element in raw_elements:
                parsed = self.parse_element(element, city)
                if parsed is not None:
                    parsed_records.append(parsed)
            if index < total:
                time.sleep(self.city_delay_seconds)

        if not parsed_records:
            return pd.DataFrame(
                columns=[
                    "osm_id",
                    "factory_name",
                    "industry_type",
                    "latitude",
                    "longitude",
                    "city",
                    "raw_tags",
                ]
            )
        return pd.DataFrame(parsed_records)

    def parse_element(self, element: Dict[str, Any], city: str) -> Optional[Dict[str, Any]]:
        """Convert one OSM element into a normalized record."""
        tags = element.get("tags", {}) if isinstance(element.get("tags", {}), dict) else {}
        raw_id = element.get("id", "")
        elem_type = element.get("type", "")
        osm_id = self._build_osm_id(elem_type, raw_id)

        latitude = element.get("lat")
        longitude = element.get("lon")
        if latitude is None or longitude is None:
            center = element.get("center", {})
            latitude = center.get("lat")
            longitude = center.get("lon")

        if latitude is None or longitude is None:
            return None

        factory_name = tags.get("name") or tags.get("operator") or f"Industrial_{osm_id}"
        return {
            "osm_id": osm_id,
            "factory_name": str(factory_name),
            "industry_type": self.resolve_industry_type(tags),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "city": city,
            "raw_tags": json.dumps(tags, sort_keys=True),
        }

    def resolve_industry_type(self, tags: Dict[str, Any]) -> str:
        """Map OSM tags to standardized industry type labels."""
        industrial = str(tags.get("industrial", "")).lower()
        landuse = str(tags.get("landuse", "")).lower()
        man_made = str(tags.get("man_made", "")).lower()
        amenity = str(tags.get("amenity", "")).lower()
        building = str(tags.get("building", "")).lower()
        text = " ".join([industrial, landuse, man_made, amenity, building]).strip()

        if any(token in text for token in ["steel", "metal", "smelting"]):
            return "steel"
        if any(token in text for token in ["chemical", "refinery"]):
            return "chemical"
        if any(token in text for token in ["textile", "garment"]):
            return "textile"
        if any(token in text for token in ["pharmaceutical", "medicine"]):
            return "pharmaceutical"
        if any(token in text for token in ["cement", "concrete"]):
            return "cement"
        if any(token in text for token in ["power", "energy"]):
            return "power"
        if any(token in text for token in ["food", "brewery", "dairy"]):
            return "food_processing"
        if any(token in text for token in ["auto", "automobile", "vehicle"]):
            return "automotive"
        if any(token in text for token in ["paper", "pulp"]):
            return "paper"
        if landuse == "industrial":
            return "general_industrial"
        if man_made == "works":
            return "manufacturing"
        if amenity == "factory":
            return "factory"
        return "unknown"


def _main_factory_path(config: Dict[str, Any]) -> str:
    paths = config.get("paths", {})
    return str(paths.get("factories_clean") or paths.get("factories_raw") or "data/raw/factories/factories.csv")


def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in ["factory_name", "industry_type", "city"]:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column]
                .fillna("")
                .astype(str)
                .map(lambda value: re.sub(r"\s+", " ", value).strip())
            )
    return cleaned


def collect_factory_data(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Compatibility entrypoint used by existing project scripts."""
    runtime_config = config or initialize_environment()
    collector = OverpassFactoryCollector(runtime_config)

    cities = runtime_config.get("factory_pipeline", {}).get("target_cities")
    if not cities:
        cities = runtime_config.get("ingestion", {}).get("cities") or TARGET_CITIES

    raw_df = collector.collect_all([str(city) for city in cities])
    raw_df = _sanitize_columns(raw_df)

    raw_path = get_project_root() / str(
        runtime_config.get("paths", {}).get("factories_raw", "data/raw/factories/factories_raw.csv")
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)

    # Keep legacy downstream behavior by writing the API-consumed factories file too.
    main_df = raw_df.copy()
    if not main_df.empty:
        main_df["factory_id"] = main_df["osm_id"].map(lambda value: f"OSM_{value}")
        main_df["source"] = "OpenStreetMap"
        main_df["country"] = "India"
        main_df["state"] = main_df["city"].map(lambda value: CITY_TO_STATE.get(value, "Unknown"))
        main_df["last_updated"] = date.today().isoformat()
        main_df = main_df[
            [
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
            ]
        ]

    # Local import avoids circular dependency (factory_data_cleaner imports CITY_TO_STATE from here)
    from src.ingestion.factory_data_cleaner import FactoryDataCleaner
    cleaner = FactoryDataCleaner()
    main_df = cleaner.clean(main_df)
    main_path = get_project_root() / _main_factory_path(runtime_config)
    main_path.parent.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(main_path, index=False)
    LOGGER.info("Factory datasets written: raw=%s rows, main=%s rows", len(raw_df), len(main_df))
    return main_df


CITY_TO_STATE: Dict[str, str] = {
    "Pune": "Maharashtra",
    "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Nashik": "Maharashtra",
    "Aurangabad": "Maharashtra",
    "Surat": "Gujarat",
    "Ahmedabad": "Gujarat",
    "Vadodara": "Gujarat",
    "Rajkot": "Gujarat",
    "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu",
    "Madurai": "Tamil Nadu",
    "Hyderabad": "Telangana",
    "Visakhapatnam": "Andhra Pradesh",
    "Bengaluru": "Karnataka",
    "Mangaluru": "Karnataka",
    "Delhi": "Delhi",
    "Noida": "Uttar Pradesh",
    "Gurgaon": "Haryana",
    "Faridabad": "Haryana",
    "Kolkata": "West Bengal",
    "Howrah": "West Bengal",
    "Jaipur": "Rajasthan",
    "Jodhpur": "Rajasthan",
    "Bhopal": "Madhya Pradesh",
    "Indore": "Madhya Pradesh",
    "Lucknow": "Uttar Pradesh",
    "Kanpur": "Uttar Pradesh",
}


def main() -> None:
    """Run factory collection module standalone."""
    config = initialize_environment()
    collect_factory_data(config)


if __name__ == "__main__":
    main()
