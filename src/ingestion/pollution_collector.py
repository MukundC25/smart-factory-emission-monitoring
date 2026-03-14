"""Collect real pollution readings with a strict source-priority chain.

Source priority (configured in config.yaml ingestion.pollution_sources):
1) OpenAQ live API (primary)
2) CPCB via data.gov.in public API (secondary)
3) Kaggle offline backfill CSV (optional tertiary)
4) Synthetic fallback (dev/testing only when ALLOW_SYNTHETIC_DATA=true)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.common import get_project_root, initialize_environment, safe_request_json
from src.processing.data_validator import (
    impute_pollution_missing_values,
    validate_pollution_ranges,
)

LOGGER = logging.getLogger(__name__)

POLLUTANTS = ["pm25", "pm10", "co", "no2", "so2", "o3"]

CITY_CENTRES: Dict[str, Dict[str, float]] = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Kanpur": {"lat": 26.4499, "lon": 80.3319},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Delhi NCR": {"lat": 28.6139, "lon": 77.2090},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
}

_DATAGOV_PARAM_MAP = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "CO": "co",
    "NO2": "no2",
    "SO2": "so2",
    "OZONE": "o3",
    "O3": "o3",
}

_KAGGLE_COL_MAP = {
    "pm2.5": "pm25",
    "pm_25": "pm25",
    "pm_2_5": "pm25",
    "pm_10": "pm10",
    "ozone": "o3",
    "station": "station_name",
    "lat": "station_lat",
    "latitude": "station_lat",
    "lon": "station_lon",
    "lng": "station_lon",
    "longitude": "station_lon",
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
}


def _load_factories(config: Dict[str, Any]) -> pd.DataFrame:
    path = get_project_root() / config["paths"]["factories_raw"]
    if not path.exists():
        raise FileNotFoundError(f"Factory dataset not found at {path}")
    return pd.read_csv(path)


def _build_openaq_headers() -> Dict[str, str]:
    headers = {"accept": "application/json"}
    api_key = os.getenv("OPENAQ_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _build_datagov_headers() -> Dict[str, str]:
    headers = {"accept": "application/json"}
    api_key = os.getenv("DATAGOV_API_KEY")
    if api_key:
        headers["api-key"] = api_key
    return headers


def _date_window(lookback_days: int) -> Tuple[datetime, datetime]:
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=lookback_days)
    return date_from, date_to


def _normalise_row(values: Dict[str, Any], source: str) -> Dict[str, Any]:
    pm25 = float(values.get("pm25", np.nan))
    pm10 = float(values.get("pm10", np.nan))
    valid = [v for v in [pm25, pm10] if not np.isnan(v)]
    aqi_index = float(np.mean(valid)) if valid else np.nan
    return {
        "pm25": pm25,
        "pm10": pm10,
        "co": float(values.get("co", np.nan)),
        "no2": float(values.get("no2", np.nan)),
        "so2": float(values.get("so2", np.nan)),
        "o3": float(values.get("o3", np.nan)),
        "aqi_index": aqi_index,
        "timestamp": values["timestamp"],
        "station_name": values["station_name"],
        "station_lat": float(values.get("station_lat", np.nan)),
        "station_lon": float(values.get("station_lon", np.nan)),
        "city": values.get("city", "Unknown"),
        "country": values.get("country", "India"),
        "source": source,
    }


def _openaq_is_reachable(config: Dict[str, Any]) -> bool:
    url = f"{config['apis']['openaq_url']}/countries"
    payload = safe_request_json(
        method="GET",
        url=url,
        timeout=5,
        max_retries=1,
        backoff_base_seconds=1.0,
        rate_limit_seconds=0.0,
        params={"limit": 1},
        headers=_build_openaq_headers(),
    )
    if payload is None:
        LOGGER.warning("OpenAQ pre-flight failed; skipping OpenAQ source")
        return False
    LOGGER.info("OpenAQ pre-flight passed")
    return True


def _fetch_openaq_locations(config: Dict[str, Any], latitude: float, longitude: float) -> List[Dict[str, Any]]:
    ingestion_cfg = config["ingestion"]
    url = f"{config['apis']['openaq_url']}/locations"
    payload = safe_request_json(
        method="GET",
        url=url,
        timeout=ingestion_cfg["timeout_seconds"],
        max_retries=ingestion_cfg["max_retries"],
        backoff_base_seconds=ingestion_cfg["backoff_base_seconds"],
        rate_limit_seconds=ingestion_cfg["rate_limit_seconds"],
        params={
            "coordinates": f"{latitude},{longitude}",
            "radius": 25000,
            "limit": 25,
            "country": "IN",
        },
        headers=_build_openaq_headers(),
    )
    return payload.get("results", []) if payload else []


def _parse_openaq_measurements(
    results: List[Dict[str, Any]],
    station_meta: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    parsed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in results:
        parameter_raw = row.get("parameter")
        if isinstance(parameter_raw, dict):
            parameter = (parameter_raw.get("name") or "").lower()
        else:
            parameter = (parameter_raw or "").lower()
        if parameter not in POLLUTANTS:
            continue

        station_name = row.get("location") or (station_meta or {}).get("name", "Unknown Station")
        timestamp = row.get("date", {}).get("utc") or row.get("datetime", "")
        if not timestamp:
            period = row.get("period", {})
            timestamp = (
                period.get("datetimeFrom", {}).get("utc")
                or period.get("datetimeTo", {}).get("utc")
                or ""
            )
        if not timestamp:
            continue

        key = (station_name, timestamp)
        if key not in parsed:
            coordinates = row.get("coordinates") or (station_meta or {}).get("coordinates", {}) or {}
            country_val = row.get("country")
            if isinstance(country_val, dict):
                country_name = country_val.get("name", "India")
            elif country_val:
                country_name = str(country_val)
            elif station_meta and isinstance(station_meta.get("country"), dict):
                country_name = station_meta["country"].get("name", "India")
            else:
                country_name = "India"
            parsed[key] = {
                "station_name": station_name,
                "station_lat": coordinates.get("latitude", np.nan),
                "station_lon": coordinates.get("longitude", np.nan),
                "timestamp": timestamp,
                "city": row.get("city") or (station_meta or {}).get("locality") or "Unknown",
                "country": country_name,
            }
        value = row.get("value")
        parsed[key][parameter] = float(value) if value is not None else np.nan
    return parsed


def _fetch_station_days_openaq(
    config: Dict[str, Any],
    station: Dict[str, Any],
    date_from: datetime,
    date_to: datetime,
) -> List[Dict[str, Any]]:
    ingestion_cfg = config["ingestion"]
    station_name = station.get("name", "Unknown Station")
    sensor_ids: List[int] = []
    seen: set = set()

    for sensor in station.get("sensors", []):
        sensor_id = sensor.get("id")
        param = ((sensor.get("parameter") or {}).get("name", "")).lower()
        if sensor_id is None or param not in POLLUTANTS or param in seen:
            continue
        seen.add(param)
        sensor_ids.append(int(sensor_id))

    if not sensor_ids:
        LOGGER.debug("No relevant sensors for station %s", station_name)
        return []

    raw_results: List[Dict[str, Any]] = []
    for sensor_id in sensor_ids:
        # Faster than raw measurements for long windows.
        url = f"{config['apis']['openaq_url']}/sensors/{sensor_id}/days"
        payload = safe_request_json(
            method="GET",
            url=url,
            timeout=ingestion_cfg["timeout_seconds"],
            max_retries=ingestion_cfg["max_retries"],
            backoff_base_seconds=ingestion_cfg["backoff_base_seconds"],
            rate_limit_seconds=ingestion_cfg["rate_limit_seconds"],
            params={
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "limit": 500,
                "sort": "desc",
            },
            headers=_build_openaq_headers(),
        )
        if payload:
            raw_results.extend(payload.get("results", []))

    if not raw_results:
        return []

    parsed = _parse_openaq_measurements(raw_results, station_meta=station)
    return [_normalise_row(v, "openaq") for v in parsed.values()]


def _fetch_pollution_from_openaq(config: Dict[str, Any], city_centres: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    if not _openaq_is_reachable(config):
        return []

    lookback = int(config["ingestion"].get("pollution_lookback_days", 365))
    date_from, date_to = _date_window(lookback)
    rows: List[Dict[str, Any]] = []
    consecutive_city_failures = 0

    for city, coords in city_centres.items():
        stations = _fetch_openaq_locations(config, coords["lat"], coords["lon"])
        if not stations:
            LOGGER.warning("OpenAQ: no stations found near %s", city)
            consecutive_city_failures += 1
            if consecutive_city_failures >= 3:
                LOGGER.warning("OpenAQ: repeated city failures; aborting OpenAQ source early")
                break
            continue

        consecutive_city_failures = 0
        # Lower cap to avoid long runtimes on unstable networks.
        for station in stations[:4]:
            station_rows = _fetch_station_days_openaq(config, station, date_from, date_to)
            for row in station_rows:
                if row.get("city") in ("Unknown", "", None):
                    row["city"] = city
            rows.extend(station_rows)

    LOGGER.info("OpenAQ: collected %s rows", len(rows))
    return rows


def _fetch_cpcb_page(config: Dict[str, Any], offset: int, limit: int = 1000) -> Optional[Dict[str, Any]]:
    ingestion_cfg = config["ingestion"]
    resource_id = config["apis"]["datagov_cpcb_resource_id"]
    url = f"{config['apis']['datagov_base_url']}/{resource_id}"
    return safe_request_json(
        method="GET",
        url=url,
        timeout=ingestion_cfg["timeout_seconds"],
        max_retries=ingestion_cfg["max_retries"],
        backoff_base_seconds=ingestion_cfg["backoff_base_seconds"],
        rate_limit_seconds=ingestion_cfg["rate_limit_seconds"],
        params={
            "api-version": "2.0",
            "format": "json",
            "offset": offset,
            "limit": limit,
        },
        headers=_build_datagov_headers(),
    )


def _parse_datagov_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pollutant_id = (record.get("pollutant_id") or record.get("parameter") or "").strip().upper()
    canonical = _DATAGOV_PARAM_MAP.get(pollutant_id)
    if not canonical:
        return None

    raw_value = record.get("pollutant_avg") or record.get("value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = np.nan

    station = (record.get("station") or record.get("station_name") or "Unknown Station").strip()
    city = (record.get("city") or record.get("state") or "Unknown").strip()
    timestamp_raw = record.get("last_update") or record.get("timestamp") or ""
    try:
        timestamp = pd.to_datetime(timestamp_raw, utc=True).isoformat()
    except Exception:
        # Skip records with invalid timestamps to avoid distorting time-series data.
        return None

    return {
        "station_name": station,
        "station_lat": np.nan,
        "station_lon": np.nan,
        "timestamp": timestamp,
        "city": city,
        "country": "India",
        canonical: value,
    }


def _fetch_pollution_from_cpcb(config: Dict[str, Any], target_cities: set[str]) -> List[Dict[str, Any]]:
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}
    offset = 0
    page_size = 1000
    total_rows = 0

    while True:
        payload = _fetch_cpcb_page(config, offset=offset, limit=page_size)
        if not payload:
            break

        records = payload.get("records") or payload.get("data") or []
        if not records:
            break

        for record in records:
            parsed = _parse_datagov_record(record)
            if parsed is None:
                continue

            city = parsed.get("city", "")
            if target_cities:
                matched = any(
                    c.lower() in city.lower() or city.lower() in c.lower()
                    for c in target_cities
                )
                if not matched:
                    continue

            key = (parsed["station_name"], parsed["timestamp"])
            if key not in aggregated:
                aggregated[key] = {k: v for k, v in parsed.items() if k not in POLLUTANTS}
            for pollutant in POLLUTANTS:
                if pollutant in parsed:
                    aggregated[key][pollutant] = parsed[pollutant]

        total_rows += len(records)
        if len(records) < page_size or total_rows >= 50_000:
            break
        offset += page_size

    rows = [_normalise_row(values, "cpcb") for values in aggregated.values()]
    LOGGER.info("CPCB/data.gov.in: collected %s rows", len(rows))
    return rows


def _load_kaggle_backfill(
    config: Dict[str, Any],
    target_cities: set[str],
    existing_record_keys: set[str],
) -> List[Dict[str, Any]]:
    kaggle_rel = config["paths"].get("kaggle_backfill")
    if not kaggle_rel:
        LOGGER.info("Kaggle backfill path not configured; skipping")
        return []

    backfill_path = get_project_root() / kaggle_rel
    if not backfill_path.exists():
        LOGGER.info("Kaggle backfill not found at %s; skipping", backfill_path)
        return []

    try:
        dataset = pd.read_csv(backfill_path, low_memory=False)
    except Exception as exc:
        LOGGER.error("Kaggle backfill read failed: %s", exc)
        return []

    dataset.columns = [_KAGGLE_COL_MAP.get(c.lower().strip(), c.lower().strip()) for c in dataset.columns]
    required = {"station_name", "timestamp"}
    missing = required - set(dataset.columns)
    if missing:
        LOGGER.warning("Kaggle backfill missing required columns %s; skipping", missing)
        return []

    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], errors="coerce", utc=True)
    dataset = dataset.dropna(subset=["timestamp"])

    if "city" in dataset.columns and target_cities:
        dataset = dataset[
            dataset["city"].apply(
                lambda city: any(
                    c.lower() in str(city).lower() or str(city).lower() in c.lower()
                    for c in target_cities
                )
            )
        ]

    if existing_record_keys:
        dataset_keys = dataset.apply(
            lambda row: _build_backfill_record_key(
                station_name=str(row.get("station_name", "Unknown Station")),
                timestamp_value=row["timestamp"],
            ),
            axis=1,
        )
        dataset = dataset[~dataset_keys.isin(existing_record_keys)]

    rows: List[Dict[str, Any]] = []
    for _, row in dataset.iterrows():
        parsed: Dict[str, Any] = {
            "station_name": str(row.get("station_name", "Unknown Station")),
            "station_lat": float(row.get("station_lat", np.nan)),
            "station_lon": float(row.get("station_lon", np.nan)),
            "timestamp": row["timestamp"].isoformat(),
            "city": str(row.get("city", "Unknown")),
            "country": str(row.get("country", "India")),
        }
        for pollutant in POLLUTANTS:
            value = row.get(pollutant, np.nan)
            try:
                parsed[pollutant] = float(value)
            except (TypeError, ValueError):
                parsed[pollutant] = np.nan
        rows.append(_normalise_row(parsed, "kaggle_backfill"))

    LOGGER.info("Kaggle backfill: loaded %s rows", len(rows))
    return rows


def _synthetic_allowed() -> bool:
    return os.getenv("ALLOW_SYNTHETIC_DATA", "false").lower() in {"1", "true", "yes"}


def _build_backfill_record_key(station_name: str, timestamp_value: Any) -> str:
    """Build a stable station+timestamp key (seconds precision) for backfill suppression."""
    normalized_station = station_name.strip().lower()
    parsed_timestamp = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
    if pd.isna(parsed_timestamp):
        return ""
    return f"{normalized_station}|{parsed_timestamp.strftime('%Y-%m-%dT%H:%M:%S')}"


def _generate_synthetic_pollution(factories: pd.DataFrame, row_count: int = 500) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    now = datetime.now(timezone.utc)
    rows: List[Dict[str, Any]] = []
    for index in range(row_count):
        factory = factories.iloc[index % len(factories)]
        station_lat = float(factory["latitude"]) + rng.normal(0, 0.05)
        station_lon = float(factory["longitude"]) + rng.normal(0, 0.05)
        timestamp = (now - timedelta(hours=index % (24 * 30))).isoformat()
        pm25 = float(np.clip(rng.normal(95, 35), 5, 350))
        pm10 = float(np.clip(rng.normal(140, 45), 10, 500))
        co = float(np.clip(rng.normal(1.6, 0.8), 0.05, 15))
        no2 = float(np.clip(rng.normal(58, 20), 5, 250))
        so2 = float(np.clip(rng.normal(24, 10), 2, 120))
        o3 = float(np.clip(rng.normal(42, 15), 5, 180))
        rows.append(
            {
                "pm25": pm25,
                "pm10": pm10,
                "co": co,
                "no2": no2,
                "so2": so2,
                "o3": o3,
                "aqi_index": float(np.mean([pm25, pm10])),
                "timestamp": timestamp,
                "station_name": f"{factory['city']} Monitoring Station {(index % 8) + 1}",
                "station_lat": station_lat,
                "station_lon": station_lon,
                "city": factory["city"],
                "country": factory.get("country", "India"),
                "source": "synthetic",
            }
        )
    return rows


def _distance_to_nearest_factory(
    pollution_df: pd.DataFrame,
    factories_df: pd.DataFrame,
    threshold_km: float,
) -> pd.DataFrame:
    if pollution_df.empty:
        return pollution_df

    def _min_haversine(slat: float, slon: float, flats: np.ndarray, flons: np.ndarray) -> float:
        if flats.size == 0:
            return float("inf")
        lat1 = np.radians(slat)
        lon1 = np.radians(slon)
        lat2 = np.radians(flats)
        lon2 = np.radians(flons)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        arc = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distances = 2 * 6371.0 * np.arctan2(np.sqrt(arc), np.sqrt(1 - arc))
        return float(np.min(distances))

    all_lats = factories_df["latitude"].astype(float).to_numpy()
    all_lons = factories_df["longitude"].astype(float).to_numpy()
    city_map: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for city, group in factories_df.groupby("city"):
        city_map[str(city)] = (
            group["latitude"].astype(float).to_numpy(),
            group["longitude"].astype(float).to_numpy(),
        )

    distances: List[float] = []
    keep: List[bool] = []
    missing_coordinate_flags: List[bool] = []
    for row in pollution_df.itertuples(index=False):
        slat = float(row.station_lat) if pd.notna(row.station_lat) else np.nan
        slon = float(row.station_lon) if pd.notna(row.station_lon) else np.nan
        if np.isnan(slat) or np.isnan(slon):
            distances.append(np.nan)
            keep.append(True)
            missing_coordinate_flags.append(True)
            continue

        city_key = str(getattr(row, "city", ""))
        flats, flons = city_map.get(city_key, (all_lats, all_lons))
        distance = _min_haversine(slat, slon, flats, flons)
        distances.append(distance)
        keep.append(distance <= threshold_km)
        missing_coordinate_flags.append(False)

    filtered = pollution_df.copy()
    filtered["nearest_factory_distance_km"] = distances
    filtered["station_coordinates_missing"] = missing_coordinate_flags
    filtered = filtered.loc[keep].reset_index(drop=True)
    LOGGER.info("Spatial filter kept %s/%s rows", len(filtered), len(pollution_df))
    return filtered


def _deduplicate(dataset: pd.DataFrame) -> pd.DataFrame:
    before = len(dataset)
    deduped = dataset.drop_duplicates(subset=["station_name", "timestamp", "source"]).reset_index(drop=True)
    LOGGER.info("Deduplication removed %s rows", before - len(deduped))
    return deduped


def collect_pollution_data(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    runtime_config = config or initialize_environment()
    factories = _load_factories(runtime_config)

    source_order = runtime_config["ingestion"].get(
        "pollution_sources", ["openaq", "cpcb", "kaggle_backfill"]
    )
    target_cities = set(runtime_config["ingestion"]["cities"])

    city_centres: Dict[str, Dict[str, float]] = {}
    for city, group in factories.groupby("city"):
        city_centres[str(city)] = {
            "lat": float(group["latitude"].mean()),
            "lon": float(group["longitude"].mean()),
        }
    for city in target_cities:
        if city not in city_centres and city in CITY_CENTRES:
            city_centres[city] = CITY_CENTRES[city]

    rows: List[Dict[str, Any]] = []
    covered_record_keys: set[str] = set()

    for source in source_order:
        if source == "openaq":
            source_rows = _fetch_pollution_from_openaq(runtime_config, city_centres)
        elif source == "cpcb":
            source_rows = _fetch_pollution_from_cpcb(runtime_config, target_cities)
        elif source == "kaggle_backfill":
            source_rows = _load_kaggle_backfill(runtime_config, target_cities, covered_record_keys)
        else:
            LOGGER.warning("Unknown pollution source %s; skipping", source)
            continue

        if source_rows:
            rows.extend(source_rows)
            for item in source_rows:
                record_key = _build_backfill_record_key(
                    station_name=str(item.get("station_name", "Unknown Station")),
                    timestamp_value=item.get("timestamp", ""),
                )
                if record_key:
                    covered_record_keys.add(record_key)
            LOGGER.info("Source %s contributed %s rows (total %s)", source, len(source_rows), len(rows))

    if not rows:
        if _synthetic_allowed():
            LOGGER.warning("Real sources empty; ALLOW_SYNTHETIC_DATA=true so generating synthetic fallback")
            rows = _generate_synthetic_pollution(factories)
        else:
            raise RuntimeError(
                "No real pollution data available from OpenAQ/CPCB/Kaggle and synthetic fallback is disabled."
            )

    pollution_df = pd.DataFrame(rows)
    pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"], errors="coerce", utc=True)
    pollution_df = pollution_df.dropna(subset=["timestamp"])
    pollution_df = _deduplicate(pollution_df)

    if len(pollution_df) > 50_000:
        pollution_df = pollution_df.sample(n=50_000, random_state=42).reset_index(drop=True)

    pollution_df = _distance_to_nearest_factory(
        pollution_df=pollution_df,
        factories_df=factories,
        threshold_km=float(runtime_config["validation"]["haversine_threshold_km"]),
    )
    pollution_df = validate_pollution_ranges(pollution_df, runtime_config["validation"])
    pollution_df = impute_pollution_missing_values(pollution_df)

    raw_path = get_project_root() / runtime_config["paths"]["pollution_raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pollution_df.to_csv(raw_path, index=False)
    LOGGER.info("Raw pollution dataset written to %s with %s rows", raw_path, len(pollution_df))

    processed_path = get_project_root() / runtime_config["paths"]["pollution_processed"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pollution_df.to_csv(processed_path, index=False)
    LOGGER.info(
        "Processed pollution dataset written to %s with %s rows",
        processed_path,
        len(pollution_df),
    )

    return pollution_df


def main() -> None:
    config = initialize_environment()
    collect_pollution_data(config)


if __name__ == "__main__":
    main()
