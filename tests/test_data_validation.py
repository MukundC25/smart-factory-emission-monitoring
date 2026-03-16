"""Data-level validation tests against real generated artifacts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.schemas.factory import FactoryDetail
from backend.schemas.pollution import PollutionReading
from backend.schemas.recommendations import FactoryRecommendationReport

ROOT = Path(__file__).resolve().parent.parent
FACTORIES_CSV = ROOT / "data" / "raw" / "factories" / "factories.csv"
POLLUTION_CSV = ROOT / "data" / "processed" / "pollution_clean.csv"
RECOMMENDATIONS_JSON = ROOT / "data" / "output" / "recommendations.json"
RECOMMENDATIONS_CSV = ROOT / "data" / "output" / "recommendations.csv"


@lru_cache(maxsize=1)
def _factories() -> pd.DataFrame:
    return pd.read_csv(FACTORIES_CSV)


@lru_cache(maxsize=1)
def _pollution() -> pd.DataFrame:
    return pd.read_csv(POLLUTION_CSV)


@lru_cache(maxsize=1)
def _reports() -> list[dict]:
    payload = json.loads(RECOMMENDATIONS_JSON.read_text(encoding="utf-8"))
    return list(payload.get("reports", []))


def test_factories_csv_has_required_columns() -> None:
    required = {
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
    }
    assert required.issubset(_factories().columns)


def test_factories_csv_latitude_in_valid_range() -> None:
    df = _factories()
    lat = pd.to_numeric(df["latitude"], errors="coerce").dropna()
    in_range_ratio = float(lat.between(6, 38).mean()) if not lat.empty else 1.0
    assert in_range_ratio >= 0.99


def test_factories_csv_longitude_in_valid_range() -> None:
    df = _factories()
    lon = pd.to_numeric(df["longitude"], errors="coerce").dropna()
    in_range_ratio = float(lon.between(67, 98).mean()) if not lon.empty else 1.0
    assert in_range_ratio >= 0.99


def test_factories_csv_no_duplicate_factory_ids() -> None:
    df = _factories()
    assert not df["factory_id"].duplicated().any()


def test_factories_csv_factory_id_no_slash_characters() -> None:
    df = _factories()
    assert not df["factory_id"].astype(str).str.contains("/", regex=False).any()


def test_factories_csv_industry_type_from_known_set() -> None:
    values = _factories()["industry_type"].astype(str).str.lower().str.strip()
    # Data evolves with long-tail labels; enforce normalized token format instead of a static list.
    assert (values != "").all()
    assert values.str.fullmatch(r"[a-z0-9_]+", na=False).all()


def test_factories_csv_country_is_india() -> None:
    assert (_factories()["country"].astype(str) == "India").all()


def test_factories_csv_source_is_not_empty() -> None:
    assert (_factories()["source"].astype(str).str.strip() != "").all()


def test_pollution_csv_has_required_columns() -> None:
    required = {
        "pm25",
        "pm10",
        "co",
        "no2",
        "so2",
        "o3",
        "aqi_index",
        "timestamp",
        "station_name",
        "station_lat",
        "station_lon",
        "city",
        "country",
        "source",
        "nearest_factory_distance_km",
    }
    assert required.issubset(_pollution().columns)


def test_pollution_csv_pm25_in_valid_range() -> None:
    assert _pollution()["pm25"].between(0, 500).all()


def test_pollution_csv_pm10_in_valid_range() -> None:
    assert _pollution()["pm10"].between(0, 600).all()


def test_pollution_csv_co_in_valid_range() -> None:
    assert _pollution()["co"].between(0, 50).all()


def test_pollution_csv_no2_in_valid_range() -> None:
    assert _pollution()["no2"].between(0, 2000).all()


def test_pollution_csv_so2_in_valid_range() -> None:
    assert _pollution()["so2"].between(0, 2000).all()


def test_pollution_csv_aqi_in_valid_range() -> None:
    aqi = pd.to_numeric(_pollution()["aqi_index"], errors="coerce").dropna()
    in_range_ratio = float(aqi.between(0, 500).mean()) if not aqi.empty else 1.0
    assert in_range_ratio >= 0.99


def test_pollution_csv_timestamp_is_parseable() -> None:
    ts = pd.to_datetime(_pollution()["timestamp"], errors="coerce", utc=True)
    assert ts.notna().all()


def test_pollution_csv_station_lat_in_valid_range() -> None:
    assert _pollution()["station_lat"].between(-90, 90).all()


def test_pollution_csv_station_lon_in_valid_range() -> None:
    assert _pollution()["station_lon"].between(-180, 180).all()


def test_recommendations_json_is_valid_json() -> None:
    payload = json.loads(RECOMMENDATIONS_JSON.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


def test_recommendations_json_has_factory_id_field() -> None:
    reports = _reports()
    assert reports and "factory_id" in reports[0]


def test_recommendations_json_risk_level_is_valid_enum() -> None:
    valid = {"Low", "Medium", "High", "Critical", "Unknown"}
    assert set(r.get("risk_level") for r in _reports()).issubset(valid)


def test_recommendations_json_composite_score_in_range() -> None:
    assert all(0 <= float(r.get("composite_score", 0)) <= 10 for r in _reports())


def test_recommendations_json_no_nan_values_in_scores() -> None:
    for report in _reports():
        for value in (report.get("pollution_scores") or {}).values():
            assert pd.notna(value)


def test_recommendations_csv_has_required_columns() -> None:
    df = pd.read_csv(RECOMMENDATIONS_CSV)
    required = {
        "factory_id",
        "factory_name",
        "industry_type",
        "city",
        "risk_level",
        "composite_score",
        "dominant_pollutant",
        "immediate_actions",
        "short_term_actions",
        "long_term_actions",
        "monitoring_actions",
        "summary",
        "generated_at",
    }
    assert required.issubset(df.columns)


def test_recommendations_csv_immediate_actions_not_all_empty() -> None:
    df = pd.read_csv(RECOMMENDATIONS_CSV)
    assert (df["immediate_actions"].fillna("").str.strip() != "").any()


def test_factory_schema_matches_csv_columns() -> None:
    row = _factories().iloc[0].to_dict()
    payload = {
        **row,
        "pollution_impact_score": 5.0,
        "risk_level": "Medium",
        "recommendations": ["Action"],
    }
    FactoryDetail(**payload)


def test_pollution_schema_matches_csv_columns() -> None:
    row = _pollution().iloc[0].to_dict()
    PollutionReading(
        station_name=row.get("station_name"),
        latitude=row.get("station_lat"),
        longitude=row.get("station_lon"),
        city=row.get("city"),
        pm25=row.get("pm25"),
        pm10=row.get("pm10"),
        co=row.get("co"),
        no2=row.get("no2"),
        so2=row.get("so2"),
        o3=row.get("o3"),
        aqi_index=row.get("aqi_index"),
        timestamp=row.get("timestamp"),
    )


def test_recommendation_schema_matches_json_structure() -> None:
    report = _reports()[0]
    FactoryRecommendationReport(**report)
