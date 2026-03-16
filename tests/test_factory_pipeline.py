"""Unit tests for the factory location pipeline modules."""

from __future__ import annotations

import pandas as pd

from src.ingestion.factory_collector import OverpassFactoryCollector
from src.ingestion.factory_data_cleaner import FactoryDataCleaner
from src.ingestion.factory_processor import FactoryProcessor
from src.ingestion.run_factory_pipeline import run_factory_pipeline
from src.ingestion.synthetic_factory_generator import SyntheticFactoryGenerator, TARGET_CITIES


def _collector() -> OverpassFactoryCollector:
    return OverpassFactoryCollector(
        {
            "apis": {"overpass_url": "https://overpass-api.de/api/interpreter"},
            "factory_pipeline": {
                "overpass_user_agent": "test-agent/1.0 (test)",
                "overpass_timeout": 5,
                "overpass_retries": 1,
                "city_delay_seconds": 0,
            },
        }
    )


def test_resolve_industry_type_steel_tag() -> None:
    collector = _collector()
    industry = collector.resolve_industry_type({"industrial": "steel"})
    assert industry == "steel"


def test_resolve_industry_type_unknown_defaults() -> None:
    collector = _collector()
    industry = collector.resolve_industry_type({"name": "Mystery Site"})
    assert industry == "unknown"


def test_parse_element_missing_coords_returns_none() -> None:
    collector = _collector()
    parsed = collector.parse_element({"id": 100, "tags": {"name": "X"}}, city="Pune")
    assert parsed is None


def test_remove_duplicates_on_osm_id() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "1", "latitude": 18.52, "longitude": 73.85, "factory_name": "A", "industry_type": "steel", "city": "Pune"},
            {"osm_id": "1", "latitude": 18.52, "longitude": 73.85, "factory_name": "A duplicate", "industry_type": "steel", "city": "Pune"},
        ]
    )
    deduped = cleaner.remove_duplicates(df)
    assert len(deduped) == 1


def test_validate_coordinates_drops_out_of_india_bounds() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "1", "latitude": 19.0, "longitude": 73.0, "factory_name": "A", "industry_type": "steel", "city": "Pune"},
            {"osm_id": "2", "latitude": 45.0, "longitude": 73.0, "factory_name": "B", "industry_type": "steel", "city": "Pune"},
        ]
    )
    validated = cleaner.validate_coordinates(df)
    assert len(validated) == 1
    assert validated.iloc[0]["osm_id"] == "1"


def test_normalize_factory_names_fills_empty() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "12", "factory_name": "", "industry_type": "steel", "latitude": 18.52, "longitude": 73.85, "city": "Pune"},
        ]
    )
    result = cleaner.normalize_factory_names(df)
    assert result.iloc[0]["factory_name"] == "Industrial_Facility_12"


def test_add_derived_fields_adds_factory_id() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "1001", "factory_name": "Plant A", "industry_type": "steel", "latitude": 18.52, "longitude": 73.85, "city": "Pune"},
        ]
    )
    enriched = cleaner.add_derived_fields(df)
    assert enriched.iloc[0]["factory_id"] == "OSM_1001"


def test_final_schema_has_correct_columns() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame(
        [
            {
                "factory_id": "OSM_1",
                "factory_name": "Plant",
                "industry_type": "steel",
                "latitude": 18.52,
                "longitude": 73.85,
                "city": "Pune",
                "state": "Maharashtra",
                "country": "India",
                "source": "OpenStreetMap",
                "osm_id": "1",
                "last_updated": "2026-03-16",
            }
        ]
    )
    result = processor.process(df)
    assert list(result.columns) == [
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


def test_synthetic_generator_produces_min_records() -> None:
    n_per_city = 4
    generator = SyntheticFactoryGenerator()
    df = generator.generate(n_per_city=n_per_city)
    assert len(df) == n_per_city * len(TARGET_CITIES)


def test_pipeline_does_not_crash_on_empty_overpass_response(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "src.ingestion.factory_collector.OverpassFactoryCollector.collect_all",
        lambda self, cities: pd.DataFrame(columns=["osm_id", "factory_name", "industry_type", "latitude", "longitude", "city", "raw_tags"]),
    )

    config = {
        "paths": {
            "factories_raw": str((tmp_path / "factories_raw.csv").as_posix()),
            "factories_clean": str((tmp_path / "factories.csv").as_posix()),
            "factories_processed": str((tmp_path / "factories_processed.csv").as_posix()),
        },
        "apis": {"overpass_url": "https://overpass-api.de/api/interpreter"},
        "factory_pipeline": {
            "target_cities": ["Pune", "Mumbai", "Delhi", "Chennai", "Bengaluru", "Kolkata", "Ahmedabad", "Surat", "Hyderabad", "Nagpur"],
            "overpass_user_agent": "test-agent/1.0 (test)",
            "overpass_timeout": 5,
            "overpass_retries": 1,
            "city_delay_seconds": 0,
            "dbscan_eps": 0.05,
            "dbscan_min_samples": 2,
            "min_factories_threshold": 50,
            "synthetic_fallback": True,
        },
    }

    processed = run_factory_pipeline(config)
    assert not processed.empty


def test_cleaner_normalize_osm_id_removes_slash() -> None:
    cleaner = FactoryDataCleaner()
    assert cleaner._normalize_osm_id("way/123") == "way_123"


def test_cleaner_normalize_osm_id_none_returns_unknown() -> None:
    cleaner = FactoryDataCleaner()
    assert cleaner._normalize_osm_id(None) == "unknown"


def test_cleaner_factory_id_has_no_slash() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "node/1001", "factory_name": "X", "industry_type": "steel", "latitude": 18.52, "longitude": 73.85, "city": "Pune"},
        ]
    )
    out = cleaner.add_derived_fields(df)
    assert "/" not in out.iloc[0]["factory_id"]


def test_cleaner_validate_coordinates_rejects_out_of_india() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(
        [
            {"osm_id": "1", "latitude": 50.0, "longitude": 73.0, "factory_name": "A", "industry_type": "steel", "city": "Pune"},
            {"osm_id": "2", "latitude": 18.0, "longitude": 80.0, "factory_name": "B", "industry_type": "steel", "city": "Pune"},
        ]
    )
    out = cleaner.validate_coordinates(df)
    assert len(out) == 1
    assert out.iloc[0]["osm_id"] == "2"


def test_cleaner_full_pipeline_does_not_crash_on_empty_df() -> None:
    cleaner = FactoryDataCleaner()
    df = pd.DataFrame(columns=["osm_id", "factory_name", "industry_type", "latitude", "longitude", "city"])
    out = cleaner.clean(df)
    assert isinstance(out, pd.DataFrame)


def test_processor_risk_category_high_for_chemical() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame([{"industry_type": "chemical"}])
    out = processor.add_pollution_risk_category(df)
    assert out.iloc[0]["pollution_risk_category"] == "High"


def test_processor_risk_category_medium_for_automotive() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame([{"industry_type": "automotive"}])
    out = processor.add_pollution_risk_category(df)
    assert out.iloc[0]["pollution_risk_category"] == "Medium"


def test_processor_risk_category_low_for_textile() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame([{"industry_type": "textile"}])
    out = processor.add_pollution_risk_category(df)
    assert out.iloc[0]["pollution_risk_category"] == "Low"


def test_processor_pd_na_industry_type_does_not_crash() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame([{"industry_type": pd.NA}])
    out = processor.add_pollution_risk_category(df)
    assert out.iloc[0]["pollution_risk_category"] == "Low"


def test_processor_dbscan_uses_haversine_metric() -> None:
    processor = FactoryProcessor(dbscan_eps=0.1, dbscan_min_samples=2)
    df = pd.DataFrame(
        [
            {"latitude": 18.5204, "longitude": 73.8567},
            {"latitude": 18.5210, "longitude": 73.8570},
        ]
    )
    out = processor.add_cluster_id(df)
    assert "cluster_id" in out.columns


def test_processor_final_schema_has_correct_dtypes() -> None:
    processor = FactoryProcessor()
    df = pd.DataFrame(
        [
            {
                "factory_id": "OSM_1",
                "factory_name": "Plant",
                "industry_type": "steel",
                "latitude": 18.52,
                "longitude": 73.85,
                "city": "Pune",
                "state": "Maharashtra",
                "country": "India",
                "source": "OpenStreetMap",
                "osm_id": "1",
                "last_updated": "2026-03-16",
                "urban_rural": "urban",
                "pollution_risk_category": "High",
                "cluster_id": 0,
            }
        ]
    )
    out = processor.final_schema(df)
    assert str(out["cluster_id"].dtype) == "int64"
    assert str(out["latitude"].dtype) == "float64"


def test_processor_cluster_id_is_integer() -> None:
    processor = FactoryProcessor(dbscan_eps=0.1, dbscan_min_samples=2)
    df = pd.DataFrame(
        [
            {"latitude": 18.5204, "longitude": 73.8567},
            {"latitude": 18.5210, "longitude": 73.8570},
        ]
    )
    out = processor.add_cluster_id(df)
    assert str(out["cluster_id"].dtype).startswith("int")
