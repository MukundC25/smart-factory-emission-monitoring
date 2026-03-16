"""End-to-end factory location data pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common import get_project_root, initialize_environment
from src.ingestion.factory_collector import OverpassFactoryCollector, TARGET_CITIES
from src.ingestion.factory_data_cleaner import FactoryDataCleaner
from src.ingestion.factory_processor import FactoryProcessor
from src.ingestion.synthetic_factory_generator import SyntheticFactoryGenerator

LOGGER = logging.getLogger(__name__)


def _resolve_path(config: Dict[str, Any], key: str, default_value: str) -> Path:
    return get_project_root() / str(config.get("paths", {}).get(key, default_value))


def _required_cities(config: Dict[str, Any]) -> List[str]:
    cities = config.get("factory_pipeline", {}).get("target_cities")
    if cities:
        return [str(city) for city in cities]
    return list(TARGET_CITIES)


def run_factory_pipeline(config: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Run collection, cleaning, processing, and persistence pipeline."""
    runtime_config = config or initialize_environment()
    pipeline_cfg = runtime_config.get("factory_pipeline", {})

    raw_path = _resolve_path(runtime_config, "factories_raw", "data/raw/factories/factories_raw.csv")
    clean_path = _resolve_path(runtime_config, "factories_clean", "data/raw/factories/factories.csv")
    processed_path = _resolve_path(runtime_config, "factories_processed", "data/processed/factories_processed.csv")

    collector = OverpassFactoryCollector(runtime_config)
    cities = _required_cities(runtime_config)

    raw_df = collector.collect_all(cities)

    min_threshold = int(pipeline_cfg.get("min_factories_threshold", 50))
    synthetic_fallback = bool(pipeline_cfg.get("synthetic_fallback", True))
    if raw_df.empty and synthetic_fallback:
        LOGGER.warning("Overpass data collection failed; switching to synthetic fallback dataset")
        num_cities = max(len(cities), 1)
        synthetic_df = SyntheticFactoryGenerator().generate(
            n_per_city=max(4, int(100 / num_cities))
        )
        synthetic_df = synthetic_df[synthetic_df["city"].isin(cities)]
        raw_df = synthetic_df[["osm_id", "factory_name", "industry_type", "latitude", "longitude", "city"]].copy()
        raw_df["raw_tags"] = "{}"

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(raw_path, index=False)
    LOGGER.info("Raw data saved: %s records", len(raw_df))

    cleaner = FactoryDataCleaner()
    cleaned_df = cleaner.clean(raw_df)
    if cleaned_df.empty and synthetic_fallback:
        LOGGER.warning("Cleaned data is empty after cleaning; generating synthetic fallback dataset")
        num_cities = max(len(cities), 1)
        generated = SyntheticFactoryGenerator().generate(
            n_per_city=max(4, int(100 / num_cities))
        )
        generated = generated[generated["city"].isin(cities)]
        cleaned_df = generated[
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
        ].copy()

    if len(cleaned_df) < min_threshold and synthetic_fallback:
        LOGGER.warning("Collected rows (%s) below threshold (%s); augmenting with synthetic data", len(cleaned_df), min_threshold)
        num_cities = max(len(cities), 1)
        n_per_city = max(4, int((min_threshold - len(cleaned_df)) / num_cities) + 1)
        generated = SyntheticFactoryGenerator().generate(n_per_city=n_per_city)
        generated = generated[generated["city"].isin(cities)]
        generated_clean = generated[
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
        cleaned_df = pd.concat([cleaned_df, generated_clean], ignore_index=True)
        cleaned_df = cleaned_df.drop_duplicates(subset=["factory_id"], keep="first")

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(clean_path, index=False)
    LOGGER.info("Cleaned data saved: %s records", len(cleaned_df))

    processor = FactoryProcessor(
        dbscan_eps=float(pipeline_cfg.get("dbscan_eps", 0.05)),
        dbscan_min_samples=int(pipeline_cfg.get("dbscan_min_samples", 2)),
    )
    processed_df = processor.process(cleaned_df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    LOGGER.info("Processed data saved: %s records", len(processed_df))

    industry_types = sorted({str(value) for value in processed_df.get("industry_type", pd.Series(dtype=str)).dropna().tolist()})
    LOGGER.info("Pipeline complete")
    LOGGER.info("Raw records collected: %s", len(raw_df))
    LOGGER.info("After cleaning: %s", len(cleaned_df))
    LOGGER.info("Processed records: %s", len(processed_df))
    LOGGER.info(
        "Cities covered: %s",
        processed_df.get("city", pd.Series(dtype=str)).nunique() if not processed_df.empty else 0,
    )
    LOGGER.info("Industry types found: %s", industry_types)
    LOGGER.info("Output: %s", clean_path.as_posix())

    return processed_df


def main() -> None:
    """CLI entry point for module and script execution."""
    result = run_factory_pipeline()
    print("✅ Pipeline complete")
    print(f"🏭 Processed records: {len(result)}")
    print("📁 Check data/raw/factories/factories.csv")


if __name__ == "__main__":
    main()
