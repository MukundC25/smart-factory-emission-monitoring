"""Entry point to execute the full smart factory emission pipeline."""

from __future__ import annotations

import logging

from src.common import initialize_environment
from src.ingestion.factory_collector import collect_factory_data
from src.ingestion.pollution_collector import collect_pollution_data
from src.ml.train import train_models
from src.processing.feature_engineering import prepare_ml_dataset
from src.recommendations.engine import generate_recommendations
from src.visualization.dashboard import build_dashboard

LOGGER = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Run full pipeline end-to-end in required module order."""
    config = initialize_environment()
    LOGGER.info("Starting Smart Factory Emission Monitoring pipeline")

    collect_factory_data(config)
    collect_pollution_data(config)
    prepare_ml_dataset(config)
    train_models(config)
    generate_recommendations(config)
    build_dashboard(config)

    LOGGER.info("Pipeline completed successfully")


def main() -> None:
    """CLI entry point."""
    run_pipeline()


if __name__ == "__main__":
    main()
