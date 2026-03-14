"""Convenience script for running raw data ingestion only."""

from src.common import initialize_environment
from src.ingestion.factory_collector import collect_factory_data
from src.ingestion.pollution_collector import collect_pollution_data


def main() -> None:
	"""Run factory and pollution collectors."""
	config = initialize_environment()
	collect_factory_data(config)
	collect_pollution_data(config)


if __name__ == "__main__":
	main()
