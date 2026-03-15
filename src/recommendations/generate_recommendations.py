"""Standalone runner for hybrid pollution-control recommendation generation."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Support both:
# 1) python -m src.recommendations.generate_recommendations
# 2) python src/recommendations/generate_recommendations.py
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.common import get_project_root, initialize_environment
from src.recommendations.engine import FactoryReport, HybridRecommendationEngine
from src.recommendations.formatter import RecommendationFormatter

LOGGER = logging.getLogger(__name__)


def _resolve_output_paths(config: Dict[str, Any]) -> tuple[Path, Path]:
    """Resolve CSV and JSON output paths from config.

    Args:
        config: Runtime configuration.

    Returns:
        tuple[Path, Path]: CSV and JSON output paths.
    """
    root = get_project_root()
    rec_cfg = config.get("recommendations", {})

    csv_rel = rec_cfg.get("output_csv") or config.get("paths", {}).get("recommendations")
    if not csv_rel:
        raise ValueError("CSV output path not configured in config.yaml")

    json_rel = rec_cfg.get("output_json")
    if not json_rel:
        csv_path_tmp = Path(str(csv_rel))
        json_rel = str(csv_path_tmp.with_suffix(".json"))

    return root / str(csv_rel), root / str(json_rel)


def _print_top_critical_summary(reports: List[FactoryReport]) -> None:
    """Print top 10 most critical factories summary table.

    Args:
        reports: Generated recommendation reports sorted by severity.
    """
    print("\nTop 10 Most Critical Factories")
    print("factory_name | risk_level | score | dominant_pollutant | top recommendation")
    print("-" * 120)

    for report in reports[:10]:
        top_action = report.recommendations[0].action if report.recommendations else "No recommendation"
        print(
            f"{report.factory_name} | {report.risk_level} | {report.composite_score:.2f} | "
            f"{report.dominant_pollutant} | {top_action}"
        )


def run(config: Optional[Dict[str, Any]] = None) -> int:
    """Execute end-to-end recommendation generation.

    Args:
        config: Optional runtime configuration.

    Returns:
        int: Process exit code.
    """
    overall_start = time.perf_counter()
    runtime_config = config or initialize_environment()
    root = get_project_root()

    try:
        LOGGER.info("Step 1/7: Resolving paths from config")
        factories_path = root / runtime_config["paths"]["factories_raw"]
        pollution_path = root / runtime_config["paths"]["pollution_processed"]
        csv_output_path, json_output_path = _resolve_output_paths(runtime_config)
        LOGGER.info("Factories path: %s", factories_path)
        LOGGER.info("Pollution path: %s", pollution_path)

        step_start = time.perf_counter()
        LOGGER.info("Step 2/7: Loading factories dataset")
        factories_df = pd.read_csv(factories_path)
        LOGGER.info(
            "Loaded factories dataset with %d rows in %.2fs",
            len(factories_df),
            time.perf_counter() - step_start,
        )

        step_start = time.perf_counter()
        LOGGER.info("Step 3/7: Loading processed pollution dataset")
        pollution_df = pd.read_csv(pollution_path)
        LOGGER.info(
            "Loaded pollution dataset with %d rows in %.2fs",
            len(pollution_df),
            time.perf_counter() - step_start,
        )

        if factories_df.empty:
            LOGGER.warning("Factories dataset is empty. Exiting gracefully without generation.")
            return 0
        if pollution_df.empty:
            LOGGER.warning("Pollution dataset is empty. Exiting gracefully without generation.")
            return 0

        step_start = time.perf_counter()
        LOGGER.info("Step 4/7: Initializing hybrid recommendation engine")
        engine = HybridRecommendationEngine(runtime_config)
        formatter = RecommendationFormatter(runtime_config)
        LOGGER.info("Initialized engine and formatter in %.2fs", time.perf_counter() - step_start)

        step_start = time.perf_counter()
        LOGGER.info("Step 5/7: Generating recommendations for all factories")
        reports = engine.generate_all(factories_df, pollution_df)
        LOGGER.info(
            "Generated %d reports in %.2fs",
            len(reports),
            time.perf_counter() - step_start,
        )

        if not reports:
            LOGGER.warning("No reports generated. Exiting gracefully.")
            return 0

        step_start = time.perf_counter()
        LOGGER.info("Step 6/7: Exporting CSV and JSON outputs")
        formatter.export_csv(reports, csv_output_path)
        formatter.export_json(reports, json_output_path)
        LOGGER.info(
            "Exported outputs in %.2fs (csv=%s, json=%s)",
            time.perf_counter() - step_start,
            csv_output_path,
            json_output_path,
        )

        LOGGER.info("Step 7/7: Printing top critical factory summary")
        _print_top_critical_summary(reports)

        LOGGER.info("Recommendation generation finished in %.2fs", time.perf_counter() - overall_start)
        return 0
    except FileNotFoundError as error:
        LOGGER.error("Input file not found: %s", error)
        return 1
    except KeyError as error:
        LOGGER.error("Missing required config key: %s", error)
        return 1
    except Exception:
        LOGGER.exception("Recommendation generation failed")
        return 1


def main() -> None:
    """CLI entrypoint for standalone recommendation generation."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
