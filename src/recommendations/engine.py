"""Generate factory-level pollution control recommendations."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.common import get_project_root, initialize_environment

LOGGER = logging.getLogger(__name__)


def _risk_level(score: float, config: Dict[str, Any]) -> str:
    """Map score to risk category.

    Args:
        score: Pollution impact score.
        config: Runtime configuration.

    Returns:
        str: Risk category.
    """
    if score <= float(config["risk_bands"]["low_max"]):
        return "Low"
    if score <= float(config["risk_bands"]["medium_max"]):
        return "Medium"
    return "High"


def _recommendation_text(risk_level: str, industry_type: str) -> str:
    """Generate plain-English recommendation text.

    Args:
        risk_level: Risk level label.
        industry_type: Factory industry type.

    Returns:
        str: Recommendation text.
    """
    if risk_level == "High":
        return (
            f"{industry_type.title()} site is high-risk. Install SO2 scrubbers and ESP filters, "
            "upgrade baghouse maintenance cadence to weekly, and deploy continuous emissions "
            "monitoring with automated alerts."
        )
    if risk_level == "Medium":
        return (
            f"{industry_type.title()} site is medium-risk. Increase stack testing to bi-weekly, "
            "perform preventive burner tuning, and add leak-detection walkthroughs each shift."
        )
    return (
        f"{industry_type.title()} site is low-risk. Maintain compliance logs, keep monthly "
        "calibration checks, and sustain preventive maintenance schedules."
    )


def generate_recommendations(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Create factory-level risk scoring and recommendations.

    Args:
        config: Optional runtime configuration.

    Returns:
        pd.DataFrame: Recommendation dataset.
    """
    runtime_config = config or initialize_environment()
    root = get_project_root()
    processed_path = root / runtime_config["paths"]["processed_dataset"]

    dataset = pd.read_parquet(processed_path)
    target_col = runtime_config["ml"]["target_column"]

    factory_view = (
        dataset.groupby(
            ["factory_id", "factory_name", "industry_type", "latitude", "longitude", "city", "state", "country"],
            as_index=False,
        )
        .agg(
            pollution_impact_score=(target_col, "mean"),
            latest_pm25=("pm25", "mean"),
            latest_pm10=("pm10", "mean"),
        )
        .sort_values("pollution_impact_score", ascending=False)
    )

    factory_view["risk_level"] = factory_view["pollution_impact_score"].apply(
        lambda score: _risk_level(float(score), runtime_config)
    )
    factory_view["recommendation"] = factory_view.apply(
        lambda row: _recommendation_text(row["risk_level"], row["industry_type"]),
        axis=1,
    )

    output_path = root / runtime_config["paths"]["recommendations"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factory_view.to_csv(output_path, index=False)
    LOGGER.info("Recommendations written to %s (%s rows)", output_path, len(factory_view))
    return factory_view


def main() -> None:
    """Run recommendation engine standalone."""
    config = initialize_environment()
    generate_recommendations(config)


if __name__ == "__main__":
    main()
