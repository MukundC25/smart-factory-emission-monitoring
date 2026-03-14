"""Validation and imputation utilities for pollution datasets."""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

LOGGER = logging.getLogger(__name__)


def validate_pollution_ranges(
    dataset: pd.DataFrame,
    validation_cfg: Dict[str, float],
) -> pd.DataFrame:
    """Clip pollution columns to configured valid ranges.

    Args:
        dataset: Pollution dataframe.
        validation_cfg: Config section with min and max values.

    Returns:
        pd.DataFrame: Range-validated dataframe.
    """
    ranges = {
        "pm25": (validation_cfg["pm25_min"], validation_cfg["pm25_max"]),
        "pm10": (validation_cfg["pm10_min"], validation_cfg["pm10_max"]),
        "co": (validation_cfg["co_min"], validation_cfg["co_max"]),
        "no2": (validation_cfg["no2_min"], validation_cfg["no2_max"]),
        "so2": (validation_cfg["so2_min"], validation_cfg["so2_max"]),
        "o3": (validation_cfg["o3_min"], validation_cfg["o3_max"]),
    }

    validated = dataset.copy()
    for column, (minimum, maximum) in ranges.items():
        if column in validated.columns:
            invalid_count = ((validated[column] < minimum) | (validated[column] > maximum)).sum()
            if invalid_count:
                LOGGER.warning("Clipping %s out-of-range values in %s", invalid_count, column)
            validated[column] = validated[column].clip(lower=minimum, upper=maximum)
    return validated


def impute_pollution_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Impute missing pollution values with station-level median then global median.

    Args:
        dataset: Pollution dataframe.

    Returns:
        pd.DataFrame: Imputed dataframe.
    """
    imputed = dataset.copy()
    pollution_columns: List[str] = ["pm25", "pm10", "co", "no2", "so2", "o3", "aqi_index"]

    for column in pollution_columns:
        if column not in imputed.columns:
            continue
        if "station_name" in imputed.columns:
            imputed[column] = imputed.groupby("station_name")[column].transform(
                lambda series: series.fillna(series.median())
            )
        imputed[column] = imputed[column].fillna(imputed[column].median())

    LOGGER.info(
        "Missing-value imputation complete using station median and global median fallback"
    )
    return imputed
