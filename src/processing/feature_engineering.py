"""Feature engineering and ML dataset preparation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import get_project_root, haversine_distance_km, initialize_environment

LOGGER = logging.getLogger(__name__)

INDUSTRY_RISK_MAP = {
    "steel": 9,
    "cement": 8,
    "chemical": 10,
    "textile": 6,
    "automotive": 7,
    "electronics": 4,
    "pharmaceutical": 6,
    "food_processing": 5,
    "general_industry": 6,
    "industrial": 7,
    "works": 6,
    "factory": 7,
}


def _load_inputs(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load factory and pollution datasets.

    Args:
        config: Runtime configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Factories and pollution dataframes.
    """
    root = get_project_root()
    factories = pd.read_csv(root / config["paths"]["factories_raw"])
    pollution = pd.read_csv(root / config["paths"]["pollution_raw"])
    pollution["timestamp"] = pd.to_datetime(pollution["timestamp"], errors="coerce", utc=True)
    pollution = pollution.dropna(subset=["timestamp"]).copy()
    return factories, pollution


def _nearest_station_join(factories: pd.DataFrame, pollution: pd.DataFrame) -> pd.DataFrame:
    """Join each pollution row with nearest factory and distance.

    Args:
        factories: Factory dataset.
        pollution: Pollution dataset.

    Returns:
        pd.DataFrame: Spatially joined dataframe.
    """
    joined_rows = []
    for _, pollution_row in pollution.iterrows():
        station_lat = float(pollution_row["station_lat"])
        station_lon = float(pollution_row["station_lon"])

        best_factory = None
        best_distance = float("inf")
        for _, factory_row in factories.iterrows():
            distance = haversine_distance_km(
                station_lat,
                station_lon,
                float(factory_row["latitude"]),
                float(factory_row["longitude"]),
            )
            if distance < best_distance:
                best_distance = distance
                best_factory = factory_row

        if best_factory is None:
            continue

        combined = {
            **pollution_row.to_dict(),
            **best_factory.to_dict(),
            "distance_to_nearest_station": float(best_distance),
        }
        joined_rows.append(combined)
    return pd.DataFrame(joined_rows)


def _add_temporal_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create rolling, season, and spike features.

    Args:
        dataset: Joined pollution-factory dataframe.

    Returns:
        pd.DataFrame: Feature-enhanced dataset.
    """
    engineered = dataset.copy()
    engineered = engineered.sort_values(["factory_id", "timestamp"]).reset_index(drop=True)

    engineered["rolling_avg_pm25_7d"] = (
        engineered.groupby("factory_id")["pm25"].transform(
            lambda series: series.rolling(window=7, min_periods=1).mean()
        )
    )
    engineered["rolling_avg_pm25_30d"] = (
        engineered.groupby("factory_id")["pm25"].transform(
            lambda series: series.rolling(window=30, min_periods=1).mean()
        )
    )

    pm25_mean = engineered["pm25"].mean()
    pm25_std = engineered["pm25"].std() if engineered["pm25"].std() else 1.0
    z_score = (engineered["pm25"] - pm25_mean) / pm25_std
    engineered["pollution_spike_flag"] = (z_score > 2.5).astype(int)

    month = engineered["timestamp"].dt.month
    season = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])],
        ["winter", "summer", "monsoon", "post_monsoon"],
        default="unknown",
    )
    engineered["season"] = season

    if "wind_direction_factor" not in engineered.columns:
        engineered["wind_direction_factor"] = 1.0

    engineered["industry_risk_weight"] = engineered["industry_type"].str.lower().map(INDUSTRY_RISK_MAP).fillna(6)
    return engineered


def _build_target(dataset: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create composite pollution impact target variable.

    Args:
        dataset: Engineered dataset.
        config: Runtime configuration.

    Returns:
        pd.DataFrame: Dataset with target column.
    """
    output = dataset.copy()
    weights = config["ml"]["pollution_weights"]
    weighted_sum = (
        output["pm25"] * weights["pm25"]
        + output["pm10"] * weights["pm10"]
        + output["no2"] * weights["no2"]
        + output["so2"] * weights["so2"]
        + output["co"] * weights["co"]
        + output["o3"] * weights["o3"]
    )

    percentile_95 = np.percentile(weighted_sum, 95) if len(weighted_sum) > 0 else 1.0
    output[config["ml"]["target_column"]] = np.clip((weighted_sum / percentile_95) * 10, 0, 10)
    return output


def _add_split_column(dataset: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Assign train/validation/test split labels with stratification.

    Args:
        dataset: Final engineered dataset.
        config: Runtime configuration.

    Returns:
        pd.DataFrame: Dataset with split labels.
    """
    split_df = dataset.copy().reset_index(drop=True)
    target = split_df[config["ml"]["target_column"]]
    stratify_bins = pd.qcut(target, q=5, duplicates="drop")

    test_size = float(config["ml"]["test_size"])
    val_size = float(config["ml"]["val_size"])
    train_val_idx, test_idx = train_test_split(
        split_df.index,
        test_size=test_size,
        random_state=int(config["ml"]["random_state"]),
        stratify=stratify_bins,
    )

    train_val = split_df.loc[train_val_idx]
    val_ratio = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val.index,
        test_size=val_ratio,
        random_state=int(config["ml"]["random_state"]),
        stratify=pd.qcut(train_val[config["ml"]["target_column"]], q=5, duplicates="drop"),
    )

    split_df["split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.loc[test_idx, "split"] = "test"
    return split_df


def prepare_ml_dataset(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Prepare processed ML dataset and save as parquet.

    Args:
        config: Optional pre-loaded configuration.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    runtime_config = config or initialize_environment()
    factories, pollution = _load_inputs(runtime_config)
    joined = _nearest_station_join(factories, pollution)
    engineered = _add_temporal_features(joined)
    labeled = _build_target(engineered, runtime_config)
    split_df = _add_split_column(labeled, runtime_config)

    output_path = get_project_root() / runtime_config["paths"]["processed_dataset"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_parquet(output_path, index=False)
    LOGGER.info("Processed ML dataset written to %s with %s rows", output_path, len(split_df))
    return split_df


def main() -> None:
    """Run feature engineering module standalone."""
    config = initialize_environment()
    prepare_ml_dataset(config)


if __name__ == "__main__":
    main()
