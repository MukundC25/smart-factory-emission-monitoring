"""Prediction utility for the trained pollution impact model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.common import get_project_root, initialize_environment

LOGGER = logging.getLogger(__name__)


def predict_impact_scores(
    input_frame: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate pollution impact predictions for input records.

    Args:
        input_frame: Input feature dataframe.
        config: Optional runtime configuration.

    Returns:
        pd.DataFrame: Input with prediction column.
    """
    runtime_config = config or initialize_environment()
    model_path = get_project_root() / runtime_config["paths"]["model"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = joblib.load(model_path)
    output = input_frame.copy()
    output["predicted_pollution_impact_score"] = model.predict(input_frame)
    return output


def predict_from_file(input_path: Path, output_path: Path) -> None:
    """Run predictions from CSV or Parquet file and save CSV output.

    Args:
        input_path: Input dataset path.
        output_path: Output CSV path.
    """
    config = initialize_environment()
    if input_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
    else:
        raise ValueError("Unsupported input format. Use .csv or .parquet")

    drop_cols = [
        col for col in ["split", "timestamp", "last_updated", config["ml"]["target_column"]] if col in frame.columns
    ]
    features = frame.drop(columns=drop_cols)
    predicted = predict_impact_scores(features, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_path, index=False)

    metadata = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(predicted),
    }
    LOGGER.info("Prediction run complete: %s", json.dumps(metadata))


def main() -> None:
    """Execute prediction utility using processed dataset as default input."""
    config = initialize_environment()
    root = get_project_root()
    input_path = root / config["paths"]["processed_dataset"]
    output_path = root / "data" / "output" / "predictions.csv"
    predict_from_file(input_path, output_path)


if __name__ == "__main__":
    main()
