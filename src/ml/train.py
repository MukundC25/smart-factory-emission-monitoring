"""Train pollution impact prediction models and persist artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import get_project_root, initialize_environment

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Container for model fit and metrics."""

    name: str
    pipeline: Pipeline
    rmse: float
    mae: float
    r2: float


def _load_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """Load processed ML dataset.

    Args:
        config: Runtime configuration.

    Returns:
        pd.DataFrame: Processed dataset.
    """
    path = get_project_root() / config["paths"]["processed_dataset"]
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_parquet(path)


def _build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features.

    Args:
        feature_frame: Training features frame.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    numeric_cols = [
        col
        for col in feature_frame.select_dtypes(include=[np.number]).columns
        if col not in ["pollution_impact_score"]
    ]
    categorical_cols = [
        col
        for col in feature_frame.select_dtypes(exclude=[np.number]).columns
        if col not in ["split", "timestamp", "last_updated", "source"]
    ]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def _evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    """Fit and evaluate one model pipeline.

    Args:
        name: Model name.
        pipeline: End-to-end model pipeline.
        x_train: Training features.
        y_train: Training labels.
        x_test: Test features.
        y_test: Test labels.

    Returns:
        ModelResult: Fitted model metrics.
    """
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    LOGGER.info("%s | RMSE=%.4f MAE=%.4f R2=%.4f", name, rmse, mae, r2)
    return ModelResult(name=name, pipeline=pipeline, rmse=rmse, mae=mae, r2=r2)


def _save_feature_importance(best_model: Pipeline, output_path: Any) -> None:
    """Persist feature importance plot for the selected model.

    Args:
        best_model: Fitted sklearn pipeline.
        output_path: Plot output path.
    """
    model = best_model.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    top_n = min(20, len(importances))
    top_indices = np.argsort(importances)[-top_n:]
    top_values = importances[top_indices]
    labels = [f"f_{idx}" for idx in top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, top_values)
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train and compare RandomForest and XGBoost regressors.

    Args:
        config: Optional pre-loaded configuration.

    Returns:
        Dict[str, Any]: Training summary report.
    """
    runtime_config = config or initialize_environment()
    dataset = _load_dataset(runtime_config)
    target_col = runtime_config["ml"]["target_column"]

    feature_cols: List[str] = [
        col for col in dataset.columns if col not in [target_col, "split", "timestamp", "last_updated"]
    ]

    train_df = dataset[dataset["split"] == "train"].copy()
    val_df = dataset[dataset["split"] == "val"].copy()
    test_df = dataset[dataset["split"] == "test"].copy()

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    x_train = train_val_df[feature_cols]
    y_train = train_val_df[target_col]
    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    preprocessor = _build_preprocessor(train_val_df[feature_cols])
    random_forest = RandomForestRegressor(
        n_estimators=int(runtime_config["ml"]["n_estimators"]),
        random_state=int(runtime_config["ml"]["random_state"]),
        n_jobs=-1,
    )

    candidates: List[tuple[str, Any]] = [("random_forest", random_forest)]
    if XGBRegressor is not None:
        xgb = XGBRegressor(
            n_estimators=350,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=int(runtime_config["ml"]["random_state"]),
            objective="reg:squarederror",
            n_jobs=-1,
        )
        candidates.append(("xgboost", xgb))

    results: List[ModelResult] = []
    for name, estimator in candidates:
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        results.append(_evaluate_model(name, pipeline, x_train, y_train, x_test, y_test))

    best = min(results, key=lambda item: item.rmse)
    root = get_project_root()
    model_path = root / runtime_config["paths"]["model"]
    scaler_path = root / runtime_config["paths"]["scaler"]
    report_path = root / runtime_config["paths"]["model_report"]
    importance_path = root / "models" / "feature_importance.png"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.pipeline, model_path)

    scaler = StandardScaler()
    numeric_features = train_val_df[feature_cols].select_dtypes(include=[np.number]).copy()
    scaler.fit(numeric_features)
    joblib.dump(scaler, scaler_path)

    _save_feature_importance(best.pipeline, importance_path)

    report = {
        "selected_model": best.name,
        "metrics": {
            item.name: {"rmse": item.rmse, "mae": item.mae, "r2": item.r2}
            for item in results
        },
        "config": {
            "target_column": target_col,
            "feature_count": len(feature_cols),
            "train_rows": int(len(train_val_df)),
            "test_rows": int(len(test_df)),
        },
        "artifacts": {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "feature_importance_plot": str(importance_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    LOGGER.info("Model training complete. Best model: %s", best.name)
    return report


def main() -> None:
    """Run model training standalone."""
    config = initialize_environment()
    train_models(config)


if __name__ == "__main__":
    main()
