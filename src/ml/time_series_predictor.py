"""Time-series forecasting for pollution impact scores using scikit-learn.

Since Prophet is not available, we use a polynomial regression approach with
trend and seasonal components, plus industry/city features for multi-variate
forecasting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common import get_project_root

LOGGER = logging.getLogger(__name__)

# Industry growth rates (annual % increase in pollution potential)
INDUSTRY_GROWTH_RATES = {
    "Steel": 2.5,
    "Cement": 2.0,
    "Power": 1.8,
    "Chemical": 3.0,
    "Textile": 1.5,
    "Automotive": 2.2,
    "Refinery": 2.8,
    "Mining": 2.4,
    "Pharmaceutical": 1.2,
    "Food Processing": 1.0,
    "Paper": 1.8,
    "Electronics": 0.8,
    "Other": 1.5,
}

# City urbanization factors (pollution multiplier per year)
CITY_URBANIZATION = {
    "Mumbai": 1.03,
    "Delhi": 1.035,
    "Bangalore": 1.025,
    "Hyderabad": 1.028,
    "Chennai": 1.022,
    "Kolkata": 1.02,
    "Pune": 1.03,
    "Ahmedabad": 1.025,
    "Jaipur": 1.02,
    "Surat": 1.028,
    "Lucknow": 1.018,
    "Kanpur": 1.015,
    "Nagpur": 1.02,
    "Visakhapatnam": 1.025,
    "Vadodara": 1.022,
    "Ludhiana": 1.015,
    "Coimbatore": 1.02,
}

# Seasonal patterns (multiplier by month)
SEASONAL_MULTIPLIERS = {
    1: 1.15,  # Winter - high pollution
    2: 1.12,  # Winter
    3: 1.05,  # Spring
    4: 0.95,  # Spring
    5: 0.90,  # Summer
    6: 0.85,  # Summer - lowest
    7: 0.82,  # Monsoon - lowest
    8: 0.85,  # Monsoon
    9: 0.90,  # Post-monsoon
    10: 1.05,  # Post-monsoon
    11: 1.15,  # Winter
    12: 1.20,  # Winter - highest
}


@dataclass
class ForecastResult:
    """Container for forecast results."""

    year: int
    predicted_score: float
    confidence_lower: float
    confidence_upper: float


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    rmse: float
    mae: float
    r2: float


class PollutionForecastModel:
    """Time-series forecast model for pollution impact scores.

    Uses GradientBoosting with trend, seasonality, and feature engineering.
    """

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.metrics: Optional[ModelMetrics] = None
        self.feature_cols: List[str] = []
        self.is_trained = False

    def _generate_synthetic_historical_data(
        self,
        n_factories: int = 500,
        years: int = 5,
        start_year: int = 2020,
    ) -> pd.DataFrame:
        """Generate synthetic historical training data.

        Args:
            n_factories: Number of factories to generate.
            years: Years of historical data per factory.
            start_year: Starting year for data.

        Returns:
            pd.DataFrame: Synthetic historical data.
        """
        np.random.seed(42)
        records = []

        cities = list(CITY_URBANIZATION.keys())
        industries = list(INDUSTRY_GROWTH_RATES.keys())

        for i in range(n_factories):
            factory_id = f"F{i:04d}"
            industry = np.random.choice(industries)
            city = np.random.choice(cities)
            base_score = np.random.uniform(3.0, 8.0)
            industry_growth = INDUSTRY_GROWTH_RATES[industry] / 100
            city_growth = (CITY_URBANIZATION[city] - 1.0)

            # Generate monthly data for each year
            for year_offset in range(years):
                year = start_year + year_offset
                for month in range(1, 13):
                    # Trend component
                    trend = base_score + (year_offset * 12 + month) * (
                        industry_growth + city_growth
                    ) * 0.1

                    # Seasonal component
                    seasonal = SEASONAL_MULTIPLIERS[month]

                    # Add noise
                    noise = np.random.normal(0, 0.3)

                    # Calculate final score (0-10 range)
                    score = np.clip(trend * seasonal + noise, 0.0, 10.0)

                    # Pollutant levels correlated with score
                    pm25 = 20 + score * 15 + np.random.normal(0, 5)
                    pm10 = 40 + score * 20 + np.random.normal(0, 8)
                    co = 0.5 + score * 0.3 + np.random.normal(0, 0.1)
                    no2 = 15 + score * 8 + np.random.normal(0, 3)
                    so2 = 8 + score * 4 + np.random.normal(0, 2)
                    o3 = 20 + score * 5 + np.random.normal(0, 4)

                    records.append({
                        "factory_id": factory_id,
                        "industry_type": industry,
                        "city": city,
                        "year": year,
                        "month": month,
                        "pollution_impact_score": score,
                        "pm25": max(0, pm25),
                        "pm10": max(0, pm10),
                        "co": max(0, co),
                        "no2": max(0, no2),
                        "so2": max(0, so2),
                        "o3": max(0, o3),
                        "rolling_avg_pm25_7d": max(0, pm25 + np.random.normal(0, 3)),
                        "rolling_avg_pm25_30d": max(0, pm25 + np.random.normal(0, 2)),
                    })

        return pd.DataFrame(records)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time-series forecasting.

        Args:
            df: Input dataframe.

        Returns:
            pd.DataFrame: DataFrame with engineered features.
        """
        df = df.copy()

        # Time features
        df["time_index"] = (df["year"] - 2020) * 12 + df["month"]
        df["year_norm"] = (df["year"] - 2020) / 10.0
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Industry growth rate feature
        df["industry_growth"] = df["industry_type"].map(INDUSTRY_GROWTH_RATES).fillna(1.5)

        # City urbanization factor
        df["city_urbanization"] = df["city"].map(CITY_URBANIZATION).fillna(1.02)

        # Rolling trend features per factory
        df = df.sort_values(["factory_id", "year", "month"])
        df["score_lag_1"] = df.groupby("factory_id")["pollution_impact_score"].shift(1)
        df["score_lag_3"] = df.groupby("factory_id")["pollution_impact_score"].shift(3)
        df["score_rolling_mean_3"] = (
            df.groupby("factory_id")["pollution_impact_score"]
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        # Fill NaN values
        df["score_lag_1"] = df["score_lag_1"].fillna(df["pollution_impact_score"])
        df["score_lag_3"] = df["score_lag_3"].fillna(df["pollution_impact_score"])

        return df

    def _build_model_pipeline(self) -> Pipeline:
        """Build the forecasting model pipeline.

        Returns:
            Pipeline: Scikit-learn pipeline.
        """
        # Numeric features
        numeric_features = [
            "time_index",
            "year_norm",
            "month_sin",
            "month_cos",
            "industry_growth",
            "city_urbanization",
            "pm25",
            "pm10",
            "co",
            "no2",
            "so2",
            "o3",
            "rolling_avg_pm25_7d",
            "rolling_avg_pm25_30d",
            "score_lag_1",
            "score_lag_3",
            "score_rolling_mean_3",
        ]

        # Categorical features
        categorical_features = ["industry_type", "city"]

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )

        # GradientBoosting for time-series with non-linear trends
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        self.feature_cols = numeric_features + categorical_features
        return pipeline

    def train(
        self, output_dir: Optional[Path] = None, save_artifacts: bool = True
    ) -> Dict[str, Any]:
        """Train the forecasting model on synthetic historical data.

        Args:
            output_dir: Directory to save model artifacts.
            save_artifacts: Whether to save model files.

        Returns:
            Dict with training metrics and paths.
        """
        LOGGER.info("Generating synthetic historical data...")
        df = self._generate_synthetic_historical_data(n_factories=500, years=5)

        LOGGER.info("Engineering features...")
        df = self._engineer_features(df)

        # Split by time (last 6 months for validation)
        max_time = df["time_index"].max()
        train_df = df[df["time_index"] <= max_time - 6].copy()
        val_df = df[df["time_index"] > max_time - 6].copy()

        LOGGER.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

        # Build and train model
        self.model = self._build_model_pipeline()

        X_train = train_df[self.feature_cols]
        y_train = train_df["pollution_impact_score"]
        X_val = val_df[self.feature_cols]
        y_val = val_df["pollution_impact_score"]

        LOGGER.info("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        self.metrics = ModelMetrics(
            rmse=val_rmse,
            mae=float(np.mean(np.abs(y_val - val_pred))),
            r2=float(1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)),
        )

        LOGGER.info(f"Training RMSE: {train_rmse:.4f}")
        LOGGER.info(f"Validation RMSE: {val_rmse:.4f}")
        LOGGER.info(f"Validation MAE: {self.metrics.mae:.4f}")
        LOGGER.info(f"Validation R²: {self.metrics.r2:.4f}")

        self.is_trained = True

        # Save artifacts
        artifacts = {}
        if save_artifacts:
            if output_dir is None:
                output_dir = get_project_root() / "models"
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / "pollution_forecast_model.pkl"
            self.save(model_path)
            artifacts["model_path"] = str(model_path)

            # Save metrics
            metrics_path = output_dir / "forecast_model_metrics.json"
            metrics_dict = {
                "rmse": self.metrics.rmse,
                "mae": self.metrics.mae,
                "r2": self.metrics.r2,
                "feature_count": len(self.feature_cols),
                "features": self.feature_cols,
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            artifacts["metrics_path"] = str(metrics_path)

            LOGGER.info(f"Model saved to {model_path}")

        return {
            "metrics": {
                "rmse": self.metrics.rmse,
                "mae": self.metrics.mae,
                "r2": self.metrics.r2,
            },
            "artifacts": artifacts,
            "is_trained": True,
        }

    def predict_future(
        self,
        factory_data: Dict[str, Any],
        current_score: float,
        years_ahead: int = 10,
        scenario: str = "business_as_usual",
    ) -> List[ForecastResult]:
        """Predict future pollution impact scores.

        Args:
            factory_data: Factory characteristics (industry_type, city, etc.).
            current_score: Current pollution impact score.
            years_ahead: Number of years to forecast.
            scenario: "business_as_usual" or "with_interventions".

        Returns:
            List[ForecastResult]: Yearly predictions with confidence intervals.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        industry = factory_data.get("industry_type", "Other")
        city = factory_data.get("city", "Mumbai")
        base_year = 2025
        base_month = 1

        # Scenario adjustments
        scenario_factor = 0.7 if scenario == "with_interventions" else 1.0

        predictions = []
        current_time_index = (base_year - 2020) * 12 + base_month

        # Generate monthly predictions and aggregate to yearly
        monthly_scores = []
        predicted_scores = [current_score]  # Track predictions for lag features

        for month_offset in range(years_ahead * 12):
            year = base_year + month_offset // 12
            month = (month_offset % 12) + 1
            time_index = current_time_index + month_offset

            # Get lag values from previous predictions
            score_lag_1 = predicted_scores[-1] if len(predicted_scores) >= 1 else current_score
            score_lag_3 = predicted_scores[-3] if len(predicted_scores) >= 3 else current_score
            score_rolling_mean_3 = np.mean(predicted_scores[-3:]) if len(predicted_scores) >= 3 else current_score

            # Apply growth factors over time for dynamic predictions
            time_factor = 1.0 + (month_offset / (years_ahead * 12)) * 0.2  # 20% growth over full period
            growth_rate = INDUSTRY_GROWTH_RATES.get(industry, 1.5) / 100  # Convert to decimal
            urbanization_rate = (CITY_URBANIZATION.get(city, 1.02) - 1.0)  # Excess over 1.0
            
            # Pollutant levels evolve based on previous score and growth
            base_pm25 = 20 + score_lag_1 * 15
            base_pm10 = 40 + score_lag_1 * 20
            
            # Build feature row with dynamic values
            feature_row = {
                "time_index": time_index,
                "year_norm": (year - 2020) / 10.0,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "industry_growth": INDUSTRY_GROWTH_RATES.get(industry, 1.5),
                "city_urbanization": CITY_URBANIZATION.get(city, 1.02),
                # Pollutant levels evolve over time
                "pm25": base_pm25 * (1 + growth_rate * month_offset / 12),
                "pm10": base_pm10 * (1 + growth_rate * month_offset / 12),
                "co": (0.5 + score_lag_1 * 0.3) * time_factor,
                "no2": (15 + score_lag_1 * 8) * (1 + urbanization_rate * month_offset / 12),
                "so2": (8 + score_lag_1 * 4) * time_factor,
                "o3": (20 + score_lag_1 * 5) * time_factor,
                "rolling_avg_pm25_7d": base_pm25 * (1 + growth_rate * month_offset / 12),
                "rolling_avg_pm25_30d": base_pm25 * (1 + growth_rate * month_offset / 12),
                "score_lag_1": score_lag_1,
                "score_lag_3": score_lag_3,
                "score_rolling_mean_3": score_rolling_mean_3,
                "industry_type": industry,
                "city": city,
            }

            # Predict
            X = pd.DataFrame([feature_row])
            pred = self.model.predict(X)[0]

            # Apply scenario factor (interventions reduce growth)
            if scenario == "with_interventions":
                pred = current_score + (pred - current_score) * scenario_factor

            # Clip to valid range
            pred = np.clip(pred, 0.0, 10.0)
            
            # Store prediction for next iteration's lag features
            predicted_scores.append(pred)
            monthly_scores.append({"year": year, "month": month, "score": pred})

        # Aggregate to yearly with confidence intervals
        yearly_data = {}
        for m in monthly_scores:
            year = m["year"]
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append(m["score"])

        for year, scores in sorted(yearly_data.items()):
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            # Confidence interval based on model uncertainty + seasonal variation
            ci_width = 0.5 + std_score * 0.8  # Base uncertainty + seasonal variance

            predictions.append(
                ForecastResult(
                    year=year,
                    predicted_score=float(mean_score),
                    confidence_lower=float(np.clip(mean_score - ci_width, 0.0, 10.0)),
                    confidence_upper=float(np.clip(mean_score + ci_width, 0.0, 10.0)),
                )
            )

        return predictions

    def analyze_trend(self, predictions: List[ForecastResult]) -> Tuple[str, str]:
        """Analyze trend direction and risk trajectory.

        Args:
            predictions: List of yearly predictions.

        Returns:
            Tuple[str, str]: (trend, risk_trajectory)
        """
        if len(predictions) < 2:
            return "stable", "stable"

        scores = [p.predicted_score for p in predictions]
        first_score = scores[0]
        last_score = scores[-1]
        change = last_score - first_score

        # Determine trend
        if change > 0.5:
            trend = "increasing"
        elif change < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        # Determine risk trajectory
        avg_score = np.mean(scores)
        if avg_score >= 7.0 or change > 1.0:
            risk_trajectory = "worsening"
        elif avg_score <= 4.0 or change < -1.0:
            risk_trajectory = "improving"
        else:
            risk_trajectory = "stable"

        return trend, risk_trajectory

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize metrics as dict to avoid pickle issues
        metrics_dict = None
        if self.metrics is not None:
            metrics_dict = {
                "rmse": self.metrics.rmse,
                "mae": self.metrics.mae,
                "r2": self.metrics.r2,
            }
        joblib.dump(
            {
                "model": self.model,
                "metrics": metrics_dict,
                "feature_cols": self.feature_cols,
                "is_trained": self.is_trained,
            },
            path,
        )

    def load(self, path: Path) -> None:
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        # Reconstruct ModelMetrics from dict
        metrics_dict = data.get("metrics")
        if metrics_dict is not None and isinstance(metrics_dict, dict):
            self.metrics = ModelMetrics(
                rmse=metrics_dict["rmse"],
                mae=metrics_dict["mae"],
                r2=metrics_dict["r2"],
            )
        else:
            self.metrics = metrics_dict  # Backward compatibility
        self.feature_cols = data["feature_cols"]
        self.is_trained = data["is_trained"]


def main():
    """Train and save the forecast model."""
    logging.basicConfig(level=logging.INFO)
    model = PollutionForecastModel()
    result = model.train()
    print(f"Training complete. RMSE: {result['metrics']['rmse']:.4f}")


if __name__ == "__main__":
    main()
