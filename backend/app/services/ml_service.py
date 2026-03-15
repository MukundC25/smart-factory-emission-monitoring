"""ML Service: Model loading, inference, and pollution impact scoring."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MLService:
    """Service for pollution impact prediction and model management."""

    def __init__(self, model_path: Optional[Path] = None, config_path: Optional[Path] = None):
        """Initialize ML service with model artifacts.

        Args:
            model_path: Path to trained model pickle file.
            config_path: Path to model configuration/report.

        Raises:
            FileNotFoundError: If model file not found.
        """
        self.model = None
        self.scaler = None
        self.config = {}
        self.model_path = model_path
        self.config_path = config_path
        self._load_model()
        self._load_config()

    def _load_model(self) -> None:
        """Load trained model from disk."""
        if not self.model_path or not self.model_path.exists():
            logger.warning("Model path not found: %s. Service may not be functional.", self.model_path)
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def _load_config(self) -> None:
        """Load model configuration/report."""
        if not self.config_path or not self.config_path.exists():
            logger.warning("Config path not found: %s", self.config_path)
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            logger.info("Model config loaded successfully")
        except Exception as e:
            logger.error("Failed to load config: %s", e)

    def is_ready(self) -> bool:
        """Check if service is ready for predictions.

        Returns:
            bool: True if model is loaded, False otherwise.
        """
        return self.model is not None

    def predict_impact_score(self, features: pd.DataFrame) -> np.ndarray:
        """Generate pollution impact predictions.

        Args:
            features: Feature dataframe (one or more rows).

        Returns:
            np.ndarray: Predicted pollution impact scores.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded. Cannot predict.")

        try:
            predictions = self.model.predict(features)
            # Clip predictions to valid range [0, 10]
            predictions = np.clip(predictions, 0, 10)
            return predictions
        except Exception as e:
            logger.error("Prediction failed: %s", e)
            raise

    def predict_single(self, features_dict: Dict[str, Any]) -> float:
        """Predict impact score for a single record.

        Args:
            features_dict: Dictionary of feature values.

        Returns:
            float: Predicted pollution impact score.
        """
        df = pd.DataFrame([features_dict])
        predictions = self.predict_impact_score(df)
        return float(predictions[0])

    def get_risk_level(self, score: float, config: Optional[Dict[str, Any]] = None) -> str:
        """Map pollution impact score to risk level.

        Args:
            score: Pollution impact score (0-10).
            config: Optional config with risk thresholds.

        Returns:
            str: Risk category (Low, Medium, High).
        """
        cfg = config or self.config
        if not cfg:
            # Default thresholds
            cfg = {"risk_bands": {"low_max": 3.0, "medium_max": 6.0}}

        risk_cfg = cfg.get("risk_bands", {"low_max": 3.0, "medium_max": 6.0})
        low_max = float(risk_cfg.get("low_max", 3.0))
        medium_max = float(risk_cfg.get("medium_max", 6.0))

        if score <= low_max:
            return "Low"
        elif score <= medium_max:
            return "Medium"
        else:
            return "High"

    def get_recommendation(self, risk_level: str, industry_type: str) -> str:
        """Generate control recommendation based on risk level.

        Args:
            risk_level: Risk category (Low, Medium, High).
            industry_type: Factory industry type.

        Returns:
            str: Plain-English recommendation text.
        """
        industry_title = industry_type.title() if industry_type else "Industrial"

        recommendations = {
            "High": (
                f"{industry_title} site is HIGH RISK. Install SO2 scrubbers and ESP filters, "
                "upgrade baghouse maintenance to weekly, deploy continuous emissions monitoring "
                "with automated alerts, and conduct quarterly compliance audits."
            ),
            "Medium": (
                f"{industry_title} site is MEDIUM RISK. Increase stack testing to bi-weekly, "
                "perform preventive burner tuning monthly, add leak-detection walkthroughs each shift, "
                "and maintain monthly air quality reports."
            ),
            "Low": (
                f"{industry_title} site is LOW RISK. Maintain compliance logs, perform monthly "
                "calibration checks, sustain preventive maintenance schedules, and conduct "
                "annual third-party audits."
            ),
        }
        return recommendations.get(risk_level, "Unknown risk level")

    def get_control_measures(self, risk_level: str, industry_type: str) -> List[str]:
        """Get specific control measures for a factory.

        Args:
            risk_level: Risk category.
            industry_type: Factory industry type.

        Returns:
            List[str]: Recommended control measures.
        """
        high_risk_measures = [
            "Install SO2 scrubbers",
            "Upgrade ESP (Electrostatic Precipitator) filters",
            "Weekly baghouse maintenance",
            "Continuous emissions monitoring system (CEMS)",
            "Automated alert system deployment",
            "Quarterly compliance audits",
        ]

        medium_risk_measures = [
            "Bi-weekly stack testing",
            "Monthly preventive burner tuning",
            "Shift-level leak detection walkthroughs",
            "Monthly air quality reports",
            "Enhanced maintenance schedules",
            "Semi-annual audits",
        ]

        low_risk_measures = [
            "Maintain compliance documentation",
            "Monthly calibration checks",
            "Standard maintenance schedules",
            "Annual third-party audit",
            "Quarterly emissions reports",
        ]

        measures_map = {
            "High": high_risk_measures,
            "Medium": medium_risk_measures,
            "Low": low_risk_measures,
        }
        return measures_map.get(risk_level, [])

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance info.

        Returns:
            Dict[str, Any]: Model information.
        """
        return {
            "is_loaded": self.is_ready(),
            "model_type": self.config.get("selected_model", "unknown"),
            "metrics": self.config.get("metrics", {}),
            "config": self.config.get("config", {}),
        }


# Global service instance
_service_instance: Optional[MLService] = None


def get_ml_service(
    model_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> MLService:
    """Get or create singleton ML service instance.

    Args:
        model_path: Optional path to override default model location.
        config_path: Optional path to override default config location.

    Returns:
        MLService: Initialized ML service instance.
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = MLService(model_path, config_path)

    return _service_instance


def reset_service() -> None:
    """Reset the global service instance (useful for testing)."""
    global _service_instance
    _service_instance = None
