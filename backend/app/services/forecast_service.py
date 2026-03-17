"""Forecast service for time-series pollution predictions."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

# Lazy import to avoid loading during module import
_pollution_forecast_model = None


def _get_project_root() -> Path:
    """Get project root directory."""
    # Starting from backend/app/services/forecast_service.py,
    # go up three levels: services -> app -> backend -> <repo root>
    return Path(__file__).resolve().parent.parent.parent


def _load_forecast_model():
    """Load the pollution forecast model lazily."""
    global _pollution_forecast_model
    if _pollution_forecast_model is None:
        model_path = _get_project_root() / "models" / "pollution_forecast_model.pkl"
        if model_path.exists():
            try:
                # Import here to avoid circular imports
                from src.ml.time_series_predictor import PollutionForecastModel
                _pollution_forecast_model = PollutionForecastModel()
                _pollution_forecast_model.load(model_path)
                LOGGER.info("Loaded pollution forecast model from %s", model_path)
            except Exception as e:
                LOGGER.error("Failed to load forecast model: %s", e)
                _pollution_forecast_model = None
        else:
            LOGGER.warning("Forecast model not found at %s", model_path)
    return _pollution_forecast_model


def get_current_pollution_score(factory_id: str, factories_df: pd.DataFrame) -> Optional[float]:
    """Get current pollution score for a factory.
    
    Args:
        factory_id: Factory identifier.
        factories_df: DataFrame with factory data.
        
    Returns:
        Optional[float]: Current pollution score or None.
    """
    if factories_df.empty:
        return None
    
    factory_row = factories_df[factories_df["factory_id"] == factory_id]
    if factory_row.empty:
        return None
    
    # Try to get existing score
    score = factory_row.iloc[0].get("pollution_impact_score")
    if pd.notna(score):
        return float(score)
    
    # Estimate from pollutant levels if available
    pm25 = factory_row.iloc[0].get("pm25", 0)
    if pd.notna(pm25):
        # Rough estimation: score = (pm25 - 20) / 15
        estimated = (float(pm25) - 20) / 15
        return max(0.0, min(10.0, estimated))
    
    return 5.0  # Default mid-range score


def predict_future_impact(
    factory_id: str,
    factory_data: Dict[str, Any],
    current_score: float,
    years_ahead: int = 10,
    scenario: str = "business_as_usual",
) -> Optional[Dict[str, Any]]:
    """Predict future pollution impact for a factory.
    
    Args:
        factory_id: Factory identifier.
        factory_data: Factory characteristics.
        current_score: Current pollution impact score.
        years_ahead: Number of years to forecast.
        scenario: Prediction scenario.
        
    Returns:
        Optional[Dict]: Prediction results or None if model unavailable.
    """
    model = _load_forecast_model()
    if model is None:
        LOGGER.error("Forecast model not available")
        return None
    
    try:
        predictions = model.predict_future(
            factory_data=factory_data,
            current_score=current_score,
            years_ahead=years_ahead,
            scenario=scenario,
        )
        
        trend, risk_trajectory = model.analyze_trend(predictions)
        
        return {
            "predictions": predictions,
            "trend": trend,
            "risk_trajectory": risk_trajectory,
        }
    except Exception as e:
        LOGGER.error("Prediction failed for factory %s: %s", factory_id, e)
        return None


def is_forecast_model_ready() -> bool:
    """Check if forecast model is loaded and ready."""
    model = _load_forecast_model()
    return model is not None and getattr(model, "is_trained", False)


def get_model_metrics() -> Optional[Dict[str, float]]:
    """Get forecast model metrics if available."""
    model = _load_forecast_model()
    if model is None or not hasattr(model, "metrics") or model.metrics is None:
        return None
    
    return {
        "rmse": model.metrics.rmse,
        "mae": model.metrics.mae,
        "r2": model.metrics.r2,
    }
