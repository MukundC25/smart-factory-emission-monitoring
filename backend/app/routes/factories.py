"""Factory data and prediction endpoints."""

import logging
import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from ..schemas import (
    Factory,
    PollutionImpactPredictionRequest,
    PollutionImpactPredictionResponse,
    FuturePredictionRequest,
    FuturePredictionResponse,
    YearlyPrediction,
)
from ..services.ml_service import get_ml_service
from ..services.forecast_service import predict_future_impact, get_current_pollution_score, is_forecast_model_ready
from ..database.db import get_db
from ..database.models import Factory as DBFactory, Recommendation
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/factories", tags=["Factories"])

# Cache for factory data
_factories_cache: Optional[pd.DataFrame] = None

# Path to model report containing expected feature list for predictions
_MODEL_REPORT_PATH = Path(__file__).parent.parent.parent.parent / "models" / "model_report.json"


def _load_model_feature_list() -> List[str]:
    """Load the model's expected feature list from the model report JSON.

    Returns:
        List[str]: List of feature names expected by the model.
    """
    try:
        if not _MODEL_REPORT_PATH.exists():
            logger.warning("Model report JSON not found at %s", _MODEL_REPORT_PATH)
            return []
        with _MODEL_REPORT_PATH.open("r", encoding="utf-8") as f:
            report = json.load(f)
        # Common keys for feature list in model reports
        feature_list = report.get("feature_list") or report.get("features")
        if isinstance(feature_list, list):
            return [str(name) for name in feature_list]
        logger.warning("No valid feature_list found in model report at %s", _MODEL_REPORT_PATH)
        return []
    except Exception as exc:
        logger.error("Failed to load model feature list from %s: %s", _MODEL_REPORT_PATH, exc)
        return []


def _load_factories_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load factory data from CSV.

    Args:
        csv_path: Optional path to factories CSV.

    Returns:
        pd.DataFrame: Factory dataset.
    """
    global _factories_cache

    if _factories_cache is not None:
        return _factories_cache

    if csv_path is None:
        from pathlib import Path as PathlibPath

        csv_path = PathlibPath(__file__).parent.parent.parent.parent / "data" / "raw" / "factories" / "factories.csv"

    if not csv_path.exists():
        logger.warning("Factories CSV not found at %s", csv_path)
        return pd.DataFrame()

    _factories_cache = pd.read_csv(csv_path)
    logger.info("Loaded %d factories from %s", len(_factories_cache), csv_path)
    return _factories_cache


@router.get("/", response_model=List[Factory])
def get_factories(city: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=1000)):
    """Get list of all factories with optional filtering.

    Args:
        city: Optional city filter.
        limit: Maximum number of results.

    Returns:
        List[Factory]: List of factories.
    """
    try:
        df = _load_factories_data()

        if df.empty:
            return []

        if city:
            df = df[df["city"].str.lower() == city.lower()]

        df = df.head(limit)
        factories = []
        for _, row in df.iterrows():
            factory = Factory(
                factory_id=str(row.get("factory_id", "")),
                factory_name=str(row.get("factory_name", "")),
                industry_type=str(row.get("industry_type", "")),
                latitude=float(row.get("latitude", 0)),
                longitude=float(row.get("longitude", 0)),
                city=str(row.get("city", "")),
                state=str(row.get("state", "")),
                country=str(row.get("country", "")),
            )
            factories.append(factory)
        return factories

    except Exception as e:
        logger.error("Error fetching factories: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve factories")


@router.get("/{factory_id}", response_model=Factory)
def get_factory_by_id(factory_id: str):
    """Get specific factory by ID.

    Args:
        factory_id: Factory identifier.

    Returns:
        Factory: Factory data.

    Raises:
        HTTPException: If factory not found.
    """
    try:
        df = _load_factories_data()

        if df.empty:
            raise HTTPException(status_code=404, detail="Factory not found")

        factory_row = df[df["factory_id"] == factory_id]

        if factory_row.empty:
            raise HTTPException(status_code=404, detail=f"Factory {factory_id} not found")

        row = factory_row.iloc[0]
        return Factory(
            factory_id=str(row.get("factory_id", "")),
            factory_name=str(row.get("factory_name", "")),
            industry_type=str(row.get("industry_type", "")),
            latitude=float(row.get("latitude", 0)),
            longitude=float(row.get("longitude", 0)),
            city=str(row.get("city", "")),
            state=str(row.get("state", "")),
            country=str(row.get("country", "")),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching factory %s: %s", factory_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve factory")


@router.post("/predict", response_model=PollutionImpactPredictionResponse)
def predict_pollution_impact(request: PollutionImpactPredictionRequest):
    """Predict pollution impact score for a factory.

    Args:
        request: Factory data with features.

    Returns:
        PollutionImpactPredictionResponse: Prediction result with risk level.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        service = get_ml_service()

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="ML model not loaded. Service unavailable.",
            )

        # Start from request data
        features_dict = request.dict()
        # Align request features with the model's expected feature list to avoid
        # missing-column errors during prediction.
        feature_list = _load_model_feature_list()
        if feature_list:
            aligned_features = {}
            for feature_name in feature_list:
                # Use value from request if available; otherwise set to None
                aligned_features[feature_name] = features_dict.get(feature_name)
            features_dict = aligned_features

        predicted_score = service.predict_single(features_dict)
        risk_level = service.get_risk_level(predicted_score)

        return PollutionImpactPredictionResponse(
            factory_id=request.factory_id,
            factory_name=request.factory_name,
            industry_type=request.industry_type,
            latitude=request.latitude,
            longitude=request.longitude,
            city=request.city,
            predicted_pollution_impact_score=predicted_score,
            risk_level=risk_level,
            confidence_context=f"Predicted score based on {request.industry_type} factory in {request.city}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/store", response_model=dict)
def store_factory(factory: Factory, db: Session = Depends(get_db)):
    """Store a new factory in the database.

    Args:
        factory: Factory data.

    Returns:
        dict: Success message.
    """
    try:
        db_factory = DBFactory(
            factory_id=factory.factory_id,
            factory_name=factory.factory_name,
            industry_type=factory.industry_type,
            latitude=factory.latitude,
            longitude=factory.longitude,
            city=factory.city,
            state=factory.state,
            country=factory.country,
        )
        db.add(db_factory)
        db.commit()
        db.refresh(db_factory)
        return {"message": "Factory stored successfully", "id": db_factory.id}
    except IntegrityError as e:
        db.rollback()
        logger.warning(
            "IntegrityError while storing factory with factory_id %s: %s",
            factory.factory_id,
            e,
        )
        raise HTTPException(
            status_code=409,
            detail="A factory with this factory_id already exists or violates a database constraint.",
        )
    except Exception as e:
        db.rollback()
        logger.error("Error storing factory: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store factory")


@router.post("/store-recommendation", response_model=dict)
def store_recommendation(
    factory_id: str,
    risk_level: str,
    recommendation_text: str,
    predicted_score: float,
    db: Session = Depends(get_db)
):
    """Store a new recommendation in the database.

    Args:
        factory_id: Factory ID.
        risk_level: Risk level.
        recommendation_text: Recommendation text.
        predicted_score: Predicted score.

    Returns:
        dict: Success message.
    """
    try:
        db_rec = Recommendation(
            factory_id=factory_id,
            risk_level=risk_level,
            recommendation_text=recommendation_text,
            predicted_score=predicted_score,
        )
        db.add(db_rec)
        db.commit()
        db.refresh(db_rec)
        return {"message": "Recommendation stored successfully", "id": db_rec.id}
    except Exception as e:
        db.rollback()
        logger.error("Error storing recommendation: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store recommendation")


@router.get("/batch/predict-all", response_model=List[PollutionImpactPredictionResponse])
def predict_all_factories():
    """Predict pollution impact for all factories in database.

    Returns:
        List[PollutionImpactPredictionResponse]: Predictions for all factories.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        service = get_ml_service()

        if not service.is_ready():
            raise HTTPException(
                status_code=503,
                detail="ML model not loaded. Service unavailable.",
            )

        df = _load_factories_data()

        if df.empty:
            return []

        predictions = []
        for _, row in df.iterrows():
            try:
                features_dict = {
                    "factory_id": str(row.get("factory_id", "")),
                    "factory_name": str(row.get("factory_name", "")),
                    "industry_type": str(row.get("industry_type", "")),
                    "latitude": float(row.get("latitude", 0)),
                    "longitude": float(row.get("longitude", 0)),
                    "city": str(row.get("city", "")),
                    "state": str(row.get("state", "")),
                    "country": str(row.get("country", "")),
                    "pm25": float(row.get("pm25", 0)),
                    "pm10": float(row.get("pm10", 0)),
                    "co": float(row.get("co", 0)),
                    "no2": float(row.get("no2", 0)),
                    "so2": float(row.get("so2", 0)),
                    "o3": float(row.get("o3", 0)),
                    "distance_to_nearest_station": float(row.get("distance_to_nearest_station", 0)),
                    "rolling_avg_pm25_7d": float(row.get("rolling_avg_pm25_7d", 0)),
                    "rolling_avg_pm25_30d": float(row.get("rolling_avg_pm25_30d", 0)),
                    "pollution_spike_flag": int(row.get("pollution_spike_flag", 0)),
                    "season": str(row.get("season", "unknown")),
                    "wind_direction_factor": float(row.get("wind_direction_factor", 1.0)),
                    "industry_risk_weight": float(row.get("industry_risk_weight", 6.0)),
                }
                predicted_score = service.predict_single(features_dict)
                risk_level = service.get_risk_level(predicted_score)

                predictions.append(
                    PollutionImpactPredictionResponse(
                        factory_id=features_dict["factory_id"],
                        factory_name=features_dict["factory_name"],
                        industry_type=features_dict["industry_type"],
                        latitude=features_dict["latitude"],
                        longitude=features_dict["longitude"],
                        city=features_dict["city"],
                        predicted_pollution_impact_score=predicted_score,
                        risk_level=risk_level,
                    )
                )
            except Exception as e:
                logger.warning("Failed to predict for factory %s: %s", row.get("factory_id"), e)
                continue

        logger.info("Generated predictions for %d factories", len(predictions))
        return predictions

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@router.post("/{factory_id}/predict-future", response_model=FuturePredictionResponse)
def predict_future_pollution(factory_id: str, request: FuturePredictionRequest):
    """Predict future pollution impact scores for a factory.

    Args:
        factory_id: Factory identifier.
        request: Future prediction parameters.

    Returns:
        FuturePredictionResponse: 10-year forecast with confidence intervals.

    Raises:
        HTTPException: If factory not found or prediction fails.
    """
    try:
        if not is_forecast_model_ready():
            raise HTTPException(
                status_code=503,
                detail="Forecast model not loaded. Run training first.",
            )

        # Load factory data
        df = _load_factories_data()
        if df.empty:
            raise HTTPException(status_code=404, detail="No factory data available")

        factory_row = df[df["factory_id"] == factory_id]
        if factory_row.empty:
            raise HTTPException(status_code=404, detail=f"Factory {factory_id} not found")

        row = factory_row.iloc[0]
        factory_data = {
            "factory_id": str(row.get("factory_id", "")),
            "factory_name": str(row.get("factory_name", "")),
            "industry_type": str(row.get("industry_type", "Other")),
            "city": str(row.get("city", "Mumbai")),
            "latitude": float(row.get("latitude", 0)),
            "longitude": float(row.get("longitude", 0)),
            "state": str(row.get("state", "")),
            "country": str(row.get("country", "")),
        }

        # Get current pollution score
        current_score = get_current_pollution_score(factory_id, df)
        if current_score is None:
            current_score = 5.0  # Default mid-range score

        # Generate predictions
        result = predict_future_impact(
            factory_id=factory_id,
            factory_data=factory_data,
            current_score=current_score,
            years_ahead=request.years_ahead,
            scenario=request.scenario,
        )

        if result is None:
            raise HTTPException(status_code=500, detail="Prediction failed")

        # Convert to response format
        yearly_predictions = [
            YearlyPrediction(
                year=p.year,
                predicted_score=p.predicted_score,
                confidence_lower=p.confidence_lower,
                confidence_upper=p.confidence_upper,
            )
            for p in result["predictions"]
        ]

        return FuturePredictionResponse(
            factory_id=factory_id,
            current_score=current_score,
            predictions=yearly_predictions,
            trend=result["trend"],
            risk_trajectory=result["risk_trajectory"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Future prediction error for factory %s: %s", factory_id, e)
        raise HTTPException(status_code=500, detail="Future prediction failed")
