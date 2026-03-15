"""Pydantic schemas for API request/response validation."""

from typing import Optional
from pydantic import BaseModel, Field


class FactoryBase(BaseModel):
    """Base factory model."""

    factory_id: str
    factory_name: str
    industry_type: str
    latitude: float
    longitude: float
    city: str
    state: str
    country: str


class Factory(FactoryBase):
    """Factory with prediction data."""

    pollution_impact_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None


class PollutionReading(BaseModel):
    """Pollution measurement data."""

    station_name: str
    station_lat: float
    station_lon: float
    city: str
    timestamp: str
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    co: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    o3: Optional[float] = None
    aqi_index: Optional[float] = None


class PollutionImpactPredictionRequest(BaseModel):
    """Request for pollution impact prediction."""

    factory_id: str
    factory_name: str
    industry_type: str
    latitude: float
    longitude: float
    city: str
    state: str
    country: str
    pm25: Optional[float] = 0.0
    pm10: Optional[float] = 0.0
    co: Optional[float] = 0.0
    no2: Optional[float] = 0.0
    so2: Optional[float] = 0.0
    o3: Optional[float] = 0.0
    distance_to_nearest_station: Optional[float] = 0.0
    rolling_avg_pm25_7d: Optional[float] = 0.0
    rolling_avg_pm25_30d: Optional[float] = 0.0
    pollution_spike_flag: Optional[int] = 0
    season: str = "unknown"
    wind_direction_factor: Optional[float] = 1.0
    industry_risk_weight: Optional[float] = 6.0


class PollutionImpactPredictionResponse(BaseModel):
    """Response with pollution impact prediction."""

    factory_id: str
    factory_name: str
    industry_type: str
    latitude: float
    longitude: float
    city: str
    predicted_pollution_impact_score: float = Field(
        ..., description="Predicted pollution impact score (0-10)"
    )
    risk_level: str = Field(
        ..., description="Risk level: Low, Medium, or High"
    )
    confidence_context: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Recommendation response for a factory."""

    factory_id: str
    factory_name: str
    risk_level: str
    pollution_impact_score: float
    recommendation: str
    measures: Optional[list[str]] = None


class ModelMetrics(BaseModel):
    """Model training metrics."""

    model_name: str
    rmse: float
    mae: float
    r2: float


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str
