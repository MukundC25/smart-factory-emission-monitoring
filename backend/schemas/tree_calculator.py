"""Pydantic schemas for the Tree Planting Calculator API endpoints."""

from __future__ import annotations

from typing import Annotated, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PollutantReadings(BaseModel):
    """Current pollutant concentration readings for a factory location."""

    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    aqi_index: Optional[float] = None
    source: str = "OpenAQ"
    timestamp: Optional[str] = None


class TreesNeeded(BaseModel):
    """Three-tier tree planting requirement with safety buffers."""

    minimum: int = Field(description="Absolute minimum trees (0% buffer)")
    recommended: int = Field(description="Recommended trees (20% buffer)")
    optimal: int = Field(description="Optimal future-proof count (50% buffer)")


class PollutantBreakdown(BaseModel):
    """Per-pollutant tree requirements derived from individual absorption formulas."""

    pm25_trees: Optional[int] = None
    pm10_trees: Optional[int] = None
    no2_trees: Optional[int] = None
    so2_trees: Optional[int] = None
    co_trees: Optional[int] = None


class TreeRecommendationResponse(BaseModel):
    """Full tree planting recommendation for a single factory."""

    factory_id: str
    factory_name: str
    city: str
    industry_type: str
    current_aqi: float
    current_pollution_score: float
    dominant_pollutant: str
    current_readings: PollutantReadings
    target_aqi: float
    trees_needed: TreesNeeded
    impact_radius_km: float
    planting_area_hectares: float
    annual_co2_offset_tons: float
    estimated_aqi_reduction: float
    timeline_years: int
    pollutant_breakdown: PollutantBreakdown
    feasibility: str
    notes: List[str]
    data_source: str
    calculated_at: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "factory_id": "FAC001",
                "factory_name": "Pune Steel Works",
                "city": "Pune",
                "industry_type": "steel",
                "current_aqi": 156.0,
                "current_pollution_score": 7.5,
                "dominant_pollutant": "pm25",
                "target_aqi": 100.0,
                "trees_needed": {
                    "minimum": 850,
                    "recommended": 1020,
                    "optimal": 1275,
                },
                "feasibility": "Medium",
            }
        }
    )


class TreeCalculatorBulkRequest(BaseModel):
    """Request body for bulk tree-recommendation calculation."""

    factory_ids: Annotated[
        List[str],
        Field(
            max_length=50,
            description="Max 50 factory IDs per bulk request",
        ),
    ]


class TreeCalculatorBulkResponse(BaseModel):
    """Bulk response containing per-factory recommendations and any errors."""

    total: int
    results: List[TreeRecommendationResponse]
    errors: List[dict]  # {factory_id, error} for failed lookups
