"""Pydantic schemas for recommendation API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Single recommendation action item."""

    category: str = Field(description="Recommendation category")
    priority: str = Field(description="Priority level")
    action: str = Field(description="Recommended action text")
    pollutant: str = Field(description="Pollutant targeted by this recommendation")
    estimated_reduction: str = Field(description="Estimated pollution reduction impact")
    cost_category: str = Field(description="Relative cost category")
    timeline: str = Field(description="Expected implementation timeline")


class FactoryRecommendationSummary(BaseModel):
    """Summary representation of a factory recommendation report."""

    factory_id: str
    factory_name: str
    industry_type: str
    city: str
    risk_level: str
    composite_score: float
    dominant_pollutant: str
    summary: str
    generated_at: datetime


class FactoryRecommendationReport(FactoryRecommendationSummary):
    """Full recommendation report for a factory."""

    pollution_scores: Dict[str, float]
    recommendations: List[RecommendationItem]


class RecommendationsListResponse(BaseModel):
    """Paginated recommendation summaries."""

    total: int
    page: int
    page_size: int
    data: List[FactoryRecommendationSummary]


class PollutantCount(BaseModel):
    """Count for a dominant pollutant in recommendation stats."""

    pollutant: str
    count: int


class RiskLevelCounts(BaseModel):
    """Count of factories by risk level."""

    Critical: int = 0
    High: int = 0
    Medium: int = 0
    Low: int = 0


class RecommendationsStatsResponse(BaseModel):
    """Recommendation aggregate statistics payload."""

    total_factories: int
    by_risk_level: RiskLevelCounts
    top_pollutants: List[PollutantCount]
    last_generated: Optional[datetime] = None


class RecommendationsGenerateResponse(BaseModel):
    """Response returned when regeneration is triggered."""

    status: str
    factories_processed: int
    output_path: str
