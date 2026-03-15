"""Pydantic schemas for factory endpoints."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class FactoryBase(BaseModel):
    """Core factory fields returned in every list-level response.

    Attributes:
        factory_id: Unique factory identifier.
        factory_name: Human-readable name.
        industry_type: Industry classification.
        latitude: WGS84 latitude.
        longitude: WGS84 longitude.
        city: City where the factory is located.
        state: State or province.
        country: Country name.
        source: Data source label (osm, google_places, synthetic).
    """

    factory_id: str = Field(description="Unique factory identifier")
    factory_name: str = Field(description="Human-readable factory name")
    industry_type: str = Field(
        description="Industry classification (e.g. manufacturing, chemical)"
    )
    latitude: float = Field(description="WGS84 latitude")
    longitude: float = Field(description="WGS84 longitude")
    city: str = Field(description="City where the factory is located")
    state: Optional[str] = Field(default=None, description="State or province")
    country: Optional[str] = Field(default=None, description="Country name")
    source: Optional[str] = Field(
        default=None,
        description="Data source (osm, google_places, synthetic)",
    )

    model_config = {"from_attributes": True}


class FactoryDetail(FactoryBase):
    """Extended factory view including ML pollution analysis.

    Attributes:
        pollution_impact_score: ML-predicted impact score (0–10, higher = worse).
        risk_level: Risk band: Low, Medium, or High.
        recommendations: Actionable control recommendations.
    """

    pollution_impact_score: Optional[float] = Field(
        default=None,
        description="ML-predicted pollution impact score (0–10)",
    )
    risk_level: Optional[str] = Field(
        default=None,
        description="Risk classification: Low / Medium / High",
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Actionable recommendations for this factory",
    )


class FactoryListResponse(BaseModel):
    """Paginated list of factory records.

    Attributes:
        total: Total matching records available (before pagination).
        page: Current page number (1-indexed).
        page_size: Records per page.
        data: Factory records for this page.
    """

    total: int = Field(description="Total matching records (before pagination)")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Records per page")
    data: List[FactoryDetail]
