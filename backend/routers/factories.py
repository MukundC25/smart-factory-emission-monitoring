"""Factory API router.

Endpoints:
  GET /factories              — paginated list with optional filters
  GET /factory/{factory_id}  — full detail for a single factory
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_data_loader
from backend.schemas.factory import FactoryDetail, FactoryListResponse
from backend.services.factory_service import get_factories, get_factory_detail
from backend.utils.data_loader import DataLoader

router = APIRouter(tags=["Factories"])
logger = logging.getLogger(__name__)


@router.get(
    "/factories",
    response_model=FactoryListResponse,
    summary="List factories with optional filters",
)
def list_factories(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    city: Optional[str] = Query(None, description="Filter by city name (substring)"),
    industry_type: Optional[str] = Query(
        None, description="Filter by industry type (substring)"
    ),
    risk_level: Optional[str] = Query(
        None,
        pattern="^(Low|Medium|High)$",
        description="Filter by risk level",
    ),
    lat: Optional[float] = Query(None, description="Geo-filter centre latitude"),
    lon: Optional[float] = Query(None, description="Geo-filter centre longitude"),
    radius_km: float = Query(
        50.0, gt=0, description="Geo radius in km (requires lat+lon)"
    ),
    loader: DataLoader = Depends(get_data_loader),
) -> FactoryListResponse:
    """Return a paginated list of factories with optional city, industry, risk, and geo filters."""
    return get_factories(
        factories_df=loader.load_factories(),
        recommendations_df=loader.load_recommendations(),
        page=page,
        page_size=page_size,
        city=city,
        industry_type=industry_type,
        risk_level=risk_level,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
    )


@router.get(
    "/factory/{factory_id}",
    response_model=FactoryDetail,
    summary="Get factory detail by ID",
)
def get_factory(
    factory_id: str,
    loader: DataLoader = Depends(get_data_loader),
) -> FactoryDetail:
    """Return full detail for a single factory including risk score and recommendations.

    Raises:
        HTTPException: 404 if factory_id is not found in the dataset.
    """
    detail = get_factory_detail(
        factory_id=factory_id,
        factories_df=loader.load_factories(),
        recommendations_df=loader.load_recommendations(),
    )
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Factory '{factory_id}' not found. "
                "Use GET /factories to browse available factory IDs."
            ),
        )
    return detail
