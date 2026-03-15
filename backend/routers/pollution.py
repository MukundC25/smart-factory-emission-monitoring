"""Pollution API router.

Endpoints:
  GET /pollution        — paginated readings with optional filters
  GET /pollution/stats  — per-city aggregate statistics
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from backend.dependencies import get_data_loader
from backend.schemas.pollution import PollutionListResponse, PollutionStats
from backend.services.pollution_service import get_pollution, get_pollution_stats
from backend.utils.data_loader import DataLoader

router = APIRouter(tags=["Pollution"])
logger = logging.getLogger(__name__)


@router.get(
    "/pollution",
    response_model=PollutionListResponse,
    summary="List pollution readings with optional filters",
)
def list_pollution(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    city: Optional[str] = Query(None, description="Filter by city (substring)"),
    parameter: Optional[str] = Query(
        None,
        pattern="^(pm25|pm10|co|no2|so2)$",
        description="Only return rows where this pollutant is present",
    ),
    start_date: Optional[date] = Query(
        None, description="Inclusive start date (YYYY-MM-DD)"
    ),
    end_date: Optional[date] = Query(
        None, description="Inclusive end date (YYYY-MM-DD)"
    ),
    lat: Optional[float] = Query(None, description="Geo-filter centre latitude"),
    lon: Optional[float] = Query(None, description="Geo-filter centre longitude"),
    radius_km: float = Query(
        50.0, gt=0, description="Geo radius in km (requires lat+lon)"
    ),
    loader: DataLoader = Depends(get_data_loader),
) -> PollutionListResponse:
    """Return paginated pollution readings with optional city, parameter, date, and geo filters.

    Gracefully returns an empty list when the pollution dataset has not yet been populated.
    Will work with zero code changes once the real dataset lands in the configured path.
    """
    return get_pollution(
        df=loader.load_pollution(),
        page=page,
        page_size=page_size,
        city=city,
        parameter=parameter,
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
    )


@router.get(
    "/pollution/stats",
    response_model=List[PollutionStats],
    summary="Aggregate pollution statistics per city",
)
def pollution_stats(
    city: Optional[str] = Query(None, description="Filter to a single city"),
    days: int = Query(30, ge=1, le=9999, description="Rolling window in days"),
    loader: DataLoader = Depends(get_data_loader),
) -> List[PollutionStats]:
    """Return per-city PM2.5, PM10, and AQI aggregates over the most-recent N days."""
    return get_pollution_stats(df=loader.load_pollution(), city=city, days=days)
