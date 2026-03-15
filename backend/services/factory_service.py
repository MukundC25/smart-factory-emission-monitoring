"""Business logic for factory data queries."""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from backend.schemas.factory import FactoryBase, FactoryDetail, FactoryListResponse
from backend.utils.geo_utils import bounding_box, haversine_km

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE = 200


def _row_to_factory_base(row: pd.Series) -> FactoryBase:
    """Convert a DataFrame row to a FactoryBase schema object.

    Args:
        row: Pandas Series for one factory record.

    Returns:
        FactoryBase populated from the row.
    """

    def _str_or_none(key: str) -> Optional[str]:
        v = row.get(key)
        return str(v) if v is not None and pd.notna(v) else None

    return FactoryBase(
        factory_id=str(row.get("factory_id", "")),
        factory_name=str(row.get("factory_name", "")),
        industry_type=str(row.get("industry_type", "")),
        latitude=float(row.get("latitude", 0.0)),
        longitude=float(row.get("longitude", 0.0)),
        city=str(row.get("city", "")),
        state=_str_or_none("state"),
        country=_str_or_none("country"),
        source=_str_or_none("source"),
    )


def get_factories(
    factories_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    page: int = 1,
    page_size: int = 50,
    city: Optional[str] = None,
    industry_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 50.0,
) -> FactoryListResponse:
    """Return a paginated, optionally filtered factory list.

    Args:
        factories_df: Raw factories DataFrame.
        recommendations_df: Recommendations DataFrame used for risk_level filtering.
        page: 1-indexed page number.
        page_size: Records per page, capped at _MAX_PAGE_SIZE.
        city: Case-insensitive substring filter on city name.
        industry_type: Case-insensitive substring filter on industry_type.
        risk_level: Exact-match filter on risk level (Low/Medium/High).
        lat: Geo-filter centre latitude.
        lon: Geo-filter centre longitude.
        radius_km: Radius around (lat, lon) in km; ignored if lat/lon absent.

    Returns:
        FactoryListResponse with total count and paginated data.
    """
    page_size = min(page_size, _MAX_PAGE_SIZE)

    if factories_df.empty:
        return FactoryListResponse(total=0, page=page, page_size=page_size, data=[])

    df = factories_df.copy()

    # Merge pollution data from recommendations when needed for filter or response
    if not recommendations_df.empty:
        rec_cols = [c for c in ["factory_id", "risk_level", "composite_score", "latest_pm25", "latest_pm10", "dominant_pollutant", "immediate_actions"] if c in recommendations_df.columns]
        if rec_cols:
            rec_subset = recommendations_df[rec_cols].drop_duplicates("factory_id")
            df = df.merge(rec_subset, on="factory_id", how="left", suffixes=("", "_rec"))

    # Apply filters
    if city:
        df = df[df["city"].str.contains(city, case=False, na=False)]
    if industry_type:
        df = df[df["industry_type"].str.contains(industry_type, case=False, na=False)]
    if risk_level and "risk_level" in df.columns:
        df = df[df["risk_level"].fillna("").str.lower() == risk_level.lower()]

    if lat is not None and lon is not None:
        min_lat, max_lat, min_lon, max_lon = bounding_box(lat, lon, radius_km)
        # Fast bounding-box pre-filter, then precise haversine
        df = df[
            (df["latitude"] >= min_lat)
            & (df["latitude"] <= max_lat)
            & (df["longitude"] >= min_lon)
            & (df["longitude"] <= max_lon)
        ]
        df = df[
            df.apply(
                lambda r: haversine_km(
                    lat, lon, float(r["latitude"]), float(r["longitude"])
                )
                <= radius_km,
                axis=1,
            )
        ]

    total = len(df)
    offset = (page - 1) * page_size
    page_df = df.iloc[offset : offset + page_size]
    
    def _row_to_factory_detail(row: pd.Series) -> FactoryDetail:
        """Convert DataFrame row to FactoryDetail with pollution data."""
        def _str_or_none(key: str) -> Optional[str]:
            v = row.get(key)
            return str(v) if v is not None and pd.notna(v) else None
        
        def _float_or_none(key: str) -> Optional[float]:
            v = row.get(key)
            return float(v) if v is not None and pd.notna(v) else None
        
        def _recs_or_none(key: str) -> Optional[List[str]]:
            v = row.get(key)
            return [str(v)] if v is not None and pd.notna(v) else None
        
        return FactoryDetail(
            factory_id=str(row.get("factory_id", "")),
            factory_name=str(row.get("factory_name", "")),
            industry_type=str(row.get("industry_type", "")),
            latitude=float(row.get("latitude", 0.0)),
            longitude=float(row.get("longitude", 0.0)),
            city=str(row.get("city", "")),
            state=_str_or_none("state"),
            country=_str_or_none("country"),
            source=_str_or_none("source"),
            pollution_impact_score=_float_or_none("composite_score"),
            risk_level=_str_or_none("risk_level"),
            recommendations=_recs_or_none("immediate_actions"),
        )
    
    data = [_row_to_factory_detail(row) for _, row in page_df.iterrows()]
    return FactoryListResponse(total=total, page=page, page_size=page_size, data=data)


def get_factory_detail(
    factory_id: str,
    factories_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> Optional[FactoryDetail]:
    """Return a single factory with enriched pollution metrics.

    Args:
        factory_id: String factory ID to look up.
        factories_df: Raw factories DataFrame.
        recommendations_df: Recommendations DataFrame.

    Returns:
        FactoryDetail if found, None if the ID doesn't exist.
    """
    if factories_df.empty:
        return None

    matches = factories_df[factories_df["factory_id"].astype(str) == str(factory_id)]
    if matches.empty:
        return None

    base = _row_to_factory_base(matches.iloc[0])

    pollution_score: Optional[float] = None
    risk: Optional[str] = None
    recs: Optional[List[str]] = None

    if not recommendations_df.empty:
        rec_matches = recommendations_df[
            recommendations_df["factory_id"].astype(str) == str(factory_id)
        ]
        if not rec_matches.empty:
            rec_row = rec_matches.iloc[0]
            v = rec_row.get("pollution_impact_score")
            if v is not None and pd.notna(v):
                pollution_score = float(v)
            r = rec_row.get("risk_level")
            if r is not None and pd.notna(r):
                risk = str(r)
            t = rec_row.get("recommendation")
            if t is not None and pd.notna(t):
                recs = [str(t)]

    return FactoryDetail(
        **base.model_dump(),
        pollution_impact_score=pollution_score,
        risk_level=risk,
        recommendations=recs,
    )
