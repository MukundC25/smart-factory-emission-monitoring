"""Business logic for pollution data queries."""

from __future__ import annotations

import datetime
import logging
from typing import List, Optional

import pandas as pd

from backend.schemas.pollution import PollutionListResponse, PollutionReading, PollutionStats
from backend.utils.geo_utils import bounding_box, haversine_km

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE = 200


def _row_to_reading(row: pd.Series) -> PollutionReading:
    """Convert a pollution DataFrame row to a PollutionReading schema.

    Args:
        row: Pandas Series for one pollution reading.

    Returns:
        PollutionReading instance with None for any missing fields.
    """

    def _opt_float(key: str) -> Optional[float]:
        v = row.get(key)
        return float(v) if v is not None and pd.notna(v) else None

    timestamp: Optional[datetime.datetime] = None
    raw_ts = row.get("timestamp")
    if raw_ts is not None and pd.notna(raw_ts):
        try:
            timestamp = pd.to_datetime(raw_ts, utc=True).to_pydatetime()
        except Exception:
            logger.debug("Could not parse timestamp value: %s", raw_ts)

    return PollutionReading(
        station_name=str(row["station_name"]) if pd.notna(row.get("station_name")) else None,
        latitude=_opt_float("station_lat"),
        longitude=_opt_float("station_lon"),
        city=str(row["city"]) if pd.notna(row.get("city")) else None,
        pm25=_opt_float("pm25"),
        pm10=_opt_float("pm10"),
        co=_opt_float("co"),
        no2=_opt_float("no2"),
        so2=_opt_float("so2"),
        o3=_opt_float("o3"),
        aqi_index=_opt_float("aqi_index"),
        timestamp=timestamp,
    )


def get_pollution(
    df: pd.DataFrame,
    page: int = 1,
    page_size: int = 50,
    city: Optional[str] = None,
    parameter: Optional[str] = None,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius_km: float = 50.0,
) -> PollutionListResponse:
    """Return paginated pollution readings with optional filters.

    Args:
        df: Pollution DataFrame.
        page: 1-indexed page number.
        page_size: Records per page, capped at _MAX_PAGE_SIZE.
        city: Case-insensitive substring filter on city name.
        parameter: Only include rows where this pollutant column is not null.
        start_date: Inclusive lower bound on reading timestamp.
        end_date: Inclusive upper bound on reading timestamp.
        lat: Geo-filter centre latitude.
        lon: Geo-filter centre longitude.
        radius_km: Radius around (lat, lon) in km.

    Returns:
        PollutionListResponse with total count and paginated data.
    """
    page_size = min(page_size, _MAX_PAGE_SIZE)

    if df.empty:
        return PollutionListResponse(total=0, page=page, page_size=page_size, data=[])

    fdf = df.copy()

    if city:
        fdf = fdf[fdf["city"].str.contains(city, case=False, na=False)]
    if parameter and parameter in fdf.columns:
        fdf = fdf[fdf[parameter].notna()]

    if (start_date or end_date) and "timestamp" in fdf.columns:
        fdf["_ts"] = pd.to_datetime(fdf["timestamp"], errors="coerce", utc=True)
        if start_date:
            fdf = fdf[fdf["_ts"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            end_dt = datetime.datetime.combine(end_date, datetime.time.max)
            fdf = fdf[fdf["_ts"] <= pd.Timestamp(end_dt, tz="UTC")]
        fdf = fdf.drop(columns=["_ts"])

    if lat is not None and lon is not None and {"station_lat", "station_lon"}.issubset(fdf.columns):
        min_lat, max_lat, min_lon, max_lon = bounding_box(lat, lon, radius_km)
        fdf = fdf[
            (fdf["station_lat"] >= min_lat)
            & (fdf["station_lat"] <= max_lat)
            & (fdf["station_lon"] >= min_lon)
            & (fdf["station_lon"] <= max_lon)
        ]
        fdf = fdf[
            fdf.apply(
                lambda r: (
                    haversine_km(
                        lat, lon, float(r["station_lat"]), float(r["station_lon"])
                    )
                    <= radius_km
                    if pd.notna(r.get("station_lat")) and pd.notna(r.get("station_lon"))
                    else False
                ),
                axis=1,
            )
        ]

    total = len(fdf)
    offset = (page - 1) * page_size
    page_df = fdf.iloc[offset : offset + page_size]
    data = [_row_to_reading(row) for _, row in page_df.iterrows()]
    return PollutionListResponse(total=total, page=page, page_size=page_size, data=data)


def get_pollution_stats(
    df: pd.DataFrame,
    city: Optional[str] = None,
    days: int = 30,
) -> List[PollutionStats]:
    """Return per-city aggregate pollution statistics.

    Args:
        df: Pollution DataFrame.
        city: Optional single-city filter (case-insensitive substring).
        days: Number of most-recent days to include in aggregation.

    Returns:
        List of PollutionStats, one entry per city present in the filtered data.
    """
    if df.empty:
        return []

    fdf = df.copy()

    if "timestamp" in fdf.columns:
        fdf["_ts"] = pd.to_datetime(fdf["timestamp"], errors="coerce", utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
        fdf = fdf[fdf["_ts"] >= cutoff]

    if city:
        fdf = fdf[fdf["city"].str.contains(city, case=False, na=False)]

    if fdf.empty:
        return []

    stats: List[PollutionStats] = []
    for city_name, group in fdf.groupby("city", sort=True):
        avg_pm25 = (
            round(float(group["pm25"].mean()), 2)
            if "pm25" in group and group["pm25"].notna().any()
            else None
        )
        avg_pm10 = (
            round(float(group["pm10"].mean()), 2)
            if "pm10" in group and group["pm10"].notna().any()
            else None
        )
        max_aqi = (
            round(float(group["aqi_index"].max()), 2)
            if "aqi_index" in group and group["aqi_index"].notna().any()
            else None
        )
        stats.append(
            PollutionStats(
                city=str(city_name),
                avg_pm25=avg_pm25,
                avg_pm10=avg_pm10,
                max_aqi=max_aqi,
                reading_count=len(group),
            )
        )
    return stats
