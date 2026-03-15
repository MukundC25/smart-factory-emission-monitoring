"""Pydantic schemas for pollution endpoints.

Column mapping from pollution_readings.csv:
  station_lat  → latitude
  station_lon  → longitude
All other column names match directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PollutionReading(BaseModel):
    """Single air-quality monitoring station reading.

    Attributes:
        station_name: Monitoring station identifier.
        latitude: Station WGS84 latitude (mapped from station_lat column).
        longitude: Station WGS84 longitude (mapped from station_lon column).
        city: City for this reading.
        pm25: PM2.5 concentration in µg/m³.
        pm10: PM10 concentration in µg/m³.
        co: Carbon monoxide in mg/m³.
        no2: Nitrogen dioxide in µg/m³.
        so2: Sulphur dioxide in µg/m³.
        o3: Ozone concentration in µg/m³.
        aqi_index: Aggregate Air Quality Index.
        timestamp: UTC timestamp of this reading.
    """

    station_name: Optional[str] = Field(
        default=None, description="Monitoring station name"
    )
    latitude: Optional[float] = Field(
        default=None, description="Station WGS84 latitude"
    )
    longitude: Optional[float] = Field(
        default=None, description="Station WGS84 longitude"
    )
    city: Optional[str] = Field(default=None, description="City for this reading")
    pm25: Optional[float] = Field(
        default=None, description="PM2.5 concentration µg/m³"
    )
    pm10: Optional[float] = Field(
        default=None, description="PM10 concentration µg/m³"
    )
    co: Optional[float] = Field(default=None, description="CO concentration mg/m³")
    no2: Optional[float] = Field(default=None, description="NO₂ concentration µg/m³")
    so2: Optional[float] = Field(default=None, description="SO₂ concentration µg/m³")
    o3: Optional[float] = Field(default=None, description="O₃ concentration µg/m³")
    aqi_index: Optional[float] = Field(
        default=None, description="Aggregate Air Quality Index"
    )
    timestamp: Optional[datetime] = Field(
        default=None, description="Reading timestamp (UTC)"
    )

    model_config = {"from_attributes": True}


class PollutionListResponse(BaseModel):
    """Paginated list of pollution readings.

    Attributes:
        total: Total matching rows before pagination.
        page: Current page (1-indexed).
        page_size: Rows per page.
        data: Pollution readings for this page.
    """

    total: int = Field(description="Total matching records")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Records per page")
    data: List[PollutionReading]


class PollutionStats(BaseModel):
    """Per-city aggregate pollution statistics.

    Attributes:
        city: City name.
        avg_pm25: Mean PM2.5 for the period.
        avg_pm10: Mean PM10 for the period.
        max_aqi: Maximum AQI observed.
        reading_count: Number of readings included in the aggregation.
    """

    city: str
    avg_pm25: Optional[float] = None
    avg_pm10: Optional[float] = None
    max_aqi: Optional[float] = None
    reading_count: int
