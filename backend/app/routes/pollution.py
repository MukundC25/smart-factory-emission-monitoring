"""Pollution data endpoints."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
import pandas as pd

from ..schemas import PollutionReading
from ..database.db import get_db
from ..database.models import PollutionReading as DBPollutionReading

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pollution", tags=["Pollution"])

# Cache for pollution data
_pollution_cache: Optional[pd.DataFrame] = None


def _load_pollution_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load pollution readings from CSV.

    Args:
        csv_path: Optional path to pollution CSV.

    Returns:
        pd.DataFrame: Pollution readings dataset.
    """
    global _pollution_cache

    if _pollution_cache is not None:
        return _pollution_cache

    if csv_path is None:
        from pathlib import Path as PathlibPath

        csv_path = (
            PathlibPath(__file__).parent.parent.parent.parent
            / "data"
            / "raw"
            / "pollution"
            / "pollution_readings.csv"
        )

    if not csv_path.exists():
        logger.warning("Pollution CSV not found at %s", csv_path)
        return pd.DataFrame()

    _pollution_cache = pd.read_csv(csv_path)
    logger.info("Loaded %d pollution readings from %s", len(_pollution_cache), csv_path)
    return _pollution_cache


@router.get("/", response_model=List[PollutionReading])
def get_pollution_data(
    city: Optional[str] = Query(None),
    station: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get pollution readings with optional filtering.

    Args:
        city: Optional city filter.
        station: Optional station name filter.
        limit: Maximum number of results.

    Returns:
        List[PollutionReading]: List of pollution readings.
    """
    try:
        df = _load_pollution_data()

        if df.empty:
            return []

        if city:
            df = df[df["city"].str.lower() == city.lower()]

        if station:
            df = df[df["station_name"].str.lower().str.contains(station.lower(), na=False)]

        df = df.head(limit)

        readings = []
        for _, row in df.iterrows():
            reading = PollutionReading(
                station_name=str(row.get("station_name", "")),
                station_lat=float(row.get("station_lat", 0)),
                station_lon=float(row.get("station_lon", 0)),
                city=str(row.get("city", "")),
                timestamp=str(row.get("timestamp", "")),
                pm25=float(row.get("pm25")) if pd.notna(row.get("pm25")) else None,
                pm10=float(row.get("pm10")) if pd.notna(row.get("pm10")) else None,
                co=float(row.get("co")) if pd.notna(row.get("co")) else None,
                no2=float(row.get("no2")) if pd.notna(row.get("no2")) else None,
                so2=float(row.get("so2")) if pd.notna(row.get("so2")) else None,
                o3=float(row.get("o3")) if pd.notna(row.get("o3")) else None,
                aqi_index=float(row.get("aqi_index")) if pd.notna(row.get("aqi_index")) else None,
            )
            readings.append(reading)

        return readings

    except Exception as e:
        logger.error("Error fetching pollution data: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve pollution data")


@router.get("/summary/{city}", response_model=dict)
def get_city_pollution_summary(city: str):
    """Get aggregated pollution metrics for a city.

    Args:
        city: City name.

    Returns:
        dict: Summary statistics for the city.

    Raises:
        HTTPException: If city not found.
    """
    try:
        df = _load_pollution_data()

        if df.empty:
            raise HTTPException(status_code=404, detail="No pollution data available")

        city_df = df[df["city"].str.lower() == city.lower()]

        if city_df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for city: {city}")

        pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3"]
        summary = {
            "city": city,
            "total_readings": len(city_df),
            "stations": city_df["station_name"].nunique() if "station_name" in city_df.columns else 0,
            "pollutant_stats": {},
        }

        for pollutant in pollutants:
            if pollutant in city_df.columns:
                col_data = city_df[pollutant].dropna()
                if len(col_data) > 0:
                    summary["pollutant_stats"][pollutant] = {
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "count": int(len(col_data)),
                    }

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error computing pollution summary for %s: %s", city, e)
        raise HTTPException(status_code=500, detail="Failed to compute summary")


@router.get("/stations", response_model=List[dict])
def get_pollution_stations():
    """Get list of all pollution monitoring stations.

    Returns:
        List[dict]: List of unique stations with counts.
    """
    try:
        df = _load_pollution_data()

        if df.empty or "station_name" not in df.columns:
            return []

        stations = (
            df.groupby("station_name")
            .agg(
                {
                    "station_lat": "first",
                    "station_lon": "first",
                    "city": "first",
                    "station_name": "count",
                }
            )
            .rename(columns={"station_name": "reading_count"})
            .reset_index()
        )

        result = []
        for _, row in stations.iterrows():
            result.append(
                {
                    "station_name": str(row.get("station_name", "")),
                    "city": str(row.get("city", "")),
                    "latitude": float(row.get("station_lat", 0)),
                    "longitude": float(row.get("station_lon", 0)),
                    "reading_count": int(row.get("reading_count", 0)),
                }
            )

        return result

    except Exception as e:
        logger.error("Error fetching stations: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve stations")


@router.post("/store", response_model=dict)
def store_pollution_reading(reading: PollutionReading, db: Session = Depends(get_db)):
    """Store a new pollution reading in the database.

    Args:
        reading: Pollution reading data.

    Returns:
        dict: Success message.
    """
    from datetime import datetime

    if not isinstance(reading.timestamp, str) or not reading.timestamp.strip():
        raise HTTPException(
            status_code=422,
            detail="Timestamp is required and must be a non-empty ISO 8601 string",
        )
    try:
        parsed_timestamp = datetime.fromisoformat(
            reading.timestamp.replace("Z", "+00:00")
        )
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Timestamp must be a valid ISO 8601 datetime string",
        )
    try:
        db_reading = DBPollutionReading(
            station_name=reading.station_name,
            station_lat=reading.station_lat,
            station_lon=reading.station_lon,
            city=reading.city,
            timestamp=parsed_timestamp,
            pm25=reading.pm25,
            pm10=reading.pm10,
            co=reading.co,
            no2=reading.no2,
            so2=reading.so2,
            o3=reading.o3,
            aqi_index=reading.aqi_index,
            source="api",
        )
        db.add(db_reading)
        db.commit()
        db.refresh(db_reading)
        return {"message": "Pollution reading stored successfully", "id": db_reading.id}
    except Exception as e:
        db.rollback()
        logger.error("Error storing pollution reading: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store pollution reading")
