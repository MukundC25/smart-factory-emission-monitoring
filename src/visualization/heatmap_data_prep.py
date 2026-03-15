"""Data preparation utilities for pollution heatmap generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)

_LAT_CANDIDATES = ("station_lat", "latitude")
_LON_CANDIDATES = ("station_lon", "longitude")
_INTENSITY_PRIORITY = ("aqi_index", "pm25", "pm10", "no2", "so2", "co")


class HeatmapDataPreparator:
    """Prepare pollution datasets for heatmap rendering.

    This class loads pollution data from disk, validates coordinate quality,
    resolves the best pollution intensity metric, and returns normalized
    points suitable for Folium HeatMap consumption.
    """

    def __init__(self) -> None:
        """Initialize the data preparator."""
        self._lat_col: str | None = None
        self._lon_col: str | None = None

    def load_pollution_data(self, path: Path) -> pd.DataFrame:
        """Load pollution data from CSV or Parquet.

        Args:
            path: Source file path.

        Returns:
            Loaded dataframe.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If required columns cannot be resolved.
        """
        if not path.exists():
            raise FileNotFoundError(f"Pollution dataset not found at: {path}")

        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(
                f"Unsupported pollution file format '{suffix}'. Use CSV or Parquet."
            )

        self._lat_col, self._lon_col = self._resolve_coordinate_columns(df)
        self.resolve_intensity_column(df)

        LOGGER.info(
            "Loaded pollution data from %s with %d rows and columns=%s",
            path,
            len(df),
            list(df.columns),
        )
        return df

    def validate_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid coordinate rows.

        Args:
            df: Input dataframe.

        Returns:
            Filtered dataframe with valid latitude/longitude rows only.

        Raises:
            ValueError: If coordinate columns cannot be resolved.
        """
        lat_col, lon_col = self._resolve_coordinate_columns(df)
        lat_values = pd.to_numeric(df[lat_col], errors="coerce")
        lon_values = pd.to_numeric(df[lon_col], errors="coerce")

        null_mask = lat_values.isna() | lon_values.isna()
        out_of_range_mask = (
            (lat_values < -90)
            | (lat_values > 90)
            | (lon_values < -180)
            | (lon_values > 180)
        )
        invalid_mask = null_mask | out_of_range_mask

        dropped_null = int(null_mask.sum())
        dropped_range = int((out_of_range_mask & ~null_mask).sum())
        total_dropped = int(invalid_mask.sum())

        filtered = df.loc[~invalid_mask].copy()
        filtered[lat_col] = pd.to_numeric(filtered[lat_col], errors="coerce")
        filtered[lon_col] = pd.to_numeric(filtered[lon_col], errors="coerce")

        LOGGER.info(
            "Coordinate validation complete: dropped=%d (null=%d, out_of_range=%d), remaining=%d",
            total_dropped,
            dropped_null,
            dropped_range,
            len(filtered),
        )
        return filtered

    def resolve_intensity_column(self, df: pd.DataFrame) -> str:
        """Resolve the best available intensity metric.

        Priority order: aqi_index > pm25 > pm10 > no2 > so2 > co.

        Args:
            df: Input dataframe.

        Returns:
            Name of selected intensity column.

        Raises:
            ValueError: If none of the supported intensity columns exist.
        """
        for col in _INTENSITY_PRIORITY:
            if col in df.columns:
                LOGGER.info(
                    "Selected intensity column '%s' using priority order %s",
                    col,
                    _INTENSITY_PRIORITY,
                )
                return col

        raise ValueError(
            "No supported intensity column found. Expected one of: "
            f"{', '.join(_INTENSITY_PRIORITY)}"
        )

    def normalize_intensity(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Normalize intensity values into [0.0, 1.0].

        NaN values are filled with median intensity before normalization.

        Args:
            df: Input dataframe.
            col: Intensity source column.

        Returns:
            Dataframe with added intensity_normalized column.
        """
        normalized = df.copy()
        series = pd.to_numeric(normalized[col], errors="coerce")

        median_value = float(series.median()) if series.notna().any() else 0.0
        series = series.fillna(median_value)

        min_value = float(series.min()) if len(series) > 0 else 0.0
        max_value = float(series.max()) if len(series) > 0 else 0.0

        if max_value == min_value:
            normalized["intensity_normalized"] = 1.0 if len(series) > 0 else pd.Series(dtype="float64")
        else:
            normalized["intensity_normalized"] = (series - min_value) / (max_value - min_value)

        LOGGER.info(
            "Normalized intensity column '%s' with min=%.4f max=%.4f median_fill=%.4f",
            col,
            min_value,
            max_value,
            median_value,
        )
        return normalized

    def get_heatmap_points(self, df: pd.DataFrame) -> List[List[float]]:
        """Convert prepared dataframe into heatmap points.

        Args:
            df: Input dataframe containing coordinate and normalized intensity columns.

        Returns:
            List of [latitude, longitude, normalized_intensity].
        """
        lat_col, lon_col = self._resolve_coordinate_columns(df)
        if "intensity_normalized" not in df.columns:
            raise ValueError("Column 'intensity_normalized' is required before point export")

        lat_values = pd.to_numeric(df[lat_col], errors="coerce")
        lon_values = pd.to_numeric(df[lon_col], errors="coerce")
        intensity_values = pd.to_numeric(df["intensity_normalized"], errors="coerce")

        points_df = pd.DataFrame(
            {
                "lat": lat_values,
                "lon": lon_values,
                "intensity": intensity_values,
            }
        ).dropna(subset=["lat", "lon", "intensity"])

        points: List[List[float]] = points_df[["lat", "lon", "intensity"]].astype(float).values.tolist()
        LOGGER.info("Prepared %d heatmap points", len(points))
        return points

    def get_city_center(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute median map center from available coordinates.

        Args:
            df: Input dataframe.

        Returns:
            Tuple of (latitude, longitude). Falls back to India centroid when no data exists.
        """
        if df.empty:
            return (20.5937, 78.9629)

        lat_col, lon_col = self._resolve_coordinate_columns(df)
        lat_values = pd.to_numeric(df[lat_col], errors="coerce").dropna()
        lon_values = pd.to_numeric(df[lon_col], errors="coerce").dropna()

        if lat_values.empty or lon_values.empty:
            return (20.5937, 78.9629)

        return (float(lat_values.median()), float(lon_values.median()))

    def _resolve_coordinate_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Resolve coordinate columns from known candidates.

        Args:
            df: Input dataframe.

        Returns:
            Tuple of (latitude_column, longitude_column).

        Raises:
            ValueError: If either latitude or longitude column is unavailable.
        """
        lat_col = next((c for c in _LAT_CANDIDATES if c in df.columns), None)
        lon_col = next((c for c in _LON_CANDIDATES if c in df.columns), None)

        if lat_col is None or lon_col is None:
            raise ValueError(
                "Could not resolve coordinate columns. Expected one of "
                f"latitude={_LAT_CANDIDATES}, longitude={_LON_CANDIDATES}."
            )

        self._lat_col = lat_col
        self._lon_col = lon_col
        return lat_col, lon_col
