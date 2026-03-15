"""Unit tests for heatmap data preparation logic."""

from __future__ import annotations

import pandas as pd

from src.visualization.heatmap_data_prep import HeatmapDataPreparator


def test_validate_coordinates_drops_invalid_rows() -> None:
    """Rows with invalid or null coordinates are removed."""
    df = pd.DataFrame(
        {
            "station_lat": [18.5, 95.0, None, 19.1],
            "station_lon": [73.8, 74.1, 70.0, -181.0],
            "aqi_index": [100, 120, 80, 140],
        }
    )

    prep = HeatmapDataPreparator()
    filtered = prep.validate_coordinates(df)

    assert len(filtered) == 1
    assert float(filtered.iloc[0]["station_lat"]) == 18.5


def test_resolve_intensity_column_priority_order() -> None:
    """Intensity resolution follows priority order."""
    df = pd.DataFrame(
        {
            "station_lat": [18.5],
            "station_lon": [73.8],
            "pm25": [42.0],
            "pm10": [75.0],
        }
    )

    prep = HeatmapDataPreparator()
    assert prep.resolve_intensity_column(df) == "pm25"


def test_normalize_intensity_range_is_0_to_1() -> None:
    """Normalized intensity values stay within 0 and 1."""
    df = pd.DataFrame(
        {
            "station_lat": [18.5, 18.6, 18.7],
            "station_lon": [73.8, 73.9, 74.0],
            "aqi_index": [50.0, None, 150.0],
        }
    )

    prep = HeatmapDataPreparator()
    normalized = prep.normalize_intensity(df, "aqi_index")

    assert "intensity_normalized" in normalized.columns
    assert float(normalized["intensity_normalized"].min()) >= 0.0
    assert float(normalized["intensity_normalized"].max()) <= 1.0


def test_get_heatmap_points_no_nulls() -> None:
    """Heatmap points exclude rows with null normalized intensity."""
    df = pd.DataFrame(
        {
            "station_lat": [18.5, 18.6],
            "station_lon": [73.8, 73.9],
            "intensity_normalized": [0.2, None],
        }
    )

    prep = HeatmapDataPreparator()
    points = prep.get_heatmap_points(df)

    assert len(points) == 1
    assert points[0] == [18.5, 73.8, 0.2]


def test_get_city_center_fallback_when_no_data() -> None:
    """Empty dataframe returns India center fallback."""
    df = pd.DataFrame(columns=["station_lat", "station_lon"])
    prep = HeatmapDataPreparator()
    center = prep.get_city_center(df)
    assert center == (20.5937, 78.9629)
