"""Tests for heatmap generation and API data endpoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.visualization.heatmap_data_prep import HeatmapDataPreparator
from src.visualization.heatmap_generator import HeatmapGenerator


def test_heatmap_generates_html_file(tmp_path: Path) -> None:
    """Heatmap HTML is generated for valid pollution input."""
    prep = HeatmapDataPreparator()
    df = pd.DataFrame(
        {
            "station_lat": [18.5, 18.7],
            "station_lon": [73.8, 74.0],
            "station_name": ["A", "B"],
            "city": ["Pune", "Pune"],
            "pm25": [40.0, 80.0],
            "pm10": [60.0, 120.0],
            "aqi_index": [90.0, 180.0],
            "timestamp": ["2026-03-01T00:00:00Z", "2026-03-02T00:00:00Z"],
        }
    )
    df = prep.validate_coordinates(df)
    intensity_col = prep.resolve_intensity_column(df)
    df = prep.normalize_intensity(df, intensity_col)

    output = tmp_path / "pollution_heatmap_test.html"
    generator = HeatmapGenerator({"tile_provider": "CartoDB positron"})
    path = generator.build_full_map(df, intensity_col, output)

    assert path.exists()
    assert path.suffix == ".html"


def test_heatmap_with_empty_dataframe_does_not_crash(tmp_path: Path) -> None:
    """Empty dataframe still renders a valid map output."""
    output = tmp_path / "empty_heatmap.html"
    df = pd.DataFrame(columns=["station_lat", "station_lon", "intensity_normalized"])

    generator = HeatmapGenerator({})
    path = generator.build_full_map(df, "aqi_index", output)

    assert path.exists()


def test_heatmap_data_endpoint_returns_points(client: TestClient) -> None:
    """Heatmap API endpoint returns point payload and metadata."""
    response = client.get("/pollution/heatmap/data?parameter=pm25&limit=2")
    assert response.status_code == 200
    body = response.json()
    assert "points" in body
    assert "metadata" in body
    assert body["metadata"]["parameter"] == "pm25"
    assert len(body["points"]) <= 2


def test_heatmap_data_endpoint_empty_dataset(empty_client: TestClient) -> None:
    """Heatmap API endpoint returns empty points for empty dataset."""
    response = empty_client.get("/pollution/heatmap/data")
    assert response.status_code == 200
    body = response.json()
    assert body["points"] == []
