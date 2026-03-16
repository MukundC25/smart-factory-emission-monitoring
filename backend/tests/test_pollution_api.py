"""Comprehensive tests for pollution API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_pollution_returns_200(populated_client: TestClient) -> None:
    assert populated_client.get("/pollution").status_code == 200


def test_get_pollution_filter_by_city(populated_client: TestClient) -> None:
    body = populated_client.get("/pollution?city=Mumbai").json()
    assert body["total"] == 1
    assert body["data"][0]["city"] == "Mumbai"


def test_get_pollution_filter_by_parameter_pm25(populated_client: TestClient) -> None:
    response = populated_client.get("/pollution?parameter=pm25")
    assert response.status_code == 200


def test_get_pollution_filter_by_date_range(populated_client: TestClient) -> None:
    response = populated_client.get("/pollution?start_date=2024-01-11&end_date=2024-01-11")
    assert response.status_code == 200


def test_get_pollution_geo_filter(populated_client: TestClient) -> None:
    body = populated_client.get("/pollution?lat=18.52&lon=73.85&radius_km=10").json()
    assert body["total"] >= 1


def test_get_pollution_empty_dataset_returns_200_not_500(empty_client: TestClient) -> None:
    body = empty_client.get("/pollution").json()
    assert body["total"] == 0
    assert body["data"] == []


def test_get_pollution_invalid_parameter_returns_422(populated_client: TestClient) -> None:
    response = populated_client.get("/pollution?parameter=invalid")
    assert response.status_code == 422


def test_get_pollution_stats_returns_200(populated_client: TestClient) -> None:
    assert populated_client.get("/pollution/stats").status_code == 200


def test_get_pollution_stats_structure_has_required_fields(populated_client: TestClient) -> None:
    body = populated_client.get("/pollution/stats?days=9999").json()
    if body:
        assert {"city", "avg_pm25", "avg_pm10", "max_aqi", "reading_count"}.issubset(body[0].keys())


def test_get_pollution_stats_empty_dataset_returns_empty_list(empty_client: TestClient) -> None:
    body = empty_client.get("/pollution/stats").json()
    assert body == []


def test_get_heatmap_data_returns_200(populated_client: TestClient) -> None:
    assert populated_client.get("/pollution/heatmap/data").status_code == 200


def test_get_heatmap_data_points_are_lat_lon_intensity_triples(populated_client: TestClient) -> None:
    body = populated_client.get("/pollution/heatmap/data?parameter=pm25").json()
    if body["points"]:
        first = body["points"][0]
        assert isinstance(first, list)
        assert len(first) == 3


def test_get_heatmap_data_limit_param_respected(populated_client: TestClient) -> None:
    body = populated_client.get("/pollution/heatmap/data?parameter=pm25&limit=1").json()
    assert len(body["points"]) <= 1


def test_get_heatmap_data_invalid_parameter_returns_422(populated_client: TestClient) -> None:
    response = populated_client.get("/pollution/heatmap/data?parameter=invalid")
    assert response.status_code == 422


def test_get_heatmap_data_empty_dataset_returns_empty_points(empty_client: TestClient) -> None:
    body = empty_client.get("/pollution/heatmap/data").json()
    assert body["points"] == []
