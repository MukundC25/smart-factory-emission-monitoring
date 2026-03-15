"""Tests for pollution endpoints and system health check."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_pollution_returns_200(client: TestClient) -> None:
    """GET /pollution returns HTTP 200 with mock data loaded."""
    response = client.get("/pollution")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["data"]) == 3


def test_get_pollution_empty_dataset_does_not_crash(empty_client: TestClient) -> None:
    """GET /pollution returns 200 with empty data when no dataset is present."""
    response = empty_client.get("/pollution")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["data"] == []


def test_get_pollution_stats(client: TestClient) -> None:
    """GET /pollution/stats returns per-city aggregate statistics."""
    response = client.get("/pollution/stats?days=9999")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    stat = body[0]
    assert "city" in stat
    assert "reading_count" in stat
    assert "avg_pm25" in stat
    assert "avg_pm10" in stat
    assert "max_aqi" in stat


def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns 200 with status=ok and dataset row counts."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "datasets_loaded" in body
    assert "timestamp" in body
    assert body["datasets_loaded"]["factories"] == 3


def test_health_endpoint_empty_data(empty_client: TestClient) -> None:
    """GET /health returns 200 even when all datasets are empty."""
    response = empty_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["datasets_loaded"]["factories"] == 0
    assert body["datasets_loaded"]["pollution"] == 0


def test_get_pollution_filter_by_city(client: TestClient) -> None:
    """city filter narrows pollution results correctly."""
    response = client.get("/pollution?city=Pune")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["data"][0]["city"] == "Pune"


def test_get_pollution_pagination(client: TestClient) -> None:
    """page_size=1 returns exactly one reading."""
    response = client.get("/pollution?page=1&page_size=1")
    assert response.status_code == 200
    body = response.json()
    assert body["page_size"] == 1
    assert len(body["data"]) == 1
    assert body["total"] == 3
