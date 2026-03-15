"""Tests for factory endpoints (GET /factories, GET /factory/{id})."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_factories_returns_200(client: TestClient) -> None:
    """GET /factories returns HTTP 200."""
    response = client.get("/factories")
    assert response.status_code == 200


def test_get_factories_response_shape(client: TestClient) -> None:
    """Response envelope contains expected pagination fields."""
    body = client.get("/factories").json()
    assert "total" in body
    assert "page" in body
    assert "page_size" in body
    assert "data" in body
    assert isinstance(body["data"], list)


def test_get_factories_pagination(client: TestClient) -> None:
    """page and page_size params are honoured and reflected in the response."""
    response = client.get("/factories?page=1&page_size=1")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 1
    assert len(body["data"]) == 1


def test_get_factory_by_id_valid(client: TestClient) -> None:
    """GET /factory/{id} returns 200 and correct factory data for a known ID."""
    response = client.get("/factory/FAC001")
    assert response.status_code == 200
    body = response.json()
    assert body["factory_id"] == "FAC001"
    assert body["city"] == "Pune"
    assert "pollution_impact_score" in body
    assert body["risk_level"] == "High"
    assert body["recommendations"] == ["Install scrubbers"]


def test_get_factory_by_id_not_found_returns_404(client: TestClient) -> None:
    """GET /factory/{id} returns 404 for an unknown factory ID."""
    response = client.get("/factory/NONEXISTENT_9999")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "not_found"
    assert "hint" in body


def test_get_factories_filter_by_city(client: TestClient) -> None:
    """city filter narrows results to exactly matching factories."""
    response = client.get("/factories?city=Pune")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["data"][0]["city"] == "Pune"


def test_get_factories_geo_filter(client: TestClient) -> None:
    """lat+lon+radius_km returns only factories within the radius."""
    # Centre on Pune (18.52, 73.85) with 10 km radius
    # Pune factory is at (18.52, 73.85) → 0 km — inside
    # Mumbai is ~115 km away — outside
    response = client.get("/factories?lat=18.52&lon=73.85&radius_km=10")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    cities = [f["city"] for f in body["data"]]
    assert "Pune" in cities
    assert "Mumbai" not in cities
