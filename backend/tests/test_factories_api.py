"""Comprehensive tests for factory API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_factories_returns_200_with_valid_schema(populated_client: TestClient) -> None:
    response = populated_client.get("/factories")
    assert response.status_code == 200
    body = response.json()
    assert {"total", "page", "page_size", "data"}.issubset(body.keys())


def test_get_factories_default_pagination(populated_client: TestClient) -> None:
    body = populated_client.get("/factories").json()
    assert body["page"] == 1
    assert body["page_size"] == 50


def test_get_factories_custom_pagination(populated_client: TestClient) -> None:
    response = populated_client.get("/factories?page=2&page_size=1")
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 2
    assert body["page_size"] == 1


def test_get_factories_page_size_max_limit(populated_client: TestClient) -> None:
    response = populated_client.get("/factories?page_size=201")
    assert response.status_code == 422


def test_get_factories_filter_by_city_returns_correct_subset(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?city=Pune").json()
    assert body["total"] == 1
    assert all(item["city"] == "Pune" for item in body["data"])


def test_get_factories_filter_by_industry_type(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?industry_type=text").json()
    assert body["total"] == 1
    assert "text" in body["data"][0]["industry_type"].lower()


def test_get_factories_filter_by_risk_level_low(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?risk_level=Low").json()
    assert all((item.get("risk_level") or "").lower() == "low" for item in body["data"])


def test_get_factories_filter_by_risk_level_high(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?risk_level=High").json()
    assert body["total"] >= 1
    assert all((item.get("risk_level") or "").lower() == "high" for item in body["data"])


def test_get_factories_geo_filter_with_lat_lon_radius(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?lat=18.52&lon=73.85&radius_km=10").json()
    assert body["total"] >= 1
    assert any(item["city"] == "Pune" for item in body["data"])


def test_get_factories_geo_filter_without_radius_uses_default(populated_client: TestClient) -> None:
    response = populated_client.get("/factories?lat=18.52&lon=73.85")
    assert response.status_code == 200


def test_get_factories_empty_dataset_returns_200_not_500(empty_client: TestClient) -> None:
    response = empty_client.get("/factories")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["data"] == []


def test_get_factories_response_total_matches_data_count(populated_client: TestClient) -> None:
    body = populated_client.get("/factories?page=1&page_size=200").json()
    assert body["total"] >= len(body["data"])


def test_get_factory_by_id_returns_correct_factory(populated_client: TestClient) -> None:
    response = populated_client.get("/factory/FAC001")
    assert response.status_code == 200
    assert response.json()["factory_id"] == "FAC001"


def test_get_factory_by_id_not_found_returns_404(populated_client: TestClient) -> None:
    response = populated_client.get("/factory/DOES_NOT_EXIST")
    assert response.status_code == 404


def test_get_factory_by_id_404_message_contains_factory_id(populated_client: TestClient) -> None:
    missing_id = "DOES_NOT_EXIST"
    response = populated_client.get(f"/factory/{missing_id}")
    assert response.status_code == 404
    assert missing_id in response.json().get("message", "")


def test_get_factory_by_id_response_matches_schema(populated_client: TestClient) -> None:
    body = populated_client.get("/factory/FAC001").json()
    required = {
        "factory_id",
        "factory_name",
        "industry_type",
        "latitude",
        "longitude",
        "city",
        "pollution_impact_score",
        "risk_level",
        "recommendations",
    }
    assert required.issubset(body.keys())
