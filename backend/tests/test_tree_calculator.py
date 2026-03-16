"""API-level tests for Tree Planting Calculator endpoints.

Uses FastAPI TestClient with MockDataLoader injected — no disk I/O.
All OpenAQ HTTP calls are patched so tests remain offline/deterministic.

Fixtures from backend/tests/conftest.py:
  - test_client   → MockDataLoader (FAC001, FAC002, FAC003)
  - empty_client  → EmptyDataLoader
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock OpenAQ return value re-used across multiple tests
# ---------------------------------------------------------------------------

_MOCK_AQI_DATA = {
    "aqi": 178.0,
    "pm25": 72.0,
    "pm10": 95.0,
    "no2": 30.0,
    "so2": 12.0,
    "co": 0.9,
    "o3": 38.0,
    "source": "openaq",
    "timestamp": "2026-03-16T12:00:00Z",
}


# ===========================================================================
# 1.  GET single factory — cached data (use_live_aqi=false)
# ===========================================================================


def test_get_tree_recommendation_valid_factory_200(test_client):
    """GET /factories/FAC001/tree-recommendation?use_live_aqi=false → 200."""
    response = test_client.get(
        "/factories/FAC001/tree-recommendation?use_live_aqi=false"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["factory_id"] == "FAC001"
    assert body["factory_name"] == "Pune Steel Works"
    assert body["city"] == "Pune"
    assert "trees_needed" in body
    assert body["trees_needed"]["recommended"] >= body["trees_needed"]["minimum"]
    assert "feasibility" in body
    assert body["feasibility"] in ("High", "Medium", "Low")
    assert body["data_source"] == "cached"


# ===========================================================================
# 2.  GET single factory — 404 for unknown ID
# ===========================================================================


def test_get_tree_recommendation_unknown_factory_404(test_client):
    """GET /factories/INVALID_ID/tree-recommendation → 404."""
    response = test_client.get(
        "/factories/INVALID_ID/tree-recommendation?use_live_aqi=false"
    )
    assert response.status_code == 404
    message = response.json()["message"]
    assert "INVALID_ID" in message


# ===========================================================================
# 3.  GET single factory — live AQI (mocked OpenAQClient)
# ===========================================================================


def test_get_tree_recommendation_live_aqi_uses_openaq(test_client):
    """GET /factories/FAC001/tree-recommendation?use_live_aqi=true uses OpenAQ."""
    with patch("backend.routers.tree_calculator.OpenAQClient") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_city_aqi.return_value = _MOCK_AQI_DATA
        mock_cls.return_value = mock_instance

        response = test_client.get(
            "/factories/FAC001/tree-recommendation?use_live_aqi=true"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["current_aqi"] == pytest.approx(178.0)
    assert body["current_readings"]["aqi_index"] == pytest.approx(178.0)
    assert body["data_source"] == "openaq"


# ===========================================================================
# 4.  POST bulk — returns results for each valid ID
# ===========================================================================


def test_bulk_tree_recommendation_returns_results(test_client):
    """POST /factories/tree-recommendation/bulk returns tree recs for all valid IDs."""
    payload = {"factory_ids": ["FAC001", "FAC002", "FAC003"]}
    response = test_client.post(
        "/factories/tree-recommendation/bulk?use_live_aqi=false",
        json=payload,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["results"]) == 3
    # Errors list should be empty since all IDs are valid
    assert body["errors"] == []


# ===========================================================================
# 5.  POST bulk — max-length validation (51 IDs → 422)
# ===========================================================================


def test_bulk_tree_recommendation_rejects_over_50_factories(test_client):
    """POST with 51 factory IDs should fail Pydantic validation → 422."""
    payload = {"factory_ids": [f"FAC{i:03d}" for i in range(51)]}
    response = test_client.post(
        "/factories/tree-recommendation/bulk",
        json=payload,
    )
    assert response.status_code == 422


# ===========================================================================
# 6.  GET constants — returns expected keys
# ===========================================================================


def test_get_calculator_constants_returns_expected_keys(test_client):
    """GET /tree-calculator/constants should return scientific constants dict."""
    response = test_client.get("/tree-calculator/constants")
    assert response.status_code == 200
    body = response.json()
    assert "particulate_matter_absorption" in body
    assert "carbon_absorption" in body
    assert "aqi_thresholds_cpcb_india" in body
    assert "safety_buffers" in body
    # Spot-check a specific constant value
    assert body["particulate_matter_absorption"]["tree_pm25_absorption_ug_m3"] == pytest.approx(0.8)
