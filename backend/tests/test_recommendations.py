"""API tests for recommendations endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from backend.dependencies import get_data_loader
from backend.main import app


class MockRecommendationsLoader:
    """Mock loader for recommendations API tests."""

    def __init__(self) -> None:
        self._reports = [
            {
                "factory_id": "FAC001",
                "factory_name": "Pune Steel Works",
                "industry_type": "steel",
                "city": "Pune",
                "risk_level": "High",
                "composite_score": 7.8,
                "dominant_pollutant": "so2",
                "pollution_scores": {
                    "pm25_score": 6.1,
                    "pm10_score": 6.6,
                    "so2_score": 7.8,
                    "no2_score": 5.0,
                    "co_score": 2.1,
                    "o3_score": 2.7,
                },
                "recommendations": [
                    {
                        "category": "Emission Control",
                        "priority": "Immediate",
                        "action": "Install wet scrubber",
                        "pollutant": "so2",
                        "estimated_reduction": "60-80% SO2 reduction",
                        "cost_category": "High",
                        "timeline": "3-6 months installation",
                    }
                ],
                "summary": "High SO2 risk detected.",
                "generated_at": "2026-03-15T17:00:00+00:00",
            }
        ]

    def load_recommendation_reports(self):  # type: ignore[no-untyped-def]
        return self._reports

    def load_factories(self) -> pd.DataFrame:
        return pd.DataFrame([{"factory_id": "FAC001"}])

    def load_pollution(self) -> pd.DataFrame:
        return pd.DataFrame([{"city": "Pune"}])

    def load_recommendations(self) -> pd.DataFrame:
        return pd.DataFrame([{"factory_id": "FAC001"}])

    def dataset_info(self) -> dict:
        return {"factories": 1, "pollution": 1, "recommendations": 1}

    def refresh(self) -> None:
        return None


@pytest.fixture()
def recommendations_client() -> TestClient:
    """TestClient with recommendation-aware mock loader injected."""
    app.dependency_overrides[get_data_loader] = lambda: MockRecommendationsLoader()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    app.dependency_overrides.clear()


def test_api_get_recommendations_200(recommendations_client: TestClient) -> None:
    """GET /recommendations should return 200 with paginated payload."""
    response = recommendations_client.get("/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert "total" in body
    assert "data" in body
    assert isinstance(body["data"], list)


def test_api_get_recommendation_by_id_404(recommendations_client: TestClient) -> None:
    """GET /recommendations/{factory_id} should return clear 404 for missing ID."""
    missing_id = "UNKNOWN_FACTORY"
    response = recommendations_client.get(f"/recommendations/{missing_id}")
    assert response.status_code == 404
    body = response.json()
    assert missing_id in body.get("message", "")


def test_api_recommendations_stats(recommendations_client: TestClient) -> None:
    """GET /recommendations/stats should return aggregate recommendation statistics."""
    response = recommendations_client.get("/recommendations/stats")
    assert response.status_code == 200
    body = response.json()
    assert "total_factories" in body
    assert "by_risk_level" in body
    assert "top_pollutants" in body
