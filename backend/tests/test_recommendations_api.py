"""Comprehensive tests for recommendations API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import backend.routers.recommendations as recommendations_router


@pytest.fixture
def disable_recommendation_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent recommendation generation tests from writing files to the repo tree."""

    def _no_op_export(*args, **kwargs) -> None:  # type: ignore[override]
        # Intentionally do nothing to avoid filesystem side effects during tests.
        return None

    monkeypatch.setattr(
        recommendations_router.RecommendationFormatter,
        "export_csv",
        _no_op_export,
    )
    monkeypatch.setattr(
        recommendations_router.RecommendationFormatter,
        "export_json",
        _no_op_export,
    )


def test_get_recommendations_returns_200(recommendations_client: TestClient) -> None:
    assert recommendations_client.get("/recommendations").status_code == 200


def test_get_recommendations_pagination(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations?page=1&page_size=1").json()
    assert body["page"] == 1
    assert body["page_size"] == 1


def test_get_recommendations_filter_by_risk_level_critical(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations?risk_level=critical").json()
    assert body["total"] >= 1
    assert all(item["risk_level"].lower() == "critical" for item in body["data"])


def test_get_recommendations_filter_by_city(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations?city=Pune").json()
    assert body["total"] >= 1
    assert all(item["city"] == "Pune" for item in body["data"])


def test_get_recommendations_filter_by_industry_type(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations?industry_type=steel").json()
    assert body["total"] >= 1
    assert all("steel" in item["industry_type"].lower() for item in body["data"])


def test_get_recommendations_empty_returns_200_not_500(empty_client: TestClient) -> None:
    body = empty_client.get("/recommendations").json()
    assert body["total"] == 0


def test_get_recommendation_by_id_returns_full_report(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/FAC001").json()
    assert body["factory_id"] == "FAC001"
    assert "recommendations" in body


def test_get_recommendation_by_id_not_found_returns_404(recommendations_client: TestClient) -> None:
    assert recommendations_client.get("/recommendations/UNKNOWN").status_code == 404


def test_get_recommendation_response_has_all_required_fields(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/FAC001").json()
    required = {
        "factory_id",
        "factory_name",
        "industry_type",
        "city",
        "risk_level",
        "composite_score",
        "dominant_pollutant",
        "pollution_scores",
        "recommendations",
        "summary",
        "generated_at",
    }
    assert required.issubset(body.keys())


def test_get_recommendations_stats_returns_200(recommendations_client: TestClient) -> None:
    assert recommendations_client.get("/recommendations/stats").status_code == 200


def test_get_recommendations_stats_has_total_factories(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/stats").json()
    assert "total_factories" in body


def test_get_recommendations_stats_has_by_risk_level(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/stats").json()
    assert "by_risk_level" in body


def test_get_recommendations_stats_has_top_pollutants(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/stats").json()
    assert "top_pollutants" in body


def test_get_recommendations_stats_last_generated_is_valid_datetime(recommendations_client: TestClient) -> None:
    body = recommendations_client.get("/recommendations/stats").json()
    assert body.get("last_generated")
    assert "T" in body["last_generated"]


def test_post_generate_returns_200(
    recommendations_client: TestClient,
    disable_recommendation_exports: None,
) -> None:
    assert recommendations_client.post("/recommendations/generate").status_code == 200


def test_post_generate_returns_factories_processed_count(
    recommendations_client: TestClient,
    disable_recommendation_exports: None,
) -> None:
    body = recommendations_client.post("/recommendations/generate").json()
    assert "factories_processed" in body


def test_post_generate_empty_datasets_returns_success_not_500(
    empty_client: TestClient,
    disable_recommendation_exports: None,
) -> None:
    response = empty_client.post("/recommendations/generate")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
