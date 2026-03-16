"""Shared fixtures for backend API tests.

Uses FastAPI dependency_overrides to inject mock DataLoaders — no real
disk I/O during tests, and no fragility from external API keys.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.dependencies import get_data_loader
from backend.main import app


# ---------------------------------------------------------------------------
# Mock DataFrames — column names match real CSV schemas exactly
# ---------------------------------------------------------------------------


def _mock_factories_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factory_id": ["FAC001", "FAC002", "FAC003"],
            "factory_name": ["Pune Steel Works", "Mumbai Textiles", "Bengaluru Electronics"],
            "industry_type": ["manufacturing", "textiles", "electronics"],
            "latitude": [18.52, 19.07, 12.97],
            "longitude": [73.85, 72.87, 77.59],
            "city": ["Pune", "Mumbai", "Bengaluru"],
            "state": ["Maharashtra", "Maharashtra", "Karnataka"],
            "country": ["India", "India", "India"],
            "source": ["osm", "osm", "osm"],
            "osm_id": [1001, 1002, 1003],
            "last_updated": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )


def _mock_pollution_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pm25": [45.2, 32.1, 78.5],
            "pm10": [60.0, 42.0, 95.0],
            "co": [0.5, 0.3, 1.2],
            "no2": [20.0, 15.0, 35.0],
            "so2": [10.0, 8.0, 20.0],
            "o3": [30.0, 25.0, 40.0],
            "aqi_index": [120.0, 90.0, 180.0],
            "timestamp": [
                "2024-01-10T12:00:00Z",
                "2024-01-11T12:00:00Z",
                "2024-01-12T12:00:00Z",
            ],
            "station_name": ["Pune CPCB", "Mumbai Chembur", "Bengaluru BTM"],
            "station_lat": [18.52, 19.07, 12.97],
            "station_lon": [73.85, 72.87, 77.59],
            "city": ["Pune", "Mumbai", "Bengaluru"],
            "country": ["India", "India", "India"],
            "source": ["synthetic", "synthetic", "synthetic"],
            "nearest_factory_distance_km": [2.1, 3.4, 1.8],
        }
    )


def _mock_recommendations_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factory_id": ["FAC001", "FAC002", "FAC003"],
            "factory_name": ["Pune Steel Works", "Mumbai Textiles", "Bengaluru Electronics"],
            "industry_type": ["manufacturing", "textiles", "electronics"],
            "latitude": [18.52, 19.07, 12.97],
            "longitude": [73.85, 72.87, 77.59],
            "city": ["Pune", "Mumbai", "Bengaluru"],
            "state": ["Maharashtra", "Maharashtra", "Karnataka"],
            "country": ["India", "India", "India"],
            "pollution_impact_score": [7.5, 4.2, 8.9],
            "latest_pm25": [45.2, 32.1, 78.5],
            "latest_pm10": [60.0, 42.0, 95.0],
            "risk_level": ["High", "Low", "High"],
            "recommendation": [
                "Install scrubbers",
                "Regular maintenance",
                "Switch to clean energy",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Mock loader classes
# ---------------------------------------------------------------------------


class MockDataLoader:
    """Test double returning fixed small DataFrames."""

    def load_factories(self) -> pd.DataFrame:
        """Return mock factories."""
        return _mock_factories_df()

    def load_pollution(self) -> pd.DataFrame:
        """Return mock pollution readings."""
        return _mock_pollution_df()

    def load_recommendations(self) -> pd.DataFrame:
        """Return mock recommendations."""
        return _mock_recommendations_df()

    def load_recommendation_reports(self) -> list[dict]:
        """Return recommendation reports used by recommendations endpoints."""
        records = []
        for row in _mock_recommendations_df().to_dict(orient="records"):
            records.append(
                {
                    "factory_id": row["factory_id"],
                    "factory_name": row["factory_name"],
                    "industry_type": row["industry_type"],
                    "city": row["city"],
                    "risk_level": row.get("risk_level", "Low"),
                    "composite_score": float(row.get("pollution_impact_score", 0.0) or 0.0),
                    "dominant_pollutant": "pm25",
                    "pollution_scores": {
                        "pm25_score": 5.0,
                        "pm10_score": 5.0,
                        "so2_score": 4.0,
                        "no2_score": 4.0,
                        "co_score": 2.0,
                        "o3_score": 2.0,
                    },
                    "summary": "Mock recommendation summary",
                    "generated_at": "2026-03-16T00:00:00+00:00",
                    "recommendations": [
                        {
                            "category": "Compliance",
                            "priority": "Short-term",
                            "action": "Regular monitoring",
                            "pollutant": "pm25",
                            "estimated_reduction": "N/A",
                            "cost_category": "Low",
                            "timeline": "1-2 months",
                        }
                    ],
                }
            )
        return records

    def refresh(self) -> None:
        """No-op refresh for test double."""
        return None

    def dataset_info(self) -> dict:
        """Return mock dataset row counts."""
        return {"factories": 3, "pollution": 3, "recommendations": 3}


class EmptyDataLoader(MockDataLoader):
    """Test double returning empty DataFrames for edge-case tests."""

    def load_factories(self) -> pd.DataFrame:
        """Return empty DataFrame."""
        return pd.DataFrame()

    def load_pollution(self) -> pd.DataFrame:
        """Return empty DataFrame."""
        return pd.DataFrame()

    def load_recommendations(self) -> pd.DataFrame:
        """Return empty DataFrame."""
        return pd.DataFrame()

    def load_recommendation_reports(self) -> list[dict]:
        """Return empty recommendations list."""
        return []

    def dataset_info(self) -> dict:
        """Return zero row counts."""
        return {"factories": 0, "pollution": 0, "recommendations": 0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_client() -> TestClient:
    """TestClient with populated mock data via dependency override."""
    app.dependency_overrides[get_data_loader] = lambda: MockDataLoader()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def populated_client() -> TestClient:
    """Alias fixture explicitly named for populated dataset scenarios."""
    app.dependency_overrides[get_data_loader] = lambda: MockDataLoader()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def empty_client() -> TestClient:
    """TestClient with empty datasets for edge-case / graceful-fallback tests."""
    app.dependency_overrides[get_data_loader] = lambda: EmptyDataLoader()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def recommendations_client() -> TestClient:
    """Dedicated fixture for recommendations endpoint tests."""
    app.dependency_overrides[get_data_loader] = lambda: MockDataLoader()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def client(test_client: TestClient) -> TestClient:
    """Backward-compatible fixture name used by existing backend tests."""
    return test_client
