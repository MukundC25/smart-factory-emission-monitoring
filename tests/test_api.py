"""Smoke tests for the production backend API (backend.main).

These tests run against the real DataLoader (reading actual CSV/Parquet files)
to verify the full stack is wired correctly.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_root_endpoint() -> None:
    """GET / returns 200 and includes API name."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "Smart Factory" in body["name"]
    assert "endpoints" in body


def test_factory_route() -> None:
    """GET /factories returns paginated response with data key."""
    response = client.get("/factories")
    assert response.status_code == 200
    body = response.json()
    assert "data" in body
    assert "total" in body


def test_pollution_route() -> None:
    """GET /pollution returns 200 even if dataset is empty."""
    response = client.get("/pollution")
    assert response.status_code == 200
    assert "data" in response.json()


def test_health_route() -> None:
    """GET /health always returns 200 with status=ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
