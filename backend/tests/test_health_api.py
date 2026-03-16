"""Dedicated tests for root and health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_200(test_client: TestClient) -> None:
    assert test_client.get("/health").status_code == 200


def test_health_response_has_status_ok(test_client: TestClient) -> None:
    assert test_client.get("/health").json()["status"] == "ok"


def test_health_response_has_datasets_loaded(test_client: TestClient) -> None:
    body = test_client.get("/health").json()
    assert "datasets_loaded" in body


def test_health_response_has_timestamp(test_client: TestClient) -> None:
    body = test_client.get("/health").json()
    assert "timestamp" in body


def test_root_returns_200_with_endpoint_list(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "endpoints" in body
    assert isinstance(body["endpoints"], list)
