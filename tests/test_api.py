"""Basic API tests for backend route availability."""

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root_endpoint() -> None:
	"""Ensure API root endpoint responds successfully."""
	response = client.get("/")
	assert response.status_code == 200
	assert "Smart Factory" in response.json()["message"]


def test_factory_route() -> None:
	"""Ensure factory route is wired and returns payload."""
	response = client.get("/factories/")
	assert response.status_code == 200
	assert "data" in response.json()


def test_pollution_route() -> None:
	"""Ensure pollution route is wired and returns payload."""
	response = client.get("/pollution/")
	assert response.status_code == 200
	assert "data" in response.json()


def test_recommendation_route() -> None:
	"""Ensure recommendation route accepts path parameter."""
	response = client.get("/recommendation/101")
	assert response.status_code == 200
	assert response.json()["factory_id"] == 101
