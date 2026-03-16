"""Unit tests for geographic helper utilities."""

from __future__ import annotations

from backend.utils.geo_utils import bounding_box, haversine_km


def test_haversine_known_distance_pune_mumbai() -> None:
    distance = haversine_km(18.5204, 73.8567, 19.0760, 72.8777)
    assert 115 <= distance <= 170


def test_haversine_same_point_returns_zero() -> None:
    assert haversine_km(18.52, 73.85, 18.52, 73.85) == 0.0


def test_bounding_box_returns_correct_bounds() -> None:
    min_lat, max_lat, min_lon, max_lon = bounding_box(18.52, 73.85, 50)
    assert min_lat < max_lat
    assert min_lon < max_lon


def test_bounding_box_near_pole_does_not_raise() -> None:
    min_lat, max_lat, min_lon, max_lon = bounding_box(89.9, 0.0, 10)
    assert min_lat < max_lat
    assert min_lon < max_lon


def test_bounding_box_negative_lat_lon() -> None:
    min_lat, max_lat, min_lon, max_lon = bounding_box(-33.86, -151.2, 20)
    assert min_lat < max_lat
    assert min_lon < max_lon
