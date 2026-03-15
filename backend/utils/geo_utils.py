"""Geospatial utility functions."""

from __future__ import annotations

import math
from typing import Tuple


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate great-circle distance between two GPS coordinates.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in kilometres.
    """
    earth_radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def bounding_box(
    lat: float,
    lon: float,
    radius_km: float,
) -> Tuple[float, float, float, float]:
    """Return an approximate bounding box around a centre point.

    Used as a fast pre-filter before the precise haversine_km check.

    Args:
        lat: Centre latitude in decimal degrees.
        lon: Centre longitude in decimal degrees.
        radius_km: Half-width of box in kilometres.

    Returns:
        Tuple (min_lat, max_lat, min_lon, max_lon).
    """
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return (
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )
