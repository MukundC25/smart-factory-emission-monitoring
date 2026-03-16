"""OpenAQ API v2 client with graceful fallback behaviour.

Fetches real-time air-quality measurements for a given coordinate pair.
Never raises — all failure modes return a usable defaults dict so that
upstream callers can always proceed.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.openaq.org/v2"
TIMEOUT = 10
MAX_RADIUS_M = 25_000  # 25 km

# CPCB India PM2.5 AQI breakpoints: (pm25_lo, pm25_hi, aqi_lo, aqi_hi)
_PM25_BREAKPOINTS = [
    (0.0,   29.999,   0,   50),
    (30.0,  59.999,  51,  100),
    (60.0,  89.999, 101,  200),
    (90.0, 119.999, 201,  300),
    (120.0, 249.999, 301, 400),
    (250.0, 500.0, 401, 500),
]


class OpenAQClient:
    """HTTP client for the OpenAQ REST API v2.

    Args:
        api_key: Optional OpenAQ API key — reads ``OPENAQ_API_KEY`` env var if
                 not supplied. V2 works without a key (lower rate limits).
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        resolved_key = api_key or os.getenv("OPENAQ_API_KEY")
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "smart-factory-emission-monitor/1.0"})
        if resolved_key:
            self._session.headers["X-API-Key"] = resolved_key

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_latest_measurements(
        self,
        lat: float,
        lon: float,
        radius_m: int = 10_000,
    ) -> dict:
        """Fetch the latest measurements near (lat, lon) from OpenAQ.

        Args:
            lat: Latitude.
            lon: Longitude.
            radius_m: Search radius in metres (max 25 000).

        Returns:
            Raw API response dict, or empty dict on any error.
        """
        radius_m = min(radius_m, MAX_RADIUS_M)
        url = f"{BASE_URL}/latest"
        params: Dict[str, object] = {
            "coordinates": f"{lat},{lon}",
            "radius": radius_m,
            "limit": 10,
        }
        try:
            response = self._session.get(url, params=params, timeout=TIMEOUT)
            if response.status_code == 429:
                logger.warning("OpenAQ rate-limited (429) — returning empty dict")
                return {}
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning("OpenAQ request timed out for (%.4f, %.4f)", lat, lon)
            return {}
        except requests.exceptions.ConnectionError:
            logger.warning("OpenAQ connection error for (%.4f, %.4f)", lat, lon)
            return {}
        except Exception as exc:  # pragma: no cover — catch-all guard
            logger.warning("OpenAQ unexpected error: %s", exc)
            return {}

    def extract_pollutant_values(self, measurements: dict) -> dict:
        """Parse raw OpenAQ response and extract the latest value per parameter.

        Handles the OpenAQ v2 convention for both ``pm25`` and ``pm2.5`` parameter
        naming.

        Args:
            measurements: Raw OpenAQ API response dict.

        Returns:
            Dict with optional float values for pm25, pm10, no2, so2, co, o3,
            aqi_index.
        """
        result: Dict[str, Optional[float]] = {
            "pm25": None,
            "pm10": None,
            "no2":  None,
            "so2":  None,
            "co":   None,
            "o3":   None,
            "aqi_index": None,
        }

        results_list = measurements.get("results", [])
        if not results_list:
            return result

        for location in results_list:
            measurements_list = location.get("measurements", [])
            for m in measurements_list:
                raw_param: str = str(m.get("parameter", "")).lower().strip()
                # normalise pm2.5 → pm25
                param = raw_param.replace(".", "")
                if param not in result:
                    continue
                try:
                    val = float(m.get("value", float("nan")))
                except (TypeError, ValueError):
                    continue
                if val < 0 or val != val:  # negative or NaN
                    continue
                # Keep first non-None value found per parameter
                if result[param] is None:
                    result[param] = val

        return result

    @staticmethod
    def calculate_aqi_from_pm25(pm25: float) -> float:
        """Compute CPCB India AQI from PM2.5 concentration using linear interpolation.

        Args:
            pm25: PM2.5 concentration in μg/m³.

        Returns:
            Float AQI value (clamped to 0–500).
        """
        if pm25 <= 0.0:
            return 0.0

        for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
            if c_lo <= pm25 <= c_hi:
                # Linear interpolation
                aqi = (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo
                return float(aqi)

        # Beyond the last breakpoint → severe
        return 500.0

    def get_city_aqi(self, city: str, lat: float, lon: float) -> dict:
        """Return AQI and pollutant readings for a city coordinate.

        Orchestrates ``get_latest_measurements`` → ``extract_pollutant_values``
        → AQI derivation.  Never raises; always returns a usable dict.

        Priority for AQI value:
            1. ``aqi_index`` from measurements (if available)
            2. Calculated from ``pm25`` via :meth:`calculate_aqi_from_pm25`
            3. PM10-based estimate (pm10 / 2)
            4. Fallback default (``source="fallback"``)

        Args:
            city: City name (used only for logging).
            lat: Factory latitude.
            lon: Factory longitude.

        Returns:
            Dict with keys: aqi, pm25, pm10, no2, so2, co, o3, source, timestamp.
        """
        measurements = self.get_latest_measurements(lat, lon)
        values = self.extract_pollutant_values(measurements)

        # Derive AQI
        aqi: Optional[float] = None
        source = "openaq"
        if values.get("aqi_index") is not None:
            aqi = values["aqi_index"]
        elif values.get("pm25") is not None:
            aqi = self.calculate_aqi_from_pm25(values["pm25"])  # type: ignore[arg-type]
            source = "openaq_calculated"
        elif values.get("pm10") is not None:
            # rough PM10 → AQI approximation
            aqi = float(values["pm10"]) / 2.0  # type: ignore[arg-type]
            source = "openaq_estimated"

        if aqi is None:
            logger.warning(
                "No AQI data available for %s (%.4f, %.4f) — using fallback",
                city, lat, lon,
            )
            return {
                "aqi": 100.0,
                "pm25": None,
                "pm10": None,
                "no2":  None,
                "so2":  None,
                "co":   None,
                "o3":   None,
                "source": "fallback",
                "timestamp": None,
            }

        # Attempt to pull a representative timestamp
        timestamp: Optional[str] = None
        try:
            r0 = measurements.get("results", [{}])[0]
            m0 = r0.get("measurements", [{}])[0]
            timestamp = str(m0.get("lastUpdated", "")) or None
        except (IndexError, KeyError, TypeError):
            pass

        return {
            "aqi":   round(aqi, 2),
            "pm25":  values.get("pm25"),
            "pm10":  values.get("pm10"),
            "no2":   values.get("no2"),
            "so2":   values.get("so2"),
            "co":    values.get("co"),
            "o3":    values.get("o3"),
            "source": source,
            "timestamp": timestamp,
        }
