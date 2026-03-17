"""OpenAQ API service for fetching real-time AQI data."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests

LOGGER = logging.getLogger(__name__)


class OpenAQService:
    """Service for fetching real-time AQI data from OpenAQ API."""

    BASE_URL = "https://api.openaq.org/v2"
    CACHE_TTL = 300  # 5 minutes in seconds

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAQ service.

        Args:
            api_key: Optional OpenAQ API key for higher rate limits.
        """
        self.api_key = api_key or os.getenv("OPENAQ_API_KEY")
        self._cache: Dict[str, tuple] = {}

    def get_latest_aqi(self, city: str, lat: float, lon: float) -> Dict[str, Any]:
        """Get latest AQI data for a location.

        This method performs a synchronous HTTP request using ``requests`` and
        will block while waiting for the OpenAQ API response.

        When calling this service from an async context (e.g., an async FastAPI
        endpoint), you **must** run it in a thread or process executor
        (such as ``loop.run_in_executor``) to avoid blocking the event loop.

        See the usage pattern in ``tree_calculator.py`` for an example.

        Args:
            city: City name for fallback lookup.
            lat: Latitude coordinate.
            lon: Longitude coordinate.

        Returns:
            Dict containing AQI data with keys: aqi, pm25, category, source.
        """
        cache_key = f"{city}_{lat}_{lon}"

        # Check cache
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.CACHE_TTL):
                LOGGER.info("Returning cached AQI data for %s", city)
                return cached_data

        # Fetch from API
        try:
            response = requests.get(
                f"{self.BASE_URL}/latest",
                params={
                    "coordinates": f"{lat},{lon}",
                    "radius": 10000,  # 10km radius
                    "limit": 1,
                },
                headers={"X-API-Key": self.api_key} if self.api_key else {},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                LOGGER.warning("No data found for %s from OpenAQ", city)
                return self._fallback_aqi(city)

            # Parse and calculate AQI
            measurements = data["results"][0].get("measurements", [])
            aqi_data = self._calculate_aqi(measurements)
            aqi_data["source"] = "openaq"

            # Cache result
            self._cache[cache_key] = (aqi_data, datetime.now())
            LOGGER.info("Fetched and cached AQI data for %s: AQI=%s", city, aqi_data["aqi"])
            return aqi_data

        except requests.exceptions.RequestException as e:
            LOGGER.error("OpenAQ API request error for %s: %s", city, e)
            return self._fallback_aqi(city)
        except Exception as e:
            LOGGER.error("OpenAQ API error for %s: %s", city, e)
            return self._fallback_aqi(city)

    def _calculate_aqi(self, measurements: list) -> Dict[str, Any]:
        """Calculate AQI from PM2.5/PM10 measurements.

        Args:
            measurements: List of measurement dicts with parameter and value.

        Returns:
            Dict with aqi, pm25, and category.
        """
        pm25 = next(
            (m["value"] for m in measurements if m.get("parameter") == "pm25"),
            None,
        )
        pm10 = next(
            (m["value"] for m in measurements if m.get("parameter") == "pm10"),
            None,
        )

        # Use PM2.5 if available, otherwise PM10
        if pm25 is not None:
            aqi = self._pm25_to_aqi(pm25)
            return {
                "aqi": aqi,
                "pm25": pm25,
                "pm10": pm10,
                "category": self._aqi_category(aqi),
            }
        elif pm10 is not None:
            aqi = self._pm10_to_aqi(pm10)
            return {
                "aqi": aqi,
                "pm25": pm25,
                "pm10": pm10,
                "category": self._aqi_category(aqi),
            }

        return {"aqi": 0, "pm25": None, "pm10": None, "category": "Unknown"}

    @staticmethod
    def _pm25_to_aqi(pm25: float) -> int:
        """Convert PM2.5 concentration to EPA AQI.

        Args:
            pm25: PM2.5 concentration in μg/m³.

        Returns:
            AQI value (0-500).
        """
        # EPA AQI breakpoints for PM2.5
        if pm25 <= 12.0:
            return int((50 / 12.0) * pm25)
        elif pm25 <= 35.4:
            return int(50 + ((100 - 50) / (35.4 - 12.1)) * (pm25 - 12.1))
        elif pm25 <= 55.4:
            return int(100 + ((150 - 100) / (55.4 - 35.5)) * (pm25 - 35.5))
        elif pm25 <= 150.4:
            return int(150 + ((200 - 150) / (150.4 - 55.5)) * (pm25 - 55.5))
        elif pm25 <= 250.4:
            return int(200 + ((300 - 200) / (250.4 - 150.5)) * (pm25 - 150.5))
        elif pm25 <= 350.4:
            return int(300 + ((400 - 300) / (350.4 - 250.5)) * (pm25 - 250.5))
        elif pm25 <= 500.4:
            return int(400 + ((500 - 400) / (500.4 - 350.5)) * (pm25 - 350.5))
        return 500

    @staticmethod
    def _pm10_to_aqi(pm10: float) -> int:
        """Convert PM10 concentration to EPA AQI.

        Args:
            pm10: PM10 concentration in μg/m³.

        Returns:
            AQI value (0-500).
        """
        # EPA AQI breakpoints for PM10
        if pm10 <= 54:
            return int((50 / 54) * pm10)
        elif pm10 <= 154:
            return int(50 + ((100 - 50) / (154 - 55)) * (pm10 - 55))
        elif pm10 <= 254:
            return int(100 + ((150 - 100) / (254 - 155)) * (pm10 - 155))
        elif pm10 <= 354:
            return int(150 + ((200 - 150) / (354 - 255)) * (pm10 - 255))
        elif pm10 <= 424:
            return int(200 + ((300 - 200) / (424 - 355)) * (pm10 - 355))
        elif pm10 <= 504:
            return int(300 + ((400 - 300) / (504 - 425)) * (pm10 - 425))
        elif pm10 <= 604:
            return int(400 + ((500 - 400) / (604 - 505)) * (pm10 - 505))
        return 500

    @staticmethod
    def _aqi_category(aqi: int) -> str:
        """Get AQI category name.

        Args:
            aqi: AQI value.

        Returns:
            Category name (Good, Moderate, Unhealthy for Sensitive, etc.).
        """
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        return "Hazardous"

    def _fallback_aqi(self, city: str) -> Dict[str, Any]:
        """Return fallback AQI data when API fails.

        Args:
            city: City name for reference.

        Returns:
            Dict with default AQI data marked as fallback.
        """
        LOGGER.warning("Using fallback AQI data for %s", city)
        return {
            "aqi": 0,
            "pm25": None,
            "pm10": None,
            "category": "Unknown",
            "source": "fallback",
        }

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        LOGGER.info("Cache cleared")


# Singleton instance for dependency injection
_openaq_service: Optional[OpenAQService] = None


def get_openaq_service() -> OpenAQService:
    """Get or create OpenAQ service singleton.

    Returns:
        OpenAQService instance.
    """
    global _openaq_service
    if _openaq_service is None:
        _openaq_service = OpenAQService()
    return _openaq_service


def set_openaq_service(service: OpenAQService) -> None:
    """Set the OpenAQ service singleton (for testing).

    Args:
        service: OpenAQService instance to use.
    """
    global _openaq_service
    _openaq_service = service
