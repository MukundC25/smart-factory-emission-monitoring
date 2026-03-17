"""Unit tests for OpenAQ service AQI calculations."""

import pytest
from backend.app.services.openaq_service import OpenAQService


class TestOpenAQService:
    """Test suite for OpenAQService."""

    @pytest.fixture
    def service(self):
        """Create a fresh OpenAQService instance for each test."""
        return OpenAQService(api_key=None)

    def test_pm25_to_aqi_good(self, service):
        """Test PM2.5 AQI calculation for Good category (0-50)."""
        # PM2.5 = 6.0 μg/m³ should give AQI ≈ 25
        aqi = service._pm25_to_aqi(6.0)
        assert 0 <= aqi <= 50
        assert aqi == 25

    def test_pm25_to_aqi_moderate(self, service):
        """Test PM2.5 AQI calculation for Moderate category (51-100)."""
        # PM2.5 = 25.0 μg/m³ should give AQI in moderate range
        aqi = service._pm25_to_aqi(25.0)
        assert 51 <= aqi <= 100

    def test_pm25_to_aqi_unhealthy_sensitive(self, service):
        """Test PM2.5 AQI calculation for Unhealthy for Sensitive (101-150)."""
        # PM2.5 = 45.0 μg/m³ should give AQI in unhealthy for sensitive range
        aqi = service._pm25_to_aqi(45.0)
        assert 101 <= aqi <= 150

    def test_pm25_to_aqi_unhealthy(self, service):
        """Test PM2.5 AQI calculation for Unhealthy category (151-200)."""
        # PM2.5 = 100.0 μg/m³ should give AQI in unhealthy range
        aqi = service._pm25_to_aqi(100.0)
        assert 151 <= aqi <= 200

    def test_pm25_to_aqi_very_unhealthy(self, service):
        """Test PM2.5 AQI calculation for Very Unhealthy category (201-300)."""
        # PM2.5 = 200.0 μg/m³ should give AQI in very unhealthy range
        aqi = service._pm25_to_aqi(200.0)
        assert 201 <= aqi <= 300

    def test_pm25_to_aqi_hazardous(self, service):
        """Test PM2.5 AQI calculation for Hazardous category (301+)."""
        # PM2.5 = 400.0 μg/m³ should give AQI in hazardous range
        aqi = service._pm25_to_aqi(400.0)
        assert aqi >= 301

    def test_pm25_to_aqi_zero(self, service):
        """Test PM2.5 AQI calculation for zero concentration."""
        aqi = service._pm25_to_aqi(0.0)
        assert aqi == 0

    def test_pm25_to_aqi_max_cap(self, service):
        """Test PM2.5 AQI calculation maxes at 500."""
        # Very high PM2.5 should cap at 500
        aqi = service._pm25_to_aqi(1000.0)
        assert aqi == 500

    def test_pm10_to_aqi_good(self, service):
        """Test PM10 AQI calculation for Good category."""
        aqi = service._pm10_to_aqi(25.0)
        assert 0 <= aqi <= 50

    def test_pm10_to_aqi_moderate(self, service):
        """Test PM10 AQI calculation for Moderate category."""
        aqi = service._pm10_to_aqi(100.0)
        assert 51 <= aqi <= 100

    def test_pm10_to_aqi_unhealthy_sensitive(self, service):
        """Test PM10 AQI calculation for Unhealthy for Sensitive category."""
        aqi = service._pm10_to_aqi(200.0)
        assert 101 <= aqi <= 150

    def test_pm10_to_aqi_unhealthy(self, service):
        """Test PM10 AQI calculation for Unhealthy category."""
        aqi = service._pm10_to_aqi(300.0)
        assert 151 <= aqi <= 200

    def test_pm10_to_aqi_max(self, service):
        """Test PM10 AQI calculation maxes at 500."""
        aqi = service._pm10_to_aqi(700.0)
        assert aqi == 500

    def test_aqi_category_good(self, service):
        """Test AQI category for Good."""
        assert service._aqi_category(0) == "Good"
        assert service._aqi_category(25) == "Good"
        assert service._aqi_category(50) == "Good"

    def test_aqi_category_moderate(self, service):
        """Test AQI category for Moderate."""
        assert service._aqi_category(51) == "Moderate"
        assert service._aqi_category(75) == "Moderate"
        assert service._aqi_category(100) == "Moderate"

    def test_aqi_category_unhealthy_sensitive(self, service):
        """Test AQI category for Unhealthy for Sensitive Groups."""
        assert service._aqi_category(101) == "Unhealthy for Sensitive Groups"
        assert service._aqi_category(125) == "Unhealthy for Sensitive Groups"
        assert service._aqi_category(150) == "Unhealthy for Sensitive Groups"

    def test_aqi_category_unhealthy(self, service):
        """Test AQI category for Unhealthy."""
        assert service._aqi_category(151) == "Unhealthy"
        assert service._aqi_category(175) == "Unhealthy"
        assert service._aqi_category(200) == "Unhealthy"

    def test_aqi_category_very_unhealthy(self, service):
        """Test AQI category for Very Unhealthy."""
        assert service._aqi_category(201) == "Very Unhealthy"
        assert service._aqi_category(250) == "Very Unhealthy"
        assert service._aqi_category(300) == "Very Unhealthy"

    def test_aqi_category_hazardous(self, service):
        """Test AQI category for Hazardous."""
        assert service._aqi_category(301) == "Hazardous"
        assert service._aqi_category(400) == "Hazardous"
        assert service._aqi_category(500) == "Hazardous"

    def test_calculate_aqi_with_pm25(self, service):
        """Test AQI calculation when PM2.5 is available."""
        measurements = [
            {"parameter": "pm25", "value": 12.0},
            {"parameter": "pm10", "value": 30.0},
        ]
        result = service._calculate_aqi(measurements)
        assert "aqi" in result
        assert "pm25" in result
        assert result["pm25"] == 12.0
        assert result["category"] == service._aqi_category(result["aqi"])

    def test_calculate_aqi_with_pm10_only(self, service):
        """Test AQI calculation when only PM10 is available."""
        measurements = [
            {"parameter": "pm10", "value": 54.0},
        ]
        result = service._calculate_aqi(measurements)
        assert "aqi" in result
        assert result["pm25"] is None
        assert result["pm10"] == 54.0
        assert result["category"] == service._aqi_category(result["aqi"])

    def test_calculate_aqi_no_data(self, service):
        """Test AQI calculation when no PM data is available."""
        measurements = []
        result = service._calculate_aqi(measurements)
        assert result["aqi"] == 0
        assert result["pm25"] is None
        assert result["pm10"] is None
        assert result["category"] == "Unknown"

    def test_cache_functionality(self, service, monkeypatch):
        """Test that caching works correctly."""
        # Mock the underlying HTTP call so no real network request is made
        call_counter = {"count": 0}

        class DummyResponse:
            status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {
                            "measurements": [
                                {"parameter": "pm25", "value": 12.0},
                                {"parameter": "pm10", "value": 30.0},
                            ]
                        }
                    ]
                }

        def fake_get(*args, **kwargs):
            call_counter["count"] += 1
            return DummyResponse()

        # Patch the requests.get used inside OpenAQService
        monkeypatch.setattr(
            "backend.app.services.openaq_service.requests.get",
            fake_get,
        )

        # First call should populate the cache and hit the mocked API once
        result1 = service.get_latest_aqi("TestCity", 0.0, 0.0)

        # Second call with same parameters should use the cache
        result2 = service.get_latest_aqi("TestCity", 0.0, 0.0)

        # Verify that the external HTTP call was made only once
        assert call_counter["count"] == 1

        # Cached result should be the same as the first result
        assert result1 == result2

        # And the cache structure should exist and be a dictionary
        assert hasattr(service, "_cache")
        assert isinstance(service._cache, dict)

    def test_clear_cache(self, service):
        """Test cache clearing."""
        # Add a mock entry to cache
        from datetime import datetime
        service._cache["test_key"] = ({"aqi": 50}, datetime.now())
        service.clear_cache()
        assert len(service._cache) == 0

    def test_fallback_aqi(self, service):
        """Test fallback AQI when API fails."""
        result = service._fallback_aqi("TestCity")
        assert result["aqi"] == 0
        assert result["pm25"] is None
        assert result["pm10"] is None
        assert result["category"] == "Unknown"
        assert result["source"] == "fallback"

    def test_service_singleton(self):
        """Test that get_openaq_service returns singleton."""
        from backend.app.services.openaq_service import get_openaq_service, set_openaq_service

        # Preserve original singleton to avoid leaking state to other tests
        original_service = get_openaq_service()

        try:
            # Create and set a service
            service1 = OpenAQService()
            set_openaq_service(service1)

            # Get should return same instance
            service2 = get_openaq_service()
            assert service1 is service2
        finally:
            # Restore original singleton
            set_openaq_service(original_service)

    def test_service_initialization_with_api_key(self):
        """Test OpenAQService initialization with API key."""
        service = OpenAQService(api_key="test_key")
        assert service.api_key == "test_key"
        assert service.CACHE_TTL == 300
        assert service.BASE_URL == "https://api.openaq.org/v2"
