"""Unit tests for TreePlantingCalculator and OpenAQClient.

All external HTTP calls are mocked — no real network activity.
Tests are pure unit tests using validated expected values derived from the
CPCB India AQI breakpoints and the formula constants defined in
src.recommendations.tree_calculator.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.recommendations.openaq_client import OpenAQClient
from src.recommendations.tree_calculator import (
    AQI_GOOD,
    MATURITY_YEARS,
    TreePlantingCalculator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_calc = TreePlantingCalculator()


# ===========================================================================
# 1–3  AQI formula (calculate_aqi_from_pm25)
# ===========================================================================


def test_calculate_aqi_from_pm25_good_range():
    """PM2.5 = 15 μg/m³ is in the first CPCB band (0–30 → AQI 0–50)."""
    # Linear: (50-0)/(30-0) * (15-0) + 0 = 25.0
    aqi = OpenAQClient.calculate_aqi_from_pm25(15.0)
    assert round(aqi) == 25


def test_calculate_aqi_from_pm25_severe_range():
    """PM2.5 > 500 μg/m³ (beyond all breakpoints) should return 500."""
    assert OpenAQClient.calculate_aqi_from_pm25(600.0) == 500.0


def test_calculate_aqi_from_pm25_interpolates_within_band():
    """PM2.5 = 45 μg/m³ falls in band 30–60 → AQI 51–100.

    Expected: (100-51)/(60-30) * (45-30) + 51 ≈ 75.5
    """
    aqi = OpenAQClient.calculate_aqi_from_pm25(45.0)
    expected = (100 - 51) / (60 - 30) * (45 - 30) + 51  # ≈ 75.5
    assert abs(aqi - expected) < 0.01


# ===========================================================================
# 4–5  PM2.5 trees formula (calculate_trees_for_pm25)
# ===========================================================================


def test_calculate_trees_for_pm25_zero_reduction_needed():
    """No trees needed when current PM2.5 already equals the target."""
    trees = _calc.calculate_trees_for_pm25(
        current_pm25=50.0,
        target_pm25=50.0,
        impact_area_km2=5.0,
    )
    assert trees == 0.0


def test_calculate_trees_for_pm25_positive_reduction():
    """Trees needed must be > 0 when current PM2.5 exceeds target."""
    trees = _calc.calculate_trees_for_pm25(
        current_pm25=65.0,
        target_pm25=30.0,
        impact_area_km2=1.0,
    )
    assert trees > 0
    # Spot-check: (65-30 μg/m³ * 1e6 m²) / (0.8 * 100) = 437 500
    expected = (65.0 - 30.0) * 1_000_000 / (0.8 * 100)
    assert abs(trees - expected) < 1.0


# ===========================================================================
# 6–7  Impact radius  (calculate_impact_radius)
# ===========================================================================


def test_calculate_impact_radius_scales_with_score():
    """Higher pollution score → larger impact radius."""
    r_low = _calc.calculate_impact_radius(1.0)
    r_high = _calc.calculate_impact_radius(5.0)
    assert r_high > r_low


def test_calculate_impact_radius_capped_at_5km():
    """Impact radius is hard-capped at 5.0 km regardless of score."""
    radius = _calc.calculate_impact_radius(100.0)
    assert radius == 5.0


# ===========================================================================
# 8  Target AQI step-down  (determine_target_aqi)
# ===========================================================================


def test_determine_target_aqi_steps_down_one_band():
    """Verify one-band step-down logic across all CPCB AQI bands."""
    # Severe (> 400) → Poor (300)
    assert _calc.determine_target_aqi(450.0) == 300.0
    # Very Poor (301–400) → Moderate (200)
    assert _calc.determine_target_aqi(350.0) == 200.0
    # Poor (201–300) → Satisfactory (100)
    assert _calc.determine_target_aqi(250.0) == 100.0
    # Moderate or below → Good (50)
    assert _calc.determine_target_aqi(150.0) == float(AQI_GOOD)
    assert _calc.determine_target_aqi(40.0) == float(AQI_GOOD)


# ===========================================================================
# 9–11  Feasibility  (assess_feasibility)
# ===========================================================================


def test_assess_feasibility_returns_correct_level():
    """Feasibility classifier should return High/Medium/Low across thresholds."""
    assert _calc.assess_feasibility(trees_recommended=200, planting_area_hectares=2.0) == "High"
    assert _calc.assess_feasibility(trees_recommended=1000, planting_area_hectares=10.0) == "Medium"
    assert _calc.assess_feasibility(trees_recommended=3000, planting_area_hectares=30.0) == "Low"


# ===========================================================================
# 12  Buffer ordering  (calculate_trees_needed internals)
# ===========================================================================


def test_buffers_produce_correct_multipliers():
    """Buffer tiers should follow minimum*1.2 and minimum*1.5 (ceil-rounded)."""
    rec = _calc.calculate_trees_needed(
        factory_id="TST001",
        city="TestCity",
        current_aqi=220.0,
        pollution_score=6.0,
        dominant_pollutant="pm25",
        pollution_readings={"pm25": 80.0, "pm10": 110.0},
    )
    t = rec.trees_needed
    assert t["minimum"] <= t["recommended"] <= t["optimal"]
    assert t["recommended"] == math.ceil(t["minimum"] * 1.2)
    assert t["optimal"] == math.ceil(t["minimum"] * 1.5)


# ===========================================================================
# 13  Full dataclass fields  (calculate_trees_needed)
# ===========================================================================


def test_full_calculation_returns_dataclass():
    """All key fields in TreeRecommendation must be populated after a full run."""
    rec = _calc.calculate_trees_needed(
        factory_id="FAC001",
        city="Pune",
        current_aqi=178.0,
        pollution_score=7.5,
        dominant_pollutant="pm25",
        pollution_readings={"pm25": 65.0, "pm10": 90.0, "no2": 30.0},
    )
    assert rec.factory_id == "FAC001"
    assert rec.city == "Pune"
    assert rec.target_aqi > 0
    assert rec.impact_radius_km > 0
    assert rec.planting_area_hectares > 0
    assert rec.annual_co2_offset_tons > 0
    assert rec.timeline_years == MATURITY_YEARS
    assert rec.feasibility in ("High", "Medium", "Low")
    assert len(rec.notes) >= 3
    assert rec.trees_needed["recommended"] >= rec.trees_needed["minimum"]


# ===========================================================================
# 14–15  Dominant pollutant selection
# ===========================================================================


def test_openaq_client_parses_pm25_correctly():
    """When dominant_pollutant='pm25', pm25 formula should drive the base count."""
    rec_pm25 = _calc.calculate_trees_needed(
        factory_id="FAC_A",
        city="City",
        current_aqi=200.0,
        pollution_score=5.0,
        dominant_pollutant="pm25",
        pollution_readings={"pm25": 60.0},
    )
    # pollutant_breakdown pm25_trees should be set
    assert rec_pm25.pollutant_breakdown["pm25_trees"] is not None
    assert rec_pm25.pollutant_breakdown["pm25_trees"] > 0


def test_dominant_no2_drives_recommendation():
    """When dominant_pollutant='no2', no2 formula should drive the base count."""
    rec_no2 = _calc.calculate_trees_needed(
        factory_id="FAC_B",
        city="City",
        current_aqi=200.0,
        pollution_score=5.0,
        dominant_pollutant="no2",
        pollution_readings={"no2": 50.0},
    )
    assert rec_no2.pollutant_breakdown["no2_trees"] is not None
    assert rec_no2.pollutant_breakdown["no2_trees"] > 0


# ===========================================================================
# 16  OpenAQ fallback on timeout
# ===========================================================================


def test_openaq_client_returns_fallback_on_timeout():
    """get_city_aqi must return a usable dict (not raise) when the HTTP call times out."""
    with patch("requests.Session.get", side_effect=requests.exceptions.Timeout):
        client = OpenAQClient()
        result = client.get_city_aqi(city="Pune", lat=18.52, lon=73.85)

    # Must return a dict with at minimum an 'aqi' key
    assert isinstance(result, dict)
    assert "aqi" in result
    assert result.get("source") == "fallback"


# ===========================================================================
# 17  OpenAQ pm2.5 parameter name normalisation
# ===========================================================================


def test_extract_pollutant_values_handles_pm2_5_naming():
    """OpenAQ sometimes returns 'pm2.5'; extract_pollutant_values must normalise it."""
    raw_response = {
        "results": [
            {
                "measurements": [
                    {"parameter": "pm2.5", "value": 72.0},
                    {"parameter": "pm10", "value": 95.0},
                ]
            }
        ]
    }
    client = OpenAQClient()
    values = client.extract_pollutant_values(raw_response)
    assert values["pm25"] == 72.0
    assert values["pm10"] == 95.0


def test_openaq_client_handles_429_gracefully():
    """429 from OpenAQ should be swallowed and converted to fallback output."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.raise_for_status.return_value = None
    with patch("requests.Session.get", return_value=mock_response):
        client = OpenAQClient()
        result = client.get_city_aqi(city="Pune", lat=18.52, lon=73.85)

    assert isinstance(result, dict)
    assert result.get("source") == "fallback"
    assert result.get("aqi") == 100.0
