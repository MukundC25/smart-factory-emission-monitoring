"""Unit tests for PollutionRiskScorer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommendations.risk_scorer import PollutionRiskScorer


def test_score_parameter_good_range() -> None:
    """Low pollutant values should map into score band 0-2."""
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    score = PollutionRiskScorer.score_parameter(6.0, thresholds)
    assert 0.0 <= score <= 2.0


def test_score_parameter_severe_range() -> None:
    """Very high pollutant values should map into score band 8-10."""
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    score = PollutionRiskScorer.score_parameter(170.0, thresholds)
    assert 8.0 <= score <= 10.0


def test_compute_factory_risk_returns_all_fields() -> None:
    """Factory risk output includes all required fields and pollutant scores."""
    scorer = PollutionRiskScorer()
    factory_row = pd.Series(
        {
            "factory_id": "F001",
            "factory_name": "Test Steel Works",
            "industry_type": "steel",
            "city": "Pune",
        }
    )
    pollution_row = pd.Series(
        {
            "pm25": 30.0,
            "pm10": 110.0,
            "so2": 70.0,
            "no2": 60.0,
            "co": 1.2,
            "o3": 45.0,
        }
    )

    result = scorer.compute_factory_risk(factory_row, pollution_row)

    expected_keys = {
        "factory_id",
        "factory_name",
        "industry_type",
        "city",
        "pm25_score",
        "pm10_score",
        "so2_score",
        "no2_score",
        "co_score",
        "o3_score",
        "composite_score",
        "risk_level",
        "dominant_pollutant",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["factory_id"] == "F001"
    assert result["factory_name"] == "Test Steel Works"


def test_compute_factory_risk_all_nan_returns_unknown() -> None:
    scorer = PollutionRiskScorer()
    factory = {
        "factory_id": "f1",
        "factory_name": "Test",
        "industry_type": "steel",
        "city": "Mumbai",
        "latitude": 19.0,
        "longitude": 72.8,
    }
    pollution = {
        "pm25": np.nan,
        "pm10": np.nan,
        "so2": np.nan,
        "no2": np.nan,
        "co": np.nan,
        "o3": np.nan,
    }
    result = scorer.compute_factory_risk(pd.Series(factory), pd.Series(pollution))
    assert result["risk_level"] == "Unknown"
    assert result["dominant_pollutant"] == "Unknown"


def test_compute_factory_risk_empty_pollution_does_not_crash() -> None:
    scorer = PollutionRiskScorer()
    factory = {
        "factory_id": "f1",
        "factory_name": "Test",
        "industry_type": "steel",
        "city": "Mumbai",
        "latitude": 19.0,
        "longitude": 72.8,
    }
    result = scorer.compute_factory_risk(pd.Series(factory), pd.Series({}))
    assert "risk_level" in result
    assert result["risk_level"] == "Unknown"
