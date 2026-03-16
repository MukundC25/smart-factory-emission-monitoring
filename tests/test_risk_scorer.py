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


def test_score_parameter_moderate_range_returns_2_to_5() -> None:
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    score = PollutionRiskScorer.score_parameter(25.0, thresholds)
    assert 2.0 <= score <= 5.0


def test_score_parameter_poor_range_returns_5_to_8() -> None:
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    score = PollutionRiskScorer.score_parameter(45.0, thresholds)
    assert 5.0 <= score <= 8.0


def test_score_parameter_severe_range() -> None:
    """Very high pollutant values should map into score band 8-10."""
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    score = PollutionRiskScorer.score_parameter(170.0, thresholds)
    assert 8.0 <= score <= 10.0


def test_score_parameter_zero_returns_zero() -> None:
    thresholds = {"good": 12, "moderate": 35, "poor": 55, "severe": 150}
    assert PollutionRiskScorer.score_parameter(0.0, thresholds) == 0.0


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


def test_compute_factory_risk_correct_risk_level_low() -> None:
    scorer = PollutionRiskScorer()
    factory = pd.Series({"factory_id": "f1", "factory_name": "Test", "industry_type": "steel", "city": "Pune"})
    pollution = pd.Series({"pm25": 5.0, "pm10": 10.0, "so2": 5.0, "no2": 5.0, "co": 0.2, "o3": 5.0})
    result = scorer.compute_factory_risk(factory, pollution)
    assert result["risk_level"] == "Low"


def test_compute_factory_risk_correct_risk_level_high() -> None:
    scorer = PollutionRiskScorer()
    factory = pd.Series({"factory_id": "f1", "factory_name": "Test", "industry_type": "steel", "city": "Pune"})
    pollution = pd.Series({"pm25": 120.0, "pm10": 280.0, "so2": 220.0, "no2": 160.0, "co": 12.0, "o3": 190.0})
    result = scorer.compute_factory_risk(factory, pollution)
    assert result["risk_level"] in {"High", "Critical"}


def test_compute_factory_risk_correct_risk_level_critical() -> None:
    scorer = PollutionRiskScorer()
    factory = pd.Series({"factory_id": "f1", "factory_name": "Test", "industry_type": "steel", "city": "Pune"})
    pollution = pd.Series({"pm25": 500.0, "pm10": 600.0, "so2": 800.0, "no2": 300.0, "co": 20.0, "o3": 250.0})
    result = scorer.compute_factory_risk(factory, pollution)
    assert result["risk_level"] == "Critical"


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


def test_compute_factory_risk_dominant_pollutant_is_highest_scorer() -> None:
    scorer = PollutionRiskScorer()
    factory = pd.Series({"factory_id": "f1", "factory_name": "Test", "industry_type": "steel", "city": "Pune"})
    pollution = pd.Series({"pm25": 5.0, "pm10": 5.0, "so2": 1000.0, "no2": 5.0, "co": 0.1, "o3": 5.0})
    result = scorer.compute_factory_risk(factory, pollution)
    assert result["dominant_pollutant"] == "so2"


def test_compute_factory_risk_all_nan_dominant_pollutant_is_unknown() -> None:
    scorer = PollutionRiskScorer()
    factory = pd.Series({"factory_id": "f1", "factory_name": "Test", "industry_type": "steel", "city": "Pune"})
    pollution = pd.Series({"pm25": np.nan, "pm10": np.nan, "so2": np.nan, "no2": np.nan, "co": np.nan, "o3": np.nan})
    result = scorer.compute_factory_risk(factory, pollution)
    assert result["dominant_pollutant"] == "Unknown"


def test_score_all_factories_returns_dataframe() -> None:
    scorer = PollutionRiskScorer()
    factories_df = pd.DataFrame([
        {"factory_id": "FAC1", "factory_name": "A", "industry_type": "steel", "city": "Pune"},
        {"factory_id": "FAC2", "factory_name": "B", "industry_type": "chemical", "city": "Mumbai"},
    ])
    pollution_df = pd.DataFrame([
        {"city": "Pune", "nearest_factory_distance_km": 2.0, "pm25": 40.0, "pm10": 70.0, "so2": 20.0, "no2": 15.0, "co": 0.8, "o3": 30.0},
        {"city": "Mumbai", "nearest_factory_distance_km": 3.0, "pm25": 60.0, "pm10": 90.0, "so2": 50.0, "no2": 30.0, "co": 1.2, "o3": 40.0},
    ])
    out = scorer.score_all_factories(factories_df, pollution_df)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2


def test_score_all_factories_empty_pollution_uses_fallback() -> None:
    scorer = PollutionRiskScorer()
    factories_df = pd.DataFrame([
        {"factory_id": "FAC1", "factory_name": "A", "industry_type": "steel", "city": "Pune"},
    ])
    out = scorer.score_all_factories(factories_df, pd.DataFrame())
    assert len(out) == 1
    assert out.iloc[0]["risk_level"] == "Unknown"


def test_score_all_factories_missing_columns_uses_fallback() -> None:
    scorer = PollutionRiskScorer()
    factories_df = pd.DataFrame([
        {"factory_id": "FAC1", "factory_name": "A", "industry_type": "steel", "city": "Pune"},
    ])
    pollution_df = pd.DataFrame([{"pm25": 40.0}])
    out = scorer.score_all_factories(factories_df, pollution_df)
    assert len(out) == 1
