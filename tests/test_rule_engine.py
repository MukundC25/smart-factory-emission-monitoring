"""Unit tests for RuleEngine, HybridRecommendationEngine, and RecommendationFormatter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.recommendations.engine import FactoryReport, HybridRecommendationEngine
from src.recommendations.formatter import RecommendationFormatter
from src.recommendations.rule_engine import Recommendation, RuleEngine


@pytest.fixture()
def sample_factories_df() -> pd.DataFrame:
    """Sample factories dataframe for hybrid engine tests."""
    return pd.DataFrame(
        [
            {
                "factory_id": "F001",
                "factory_name": "Steel Plant A",
                "industry_type": "steel",
                "city": "Pune",
                "latitude": 18.52,
                "longitude": 73.85,
            }
        ]
    )


@pytest.fixture()
def sample_pollution_df() -> pd.DataFrame:
    """Sample pollution dataframe for hybrid engine tests."""
    return pd.DataFrame(
        [
            {
                "city": "Pune",
                "nearest_factory_distance_km": 4.0,
                "pm25": 65.0,
                "pm10": 170.0,
                "so2": 90.0,
                "no2": 70.0,
                "co": 1.5,
                "o3": 80.0,
            }
        ]
    )


def test_rule_engine_high_so2_returns_scrubber_recommendation() -> None:
    """High SO2 should trigger scrubber/FGD recommendation."""
    engine = RuleEngine()
    recs = engine.apply_rules(
        {
            "so2_score": 7.2,
            "pm25_score": 1.0,
            "pm10_score": 1.0,
            "no2_score": 1.0,
            "co_score": 1.0,
            "o3_score": 1.0,
            "composite_score": 4.0,
            "risk_level": "Medium",
            "industry_type": "steel",
        }
    )

    assert any("flue gas desulfurization" in rec.action.lower() or "wet scrubber" in rec.action.lower() for rec in recs)


def test_rule_engine_high_pm_returns_filter_recommendation() -> None:
    """High PM should trigger bag filter recommendation."""
    engine = RuleEngine()
    recs = engine.apply_rules(
        {
            "so2_score": 1.0,
            "pm25_score": 7.5,
            "pm10_score": 6.5,
            "no2_score": 1.0,
            "co_score": 1.0,
            "o3_score": 1.0,
            "composite_score": 5.0,
            "risk_level": "Medium",
            "industry_type": "cement",
        }
    )

    assert any("bag filter" in rec.action.lower() or "fabric filter" in rec.action.lower() for rec in recs)


def test_rule_engine_low_risk_returns_maintenance_recommendation() -> None:
    """Low risk should include maintain-current-systems recommendation."""
    engine = RuleEngine()
    recs = engine.apply_rules(
        {
            "so2_score": 1.0,
            "pm25_score": 1.0,
            "pm10_score": 1.0,
            "no2_score": 1.0,
            "co_score": 1.0,
            "o3_score": 1.0,
            "composite_score": 1.5,
            "risk_level": "Low",
            "industry_type": "industrial",
        }
    )

    assert any("maintain current emission control systems" in rec.action.lower() for rec in recs)


def test_industry_specific_rules_steel() -> None:
    """Steel industry should receive steel-specific controls recommendation."""
    engine = RuleEngine()
    recs = engine.apply_rules(
        {
            "so2_score": 2.0,
            "pm25_score": 2.0,
            "pm10_score": 2.0,
            "no2_score": 2.0,
            "co_score": 2.0,
            "o3_score": 2.0,
            "composite_score": 2.5,
            "risk_level": "Low",
            "industry_type": "steel",
        }
    )

    assert any("slag handling" in rec.action.lower() for rec in recs)


def test_hybrid_engine_generates_report(
    monkeypatch: pytest.MonkeyPatch,
    sample_factories_df: pd.DataFrame,
    sample_pollution_df: pd.DataFrame,
) -> None:
    """Hybrid engine should generate at least one report with recommendations."""
    monkeypatch.setattr(
        "src.recommendations.ml_recommender.MLRecommender.predict_recommendations",
        lambda self, _: ["Monitoring"],
    )

    engine = HybridRecommendationEngine(
        {
            "recommendations": {
                "rule_weight": 0.7,
                "ml_weight": 0.3,
                "confidence_threshold": 0.4,
                "max_station_distance_km": 100,
            }
        }
    )

    reports = engine.generate_all(sample_factories_df, sample_pollution_df)

    assert len(reports) == 1
    assert reports[0].factory_id == "F001"
    assert reports[0].recommendations


def test_hybrid_engine_empty_pollution_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
    sample_factories_df: pd.DataFrame,
) -> None:
    """Hybrid engine should handle empty pollution dataframe without exceptions."""
    monkeypatch.setattr(
        "src.recommendations.ml_recommender.MLRecommender.predict_recommendations",
        lambda self, _: [],
    )

    engine = HybridRecommendationEngine(
        {
            "recommendations": {
                "rule_weight": 0.7,
                "ml_weight": 0.3,
                "confidence_threshold": 0.4,
                "max_station_distance_km": 100,
            }
        }
    )

    reports = engine.generate_all(sample_factories_df, pd.DataFrame())

    assert len(reports) == 1
    assert reports[0].factory_name == "Steel Plant A"


def test_formatter_csv_export_creates_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Formatter should call CSV export path in write mode and target requested file."""
    calls: list[tuple[Path, str]] = []

    def _fake_to_csv(self, path, index=False, mode="w", **kwargs):  # type: ignore[no-untyped-def]
        calls.append((Path(path), mode))

    monkeypatch.setattr(pd.DataFrame, "to_csv", _fake_to_csv)

    report = FactoryReport(
        factory_id="F001",
        factory_name="Steel Plant A",
        industry_type="steel",
        city="Pune",
        risk_level="High",
        composite_score=7.5,
        dominant_pollutant="so2",
        pollution_scores={
            "pm25_score": 6.0,
            "pm10_score": 6.5,
            "so2_score": 7.5,
            "no2_score": 5.0,
            "co_score": 2.0,
            "o3_score": 3.0,
        },
        recommendations=[
            Recommendation(
                category="Emission Control",
                priority="Immediate",
                action="Install wet scrubber",
                pollutant="so2",
                estimated_reduction="60-80% SO2 reduction",
                cost_category="High",
                timeline="3-6 months installation",
            )
        ],
        summary="Factory is high risk with high SO2.",
        generated_at=datetime.now(timezone.utc),
    )

    formatter = RecommendationFormatter(
        {
            "recommendations": {"output_csv": "data/output/recommendations.csv"},
            "paths": {"recommendations": "data/output/recommendations.csv"},
        }
    )
    output = tmp_path / "recommendations.csv"
    formatter.export_csv([report], output)

    assert calls, "Expected DataFrame.to_csv to be called"
    assert calls[0][0] == output
    assert calls[0][1] == "w"
