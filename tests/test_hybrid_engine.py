"""Comprehensive tests for HybridRecommendationEngine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.recommendations.engine import HybridRecommendationEngine


@pytest.fixture

def engine_cfg(mock_config: dict) -> dict:  # type: ignore[no-untyped-def]
    return mock_config


def test_hybrid_engine_generates_factory_report(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    engine = HybridRecommendationEngine(engine_cfg)
    reports = engine.generate_all(sample_factories_df.head(1), sample_pollution_df)
    assert len(reports) == 1


def test_hybrid_engine_report_has_all_required_fields(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    report = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), sample_pollution_df)[0]
    for field in [
        "factory_id",
        "factory_name",
        "industry_type",
        "city",
        "risk_level",
        "composite_score",
        "dominant_pollutant",
        "pollution_scores",
        "recommendations",
        "summary",
        "generated_at",
    ]:
        assert hasattr(report, field)


def test_hybrid_engine_report_risk_level_is_valid_enum(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    report = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), sample_pollution_df)[0]
    assert report.risk_level in {"Low", "Medium", "High", "Critical", "Unknown"}


def test_hybrid_engine_report_recommendations_is_list(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    report = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), sample_pollution_df)[0]
    assert isinstance(report.recommendations, list)


def test_hybrid_engine_report_summary_is_non_empty_string(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    report = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), sample_pollution_df)[0]
    assert isinstance(report.summary, str)
    assert report.summary.strip()


def test_hybrid_engine_generate_all_returns_sorted_by_score_desc(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    reports = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df, sample_pollution_df)
    scores = [r.composite_score for r in reports]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_engine_empty_factories_returns_empty_list(engine_cfg: dict, sample_pollution_df: pd.DataFrame) -> None:
    reports = HybridRecommendationEngine(engine_cfg).generate_all(pd.DataFrame(), sample_pollution_df)
    assert reports == []


def test_hybrid_engine_empty_pollution_does_not_crash(engine_cfg: dict, sample_factories_df: pd.DataFrame) -> None:
    reports = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), pd.DataFrame())
    assert len(reports) == 1


def test_hybrid_engine_nan_pollution_does_not_leak_into_scores(engine_cfg: dict, sample_factories_df: pd.DataFrame, sample_pollution_df: pd.DataFrame) -> None:
    polluted = sample_pollution_df.copy()
    polluted[["pm25", "pm10", "co", "no2", "so2", "o3"]] = np.nan
    report = HybridRecommendationEngine(engine_cfg).generate_all(sample_factories_df.head(1), polluted)[0]
    assert all(isinstance(v, float) for v in report.pollution_scores.values())
