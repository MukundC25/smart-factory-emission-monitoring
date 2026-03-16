"""Comprehensive tests for RecommendationFormatter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.recommendations.engine import FactoryReport
from src.recommendations.formatter import RecommendationFormatter
from src.recommendations.rule_engine import Recommendation


def _report() -> FactoryReport:
    return FactoryReport(
        factory_id="FAC001",
        factory_name="Pune Steel Works",
        industry_type="steel",
        city="Pune",
        risk_level="High",
        composite_score=7.2,
        dominant_pollutant="so2",
        pollution_scores={
            "pm25_score": 6.5,
            "pm10_score": 6.8,
            "so2_score": 8.1,
            "no2_score": 5.2,
            "co_score": 2.0,
            "o3_score": 3.1,
        },
        recommendations=[
            Recommendation(
                category="Emission Control",
                priority="Immediate",
                action="Install wet scrubber",
                pollutant="so2",
                estimated_reduction="60-80%",
                cost_category="High",
                timeline="3-6 months",
            )
        ],
        summary="High SO2 risk.",
        generated_at=datetime.now(timezone.utc),
    )


def _cfg(tmp_path: Path) -> dict:
    return {
        "paths": {"recommendations": str((tmp_path / "recommendations.csv").as_posix())},
        "recommendations": {
            "output_csv": str((tmp_path / "recommendations.csv").as_posix()),
            "output_json": str((tmp_path / "recommendations.json").as_posix()),
        },
    }


def test_to_csv_row_returns_flat_dict(tmp_path: Path) -> None:
    row = RecommendationFormatter(_cfg(tmp_path)).to_csv_row(_report())
    assert isinstance(row, dict)
    assert "recommendations" not in row


def test_to_csv_row_has_all_required_columns(tmp_path: Path) -> None:
    row = RecommendationFormatter(_cfg(tmp_path)).to_csv_row(_report())
    required = {
        "factory_id",
        "factory_name",
        "industry_type",
        "city",
        "risk_level",
        "composite_score",
        "dominant_pollutant",
        "immediate_actions",
        "short_term_actions",
        "long_term_actions",
        "monitoring_actions",
        "summary",
        "generated_at",
    }
    assert required.issubset(row.keys())


def test_to_json_returns_nested_dict(tmp_path: Path) -> None:
    payload = RecommendationFormatter(_cfg(tmp_path)).to_json(_report())
    assert isinstance(payload, dict)
    assert isinstance(payload["recommendations"], list)


def test_export_csv_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    RecommendationFormatter(_cfg(tmp_path)).export_csv([_report()], path)
    assert path.exists()


def test_export_csv_has_correct_columns(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    RecommendationFormatter(_cfg(tmp_path)).export_csv([_report()], path)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "factory_id" in header
    assert "immediate_actions" in header


def test_export_json_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    RecommendationFormatter(_cfg(tmp_path)).export_json([_report()], path)
    assert path.exists()


def test_export_json_is_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "out.json"
    RecommendationFormatter(_cfg(tmp_path)).export_json([_report()], path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "reports" in data


def test_export_csv_empty_reports_creates_empty_file_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    RecommendationFormatter(_cfg(tmp_path)).export_csv([], path)
    assert path.exists()


def test_export_json_empty_reports_creates_empty_list_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    RecommendationFormatter(_cfg(tmp_path)).export_json([], path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["reports"] == []
