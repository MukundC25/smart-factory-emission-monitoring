"""Formatter and exporter for recommendation reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import get_project_root, initialize_environment
from src.recommendations.engine import FactoryReport

LOGGER = logging.getLogger(__name__)


class RecommendationFormatter:
    """Formats and exports recommendation reports to CSV and JSON."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RecommendationFormatter with runtime config.

        Args:
            config: Optional runtime configuration.
        """
        self.config = config or initialize_environment()

    @staticmethod
    def _join_actions_by_priority(report: FactoryReport, priority: str) -> str:
        """Join recommendation actions for a given priority.

        Args:
            report: Factory recommendation report.
            priority: Priority bucket to filter by.

        Returns:
            str: Pipe-separated action strings.
        """
        actions = [rec.action for rec in report.recommendations if rec.priority == priority]
        return " | ".join(actions)

    @staticmethod
    def _join_monitoring_actions(report: FactoryReport) -> str:
        """Join monitoring recommendations as a dedicated bucket.

        Args:
            report: Factory recommendation report.

        Returns:
            str: Pipe-separated monitoring actions.
        """
        actions = [rec.action for rec in report.recommendations if rec.category.lower() == "monitoring"]
        return " | ".join(actions)

    def to_csv_row(self, report: FactoryReport) -> Dict[str, Any]:
        """Flatten FactoryReport into a single CSV-compatible row.

        Args:
            report: Factory recommendation report.

        Returns:
            Dict[str, Any]: Flattened row dictionary.
        """
        return {
            "factory_id": report.factory_id,
            "factory_name": report.factory_name,
            "industry_type": report.industry_type,
            "city": report.city,
            "risk_level": report.risk_level,
            "composite_score": round(report.composite_score, 4),
            "dominant_pollutant": report.dominant_pollutant,
            "immediate_actions": self._join_actions_by_priority(report, "Immediate"),
            "short_term_actions": self._join_actions_by_priority(report, "Short-term"),
            "long_term_actions": self._join_actions_by_priority(report, "Long-term"),
            "monitoring_actions": self._join_monitoring_actions(report),
            "summary": report.summary,
            "generated_at": report.generated_at.isoformat(),
        }

    def to_json(self, report: FactoryReport) -> Dict[str, Any]:
        """Convert FactoryReport into nested JSON-ready dictionary.

        Args:
            report: Factory recommendation report.

        Returns:
            Dict[str, Any]: Nested JSON structure.
        """
        return {
            "factory_id": report.factory_id,
            "factory_name": report.factory_name,
            "industry_type": report.industry_type,
            "city": report.city,
            "risk_level": report.risk_level,
            "composite_score": report.composite_score,
            "dominant_pollutant": report.dominant_pollutant,
            "pollution_scores": report.pollution_scores,
            "summary": report.summary,
            "generated_at": report.generated_at.isoformat(),
            "recommendations": [
                {
                    "category": rec.category,
                    "priority": rec.priority,
                    "action": rec.action,
                    "pollutant": rec.pollutant,
                    "estimated_reduction": rec.estimated_reduction,
                    "cost_category": rec.cost_category,
                    "timeline": rec.timeline,
                }
                for rec in report.recommendations
            ],
        }

    def export_csv(self, reports: List[FactoryReport], path: Optional[Path] = None) -> None:
        """Export reports to CSV, overwriting any existing file.

        Args:
            reports: List of factory reports.
            path: Optional output path override.
        """
        rec_cfg = self.config.get("recommendations", {})
        root = get_project_root()
        default_rel = rec_cfg.get("output_csv") or self.config.get("paths", {}).get("recommendations")
        if not default_rel and path is None:
            raise ValueError("CSV output path is not configured")

        output_path = path or (root / str(default_rel))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        columns = [
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
        ]

        if not reports:
            pd.DataFrame(columns=columns).to_csv(output_path, index=False, mode="w")
            LOGGER.info("Exported empty recommendations CSV to %s", output_path)
            return

        rows = [self.to_csv_row(report) for report in reports]
        pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False, mode="w")
        LOGGER.info("Exported %d recommendation rows to %s", len(rows), output_path)

    def export_json(self, reports: List[FactoryReport], path: Optional[Path] = None) -> None:
        """Export reports to JSON, overwriting any existing file.

        Args:
            reports: List of factory reports.
            path: Optional output path override.
        """
        rec_cfg = self.config.get("recommendations", {})
        root = get_project_root()
        default_rel = rec_cfg.get("output_json")
        if not default_rel:
            csv_rel = self.config.get("paths", {}).get("recommendations")
            if csv_rel:
                default_rel = str(Path(str(csv_rel)).with_suffix(".json"))
        if not default_rel and path is None:
            raise ValueError("JSON output path is not configured")

        output_path = path or (root / str(default_rel))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "count": len(reports),
            "reports": [self.to_json(report) for report in reports],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        LOGGER.info("Exported %d recommendation reports to %s", len(reports), output_path)
