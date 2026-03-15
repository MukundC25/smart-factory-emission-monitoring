"""Recommendations API router."""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_data_loader
from backend.schemas.recommendations import (
    FactoryRecommendationReport,
    FactoryRecommendationSummary,
    RecommendationsGenerateResponse,
    RecommendationsListResponse,
    RecommendationsStatsResponse,
    RiskLevelCounts,
    PollutantCount,
)
from backend.utils.data_loader import DataLoader
from src.common import get_project_root, initialize_environment
from src.recommendations.engine import HybridRecommendationEngine
from src.recommendations.formatter import RecommendationFormatter

router = APIRouter(tags=["Recommendations"])
logger = logging.getLogger(__name__)


def _resolve_output_json_path(config: dict) -> str:
    """Resolve recommendation JSON output path from config entries.

    Args:
        config: Runtime configuration.

    Returns:
        str: Relative JSON output path.
    """
    rec_cfg = config.get("recommendations", {})
    if rec_cfg.get("output_json"):
        return str(rec_cfg["output_json"])
    csv_rel = config.get("paths", {}).get("recommendations")
    if not csv_rel:
        raise ValueError("Missing recommendations output path in config.yaml")
    return str(Path(str(csv_rel)).with_suffix(".json"))


def _to_summary(item: dict) -> FactoryRecommendationSummary:
    """Convert raw report dictionary to summary schema."""
    return FactoryRecommendationSummary(
        factory_id=str(item.get("factory_id", "")),
        factory_name=str(item.get("factory_name", "")),
        industry_type=str(item.get("industry_type", "")),
        city=str(item.get("city", "")),
        risk_level=str(item.get("risk_level", "Low")),
        composite_score=float(item.get("composite_score", 0.0) or 0.0),
        dominant_pollutant=str(item.get("dominant_pollutant", "")),
        summary=str(item.get("summary", "")),
        generated_at=item.get("generated_at"),
    )


@router.get(
    "/recommendations",
    response_model=RecommendationsListResponse,
    summary="List recommendation summaries with optional filters",
)
def list_recommendations(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=200, description="Results per page"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    industry_type: Optional[str] = Query(None, description="Filter by industry type substring"),
    city: Optional[str] = Query(None, description="Filter by city substring"),
    loader: DataLoader = Depends(get_data_loader),
) -> RecommendationsListResponse:
    """Return paginated recommendation summaries from recommendations JSON output."""
    reports = loader.load_recommendation_reports()

    if risk_level:
        reports = [r for r in reports if str(r.get("risk_level", "")).lower() == risk_level.lower()]
    if industry_type:
        reports = [
            r
            for r in reports
            if industry_type.lower() in str(r.get("industry_type", "")).lower()
        ]
    if city:
        reports = [r for r in reports if city.lower() in str(r.get("city", "")).lower()]

    reports = sorted(reports, key=lambda r: float(r.get("composite_score", 0.0) or 0.0), reverse=True)
    total = len(reports)
    start = (page - 1) * page_size
    end = start + page_size
    page_rows = reports[start:end]

    return RecommendationsListResponse(
        total=total,
        page=page,
        page_size=page_size,
        data=[_to_summary(item) for item in page_rows],
    )


@router.get(
    "/recommendations/stats",
    response_model=RecommendationsStatsResponse,
    summary="Get recommendation aggregate statistics",
)
def recommendations_stats(loader: DataLoader = Depends(get_data_loader)) -> RecommendationsStatsResponse:
    """Return total count, risk-level counts, top dominant pollutants, and last generated timestamp."""
    reports = loader.load_recommendation_reports()

    risk_counter = Counter(str(item.get("risk_level", "Low")) for item in reports)
    pollutant_counter = Counter(str(item.get("dominant_pollutant", "")) for item in reports if item.get("dominant_pollutant"))

    top_pollutants = [
        PollutantCount(pollutant=pollutant, count=count)
        for pollutant, count in pollutant_counter.most_common(5)
    ]

    parsed_generated_values = []
    for item in reports:
        value = item.get("generated_at")
        if not value:
            continue
        if isinstance(value, datetime):
            parsed_generated_values.append(value)
        elif isinstance(value, str):
            try:
                parsed_generated_values.append(datetime.fromisoformat(value))
            except (ValueError, TypeError):
                continue
    last_generated = max(parsed_generated_values) if parsed_generated_values else None

    return RecommendationsStatsResponse(
        total_factories=len(reports),
        by_risk_level=RiskLevelCounts(
            Critical=int(risk_counter.get("Critical", 0)),
            High=int(risk_counter.get("High", 0)),
            Medium=int(risk_counter.get("Medium", 0)),
            Low=int(risk_counter.get("Low", 0)),
        ),
        top_pollutants=top_pollutants,
        last_generated=last_generated,
    )


@router.get(
    "/recommendations/{factory_id}",
    response_model=FactoryRecommendationReport,
    summary="Get full recommendation report by factory ID",
)
def get_recommendation_by_factory_id(
    factory_id: str,
    loader: DataLoader = Depends(get_data_loader),
) -> FactoryRecommendationReport:
    """Return full recommendation report for a specific factory ID."""
    reports = loader.load_recommendation_reports()
    matched = next((item for item in reports if str(item.get("factory_id")) == factory_id), None)

    if matched is None:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation report not found for factory_id '{factory_id}'",
        )

    return FactoryRecommendationReport(
        factory_id=str(matched.get("factory_id", "")),
        factory_name=str(matched.get("factory_name", "")),
        industry_type=str(matched.get("industry_type", "")),
        city=str(matched.get("city", "")),
        risk_level=str(matched.get("risk_level", "Low")),
        composite_score=float(matched.get("composite_score", 0.0) or 0.0),
        dominant_pollutant=str(matched.get("dominant_pollutant", "")),
        pollution_scores={
            str(k): float(v or 0.0)
            for k, v in dict(matched.get("pollution_scores", {})).items()
        },
        recommendations=matched.get("recommendations", []),
        summary=str(matched.get("summary", "")),
        generated_at=matched.get("generated_at"),
    )


@router.post(
    "/recommendations/generate",
    response_model=RecommendationsGenerateResponse,
    summary="Generate recommendations synchronously",
)
def generate_recommendations(loader: DataLoader = Depends(get_data_loader)) -> RecommendationsGenerateResponse:
    """Trigger synchronous recommendation regeneration and return processed count."""
    config = initialize_environment()
    root = get_project_root()

    factories_df = pd.read_csv(root / config["paths"]["factories_raw"])
    pollution_df = pd.read_csv(root / config["paths"]["pollution_processed"])

    if factories_df.empty or pollution_df.empty:
        logger.warning(
            "Skipping recommendation generation due to empty inputs (factories=%d, pollution=%d)",
            len(factories_df),
            len(pollution_df),
        )
        output_path = str(root / _resolve_output_json_path(config))
        return RecommendationsGenerateResponse(
            status="success",
            factories_processed=0,
            output_path=output_path,
        )

    engine = HybridRecommendationEngine(config)
    formatter = RecommendationFormatter(config)
    reports = engine.generate_all(factories_df=factories_df, pollution_df=pollution_df)

    formatter.export_csv(reports)
    formatter.export_json(reports)

    logger.info(
        "Reports exported to %s (CSV) and %s (JSON) — legacy recommendations.csv untouched",
        config.get("recommendations", {}).get("output_csv"),
        config.get("recommendations", {}).get("output_json"),
    )

    loader.refresh()

    output_path = str(root / _resolve_output_json_path(config))
    return RecommendationsGenerateResponse(
        status="success",
        factories_processed=len(reports),
        output_path=output_path,
    )
