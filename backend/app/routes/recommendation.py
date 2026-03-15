"""Recommendation endpoints based on pollution impact predictions."""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from ..schemas import RecommendationResponse
from ..services.ml_service import get_ml_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recommendation", tags=["Recommendations"])

# Cache for recommendations data
_recommendations_cache: Optional[pd.DataFrame] = None


def _load_recommendations_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load pre-computed recommendations from CSV.

    Args:
        csv_path: Optional path to recommendations CSV.

    Returns:
        pd.DataFrame: Recommendations dataset.
    """
    global _recommendations_cache

    if _recommendations_cache is not None:
        return _recommendations_cache

    if csv_path is None:
        from pathlib import Path as PathlibPath

        csv_path = (
            PathlibPath(__file__).parent.parent.parent.parent
            / "data"
            / "output"
            / "recommendations.csv"
        )

    if not csv_path.exists():
        logger.warning("Recommendations CSV not found at %s", csv_path)
        return pd.DataFrame()

    _recommendations_cache = pd.read_csv(csv_path)
    logger.info("Loaded %d recommendations from %s", len(_recommendations_cache), csv_path)
    return _recommendations_cache


def _generate_recommendation(factory_id: str, factory_name: str, industry_type: str, score: float) -> RecommendationResponse:
    """Generate recommendation for a factory based on pollution impact score.

    Args:
        factory_id: Factory identifier.
        factory_name: Factory name.
        industry_type: Type of industry.
        score: Pollution impact score.

    Returns:
        RecommendationResponse: Recommendation with control measures.
    """
    service = get_ml_service()
    risk_level = service.get_risk_level(score)
    recommendation_text = service.get_recommendation(risk_level, industry_type)
    measures = service.get_control_measures(risk_level, industry_type)

    return RecommendationResponse(
        factory_id=factory_id,
        factory_name=factory_name,
        risk_level=risk_level,
        pollution_impact_score=score,
        recommendation=recommendation_text,
        measures=measures,
    )


@router.get("/{factory_id}", response_model=RecommendationResponse)
def get_recommendation(factory_id: str):
    """Get recommendation for a specific factory.

    Args:
        factory_id: Factory identifier.

    Returns:
        RecommendationResponse: Risk assessment and control measures.

    Raises:
        HTTPException: If factory or recommendation not found.
    """
    try:
        df = _load_recommendations_data()

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Recommendations not available. Run pipeline first.",
            )

        rec_row = df[df["factory_id"] == factory_id]

        if rec_row.empty:
            raise HTTPException(status_code=404, detail=f"No recommendation found for factory {factory_id}")

        row = rec_row.iloc[0]
        return _generate_recommendation(
            factory_id=str(row.get("factory_id", "")),
            factory_name=str(row.get("factory_name", "")),
            industry_type=str(row.get("industry_type", "")),
            score=float(row.get("pollution_impact_score", 0)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching recommendation for %s: %s", factory_id, e)
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendation")


@router.get("/", response_model=List[RecommendationResponse])
def get_all_recommendations(
    risk_level: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get recommendations with optional filtering.

    Args:
        risk_level: Optional filter by risk level (Low, Medium, High).
        city: Optional filter by city.
        limit: Maximum number of results.

    Returns:
        List[RecommendationResponse]: List of recommendations.

    Raises:
        HTTPException: If no data available.
    """
    try:
        df = _load_recommendations_data()

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Recommendations not available. Run pipeline first.",
            )

        if risk_level:
            df = df[df["risk_level"].str.lower() == risk_level.lower()]

        if city:
            df = df[df["city"].str.lower() == city.lower()]

        df = df.head(limit)

        recommendations = []
        for _, row in df.iterrows():
            rec = _generate_recommendation(
                factory_id=str(row.get("factory_id", "")),
                factory_name=str(row.get("factory_name", "")),
                industry_type=str(row.get("industry_type", "")),
                score=float(row.get("pollution_impact_score", 0)),
            )
            recommendations.append(rec)

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching recommendations: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve recommendations")


@router.get("/high-risk/list", response_model=List[RecommendationResponse])
def get_high_risk_factories():
    """Get all high-risk factories requiring urgent intervention.

    Returns:
        List[RecommendationResponse]: High-risk factory recommendations.

    Raises:
        HTTPException: If no data available.
    """
    try:
        df = _load_recommendations_data()

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Recommendations not available. Run pipeline first.",
            )

        high_risk = df[df["risk_level"].str.lower() == "high"]

        if high_risk.empty:
            return []

        recommendations = []
        for _, row in high_risk.iterrows():
            rec = _generate_recommendation(
                factory_id=str(row.get("factory_id", "")),
                factory_name=str(row.get("factory_name", "")),
                industry_type=str(row.get("industry_type", "")),
                score=float(row.get("pollution_impact_score", 0)),
            )
            recommendations.append(rec)

        logger.info("Retrieved %d high-risk factories", len(recommendations))
        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching high-risk factories: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve high-risk factories")


@router.get("/statistics/summary", response_model=dict)
def get_recommendation_statistics():
    """Get overall statistics about factory risk distribution.

    Returns:
        dict: Risk distribution and summary statistics.

    Raises:
        HTTPException: If no data available.
    """
    try:
        df = _load_recommendations_data()

        if df.empty:
            raise HTTPException(
                status_code=503,
                detail="Recommendations not available. Run pipeline first.",
            )

        risk_counts = df["risk_level"].value_counts().to_dict() if "risk_level" in df.columns else {}
        score_col = "pollution_impact_score"
        score_stats = (
            {
                "mean": float(df[score_col].mean()),
                "std": float(df[score_col].std()),
                "min": float(df[score_col].min()),
                "max": float(df[score_col].max()),
            }
            if score_col in df.columns
            else {}
        )

        city_risks = {}
        if "city" in df.columns and "risk_level" in df.columns:
            city_risks = df.groupby("city")["risk_level"].value_counts().to_dict()

        return {
            "total_factories": len(df),
            "risk_distribution": risk_counts,
            "impact_score_statistics": score_stats,
            "cities_analyzed": df["city"].nunique() if "city" in df.columns else 0,
            "city_risk_breakdown": str(city_risks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error computing recommendation statistics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to compute statistics")
