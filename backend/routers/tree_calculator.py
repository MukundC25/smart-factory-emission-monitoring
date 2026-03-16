"""Tree Planting Calculator API router.

Endpoints:
  GET  /factories/{factory_id}/tree-recommendation        — single factory
  POST /factories/tree-recommendation/bulk                — up to 50 factories
  GET  /tree-calculator/constants                         — methodology reference
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.dependencies import get_data_loader
from backend.schemas.tree_calculator import (
    PollutantBreakdown,
    PollutantReadings,
    TreeCalculatorBulkRequest,
    TreeCalculatorBulkResponse,
    TreeRecommendationResponse,
    TreesNeeded,
)
from backend.utils.data_loader import DataLoader
from src.recommendations.openaq_client import OpenAQClient
from src.recommendations.tree_calculator import (
    AQI_REDUCTION_PER_1000_TREES,
    AQI_GOOD,
    AQI_MODERATE,
    AQI_POOR,
    AQI_SATISFACTORY,
    AQI_VERY_POOR,
    MATURITY_YEARS,
    MINIMUM_BUFFER,
    OPTIMAL_BUFFER,
    RECOMMENDED_BUFFER,
    TREE_CANOPY_RADIUS_M,
    TREE_CO2_ABSORPTION_KG_YEAR,
    TREE_CO2_ABSORPTION_TONS_YEAR,
    TREE_COVERAGE_AREA_M2,
    TREE_PM10_ABSORPTION_UG_M3,
    TREE_PM25_ABSORPTION_UG_M3,
    TREES_PER_HECTARE,
    TreePlantingCalculator,
)

router = APIRouter(tags=["Tree Calculator"])
logger = logging.getLogger(__name__)

# Approximate mapping from 0–10 pollution score to representative concentration values.
# These are intentionally coarse and only used when we have score data instead of
# proper μg/m³ (or mg/m³ for CO) readings. They should be tuned if methodology changes.
POLLUTANT_SCORE_MAX_CONCENTRATIONS = {
    # μg/m³
    "pm25": 150.0,   # representative of "severe" PM2.5
    "pm10": 300.0,   # representative of "severe" PM10
    "no2": 400.0,
    "so2": 400.0,
    "o3": 300.0,
    # mg/m³ (tree calculator expects mg/m³ for CO)
    "co": 15.0,
}


def _score_to_concentration(score: Optional[float], pollutant: str) -> Optional[float]:
    """Convert a 0–10 pollution score to an approximate concentration for the calculator.

    Returns None if the score is missing/invalid or the pollutant is unknown.
    """
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    # Clamp scores to the expected range.
    if value < 0.0:
        value = 0.0
    elif value > 10.0:
        value = 10.0

    max_conc = POLLUTANT_SCORE_MAX_CONCENTRATIONS.get(pollutant)
    if max_conc is None:
        return None

    return (value / 10.0) * max_conc

_calculator = TreePlantingCalculator()
_openaq_client = OpenAQClient()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_response(
    factory_id: str,
    factory_name: str,
    city: str,
    industry_type: str,
    rec,  # TreeRecommendation dataclass
    aqi_data: dict,
    data_source: str,
) -> TreeRecommendationResponse:
    """Assemble a :class:`TreeRecommendationResponse` from a TreeRecommendation."""
    return TreeRecommendationResponse(
        factory_id=factory_id,
        factory_name=factory_name,
        city=city,
        industry_type=industry_type,
        current_aqi=rec.current_aqi,
        current_pollution_score=rec.current_pollution_score,
        dominant_pollutant=rec.dominant_pollutant,
        current_readings=PollutantReadings(
            pm25=aqi_data.get("pm25"),
            pm10=aqi_data.get("pm10"),
            no2=aqi_data.get("no2"),
            so2=aqi_data.get("so2"),
            co=aqi_data.get("co"),
            o3=aqi_data.get("o3"),
            aqi_index=aqi_data.get("aqi"),
            source=aqi_data.get("source", "cached"),
            timestamp=aqi_data.get("timestamp"),
        ),
        target_aqi=rec.target_aqi,
        trees_needed=TreesNeeded(
            minimum=rec.trees_needed["minimum"],
            recommended=rec.trees_needed["recommended"],
            optimal=rec.trees_needed["optimal"],
        ),
        impact_radius_km=rec.impact_radius_km,
        planting_area_hectares=rec.planting_area_hectares,
        annual_co2_offset_tons=rec.annual_co2_offset_tons,
        estimated_aqi_reduction=rec.estimated_aqi_reduction,
        timeline_years=rec.timeline_years,
        pollutant_breakdown=PollutantBreakdown(
            pm25_trees=rec.pollutant_breakdown.get("pm25_trees"),
            pm10_trees=rec.pollutant_breakdown.get("pm10_trees"),
            no2_trees=rec.pollutant_breakdown.get("no2_trees"),
            so2_trees=rec.pollutant_breakdown.get("so2_trees"),
            co_trees=rec.pollutant_breakdown.get("co_trees"),
        ),
        feasibility=rec.feasibility,
        notes=rec.notes,
        data_source=data_source,
        calculated_at=datetime.now(tz=timezone.utc).isoformat(),
    )


async def _fetch_aqi_live(city: str, lat: float, lon: float) -> dict:
    """Run the synchronous OpenAQ call in a thread executor with 8 s timeout."""
    loop = asyncio.get_running_loop()
    try:
        aqi_data = await asyncio.wait_for(
            loop.run_in_executor(None, _openaq_client.get_city_aqi, city, lat, lon),
            timeout=8.0,
        )
        return aqi_data
    except asyncio.TimeoutError:
        logger.warning("OpenAQ call for %s timed out — falling back to cached data", city)
        return {}
    except Exception as exc:
        logger.warning("OpenAQ call for %s failed (%s) — falling back to cached data", city, exc)
        return {}


def _factory_not_found(factory_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=f"Factory '{factory_id}' not found. Use GET /factories to list available factories.",
    )


def _resolve_factory_data(factory_id: str, loader: DataLoader):
    """Look up factory row; raise 404 if absent. Returns (factory_row, reports, report | None)."""
    factories_df = loader.load_factories()
    mask = factories_df["factory_id"] == factory_id
    if not mask.any():
        raise _factory_not_found(factory_id)
    factory_row = factories_df[mask].iloc[0]

    reports = loader.load_recommendation_reports()
    report = next((r for r in reports if str(r.get("factory_id", "")) == factory_id), None)

    return factory_row, reports, report


def _extract_report_data(report: Optional[dict]) -> tuple[float, str, dict]:
    """Extract composite_score, dominant_pollutant, pollution_readings from a report dict.

    For cached data, pollution_scores are 0–10 scores and are converted here into
    approximate concentrations for use by the TreePlantingCalculator.
    """
    if report is None:
        return 5.0, "pm25", {}

    composite_score = float(report.get("composite_score", 5.0) or 5.0)
    dominant_pollutant = str(report.get("dominant_pollutant", "pm25") or "pm25")

    scores = report.get("pollution_scores", {}) or {}
    # pollution_scores are *score* values (0–10), not raw μg/m³ — map to approximate readings
    pollution_readings = {
        "pm25": _score_to_concentration(scores.get("pm25_score"), "pm25"),
        "pm10": _score_to_concentration(scores.get("pm10_score"), "pm10"),
        "no2": _score_to_concentration(scores.get("no2_score"), "no2"),
        "so2": _score_to_concentration(scores.get("so2_score"), "so2"),
        "co": _score_to_concentration(scores.get("co_score"), "co"),
        "o3": _score_to_concentration(scores.get("o3_score"), "o3"),
    }
    return composite_score, dominant_pollutant, pollution_readings


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/factories/{factory_id}/tree-recommendation",
    response_model=TreeRecommendationResponse,
    summary="Get tree planting recommendation to offset factory pollution",
)
async def get_tree_recommendation(
    factory_id: str,
    use_live_aqi: bool = Query(
        True,
        description="Fetch live AQI from OpenAQ API. Set False to use cached/score data.",
    ),
    target_aqi: Optional[float] = Query(
        None,
        description="Override target AQI. Defaults to the next better CPCB AQI band.",
        gt=0,
        le=500,
    ),
    loader: DataLoader = Depends(get_data_loader),
) -> TreeRecommendationResponse:
    """Return tree planting requirements to offset pollution from *factory_id*.

    When ``use_live_aqi=True`` (default) the current AQI is fetched from the
    OpenAQ v2 API.  If OpenAQ is unreachable the endpoint falls back to cached
    pollution-score data and never returns a 500.
    """
    factory_row, _reports, report = _resolve_factory_data(factory_id, loader)

    city = str(factory_row.get("city", ""))
    lat = float(factory_row.get("latitude", 0.0))
    lon = float(factory_row.get("longitude", 0.0))
    factory_name = str(factory_row.get("factory_name", factory_id))
    industry_type = str(factory_row.get("industry_type", ""))

    composite_score, dominant_pollutant, pollution_readings = _extract_report_data(report)

    # AQI data: live or fallback
    aqi_data: dict = {}
    if use_live_aqi:
        aqi_data = await _fetch_aqi_live(city, lat, lon)

    # Merge live readings on top of cached score-based readings
    if aqi_data:
        for pol in ("pm25", "pm10", "no2", "so2", "co", "o3"):
            if aqi_data.get(pol) is not None:
                pollution_readings[pol] = aqi_data[pol]

    current_aqi_val: float
    if aqi_data.get("aqi"):
        current_aqi_val = float(aqi_data["aqi"])
    else:
        # Rough AQI estimate from composite score (score 0-10 → AQI ~ score * 40)
        current_aqi_val = max(composite_score * 40.0, 50.0)

    # If no live AQI data is available or used, populate aqi_data from cached readings
    if not aqi_data:
        aqi_data = {
            "pm25": pollution_readings.get("pm25"),
            "pm10": pollution_readings.get("pm10"),
            "no2": pollution_readings.get("no2"),
            "so2": pollution_readings.get("so2"),
            "co": pollution_readings.get("co"),
            "o3": pollution_readings.get("o3"),
            "aqi": current_aqi_val,
            "source": "cached",
        }

    data_source = aqi_data.get("source", "cached") if aqi_data else "cached"

    rec = _calculator.calculate_trees_needed(
        factory_id=factory_id,
        city=city,
        current_aqi=current_aqi_val,
        pollution_score=composite_score,
        dominant_pollutant=dominant_pollutant,
        pollution_readings=pollution_readings,
        target_aqi_override=target_aqi,
    )

    return _build_response(factory_id, factory_name, city, industry_type, rec, aqi_data, data_source)


@router.post(
    "/factories/tree-recommendation/bulk",
    response_model=TreeCalculatorBulkResponse,
    summary="Get tree recommendations for multiple factories",
)
async def get_bulk_tree_recommendations(
    request: TreeCalculatorBulkRequest,
    use_live_aqi: bool = Query(
        False,
        description="Fetch live AQI per factory. Defaults to False for bulk to avoid rate limits.",
    ),
    loader: DataLoader = Depends(get_data_loader),
) -> TreeCalculatorBulkResponse:
    """Return tree planting recommendations for up to 50 factories.

    Failures for individual factories are collected into ``errors`` and do not
    abort the whole request.
    """
    results = []
    errors = []

    factories_df = loader.load_factories()
    reports = loader.load_recommendation_reports()

    for fid in request.factory_ids:
        try:
            mask = factories_df["factory_id"] == fid
            if not mask.any():
                errors.append({"factory_id": fid, "error": f"Factory '{fid}' not found"})
                continue

            factory_row = factories_df[mask].iloc[0]
            city = str(factory_row.get("city", ""))
            lat = float(factory_row.get("latitude", 0.0))
            lon = float(factory_row.get("longitude", 0.0))
            factory_name = str(factory_row.get("factory_name", fid))
            industry_type = str(factory_row.get("industry_type", ""))

            report = next((r for r in reports if str(r.get("factory_id", "")) == fid), None)
            composite_score, dominant_pollutant, pollution_readings = _extract_report_data(report)

            aqi_data: dict = {}
            if use_live_aqi:
                aqi_data = await _fetch_aqi_live(city, lat, lon)

            if aqi_data:
                for pol in ("pm25", "pm10", "no2", "so2", "co", "o3"):
                    if aqi_data.get(pol) is not None:
                        pollution_readings[pol] = aqi_data[pol]

            current_aqi_val = (
                float(aqi_data["aqi"]) if aqi_data.get("aqi") else max(composite_score * 40.0, 50.0)
            )
            data_source = aqi_data.get("source", "cached") if aqi_data else "cached"

            rec = _calculator.calculate_trees_needed(
                factory_id=fid,
                city=city,
                current_aqi=current_aqi_val,
                pollution_score=composite_score,
                dominant_pollutant=dominant_pollutant,
                pollution_readings=pollution_readings,
            )

            # Fallback to cached pollution readings when live AQI data is unavailable,
            # mirroring the single-factory endpoint behavior.
            if not aqi_data:
                aqi_data = {}
                for pol in ("pm25", "pm10", "no2", "so2", "co", "o3"):
                    value = pollution_readings.get(pol)
                    if value is not None:
                        aqi_data[pol] = value
                # Use the calculator's AQI index for the response when live AQI is missing.
                aqi_index = getattr(rec, "aqi_index", None)
                if aqi_index is not None:
                    aqi_data["aqi"] = aqi_index
                aqi_data["source"] = "cached"

            results.append(
                _build_response(fid, factory_name, city, industry_type, rec, aqi_data, data_source)
            )

        except HTTPException as exc:
            errors.append({"factory_id": fid, "error": str(exc.detail)})
        except Exception as exc:
            logger.exception("Bulk tree calc failed for factory %s: %s", fid, exc)
            errors.append({"factory_id": fid, "error": str(exc)})

    return TreeCalculatorBulkResponse(total=len(results), results=results, errors=errors)


@router.get(
    "/tree-calculator/constants",
    summary="Get scientific constants used in tree calculations",
)
def get_calculator_constants() -> dict:
    """Return all constants used by :class:`TreePlantingCalculator`.

    Intended for frontend methodology transparency and documentation.
    """
    return {
        "particulate_matter_absorption": {
            "tree_pm25_absorption_ug_m3": TREE_PM25_ABSORPTION_UG_M3,
            "tree_pm10_absorption_ug_m3": TREE_PM10_ABSORPTION_UG_M3,
        },
        "carbon_absorption": {
            "tree_co2_absorption_kg_year": TREE_CO2_ABSORPTION_KG_YEAR,
            "tree_co2_absorption_tons_year": TREE_CO2_ABSORPTION_TONS_YEAR,
        },
        "physical_coverage": {
            "tree_coverage_area_m2": TREE_COVERAGE_AREA_M2,
            "tree_canopy_radius_m": TREE_CANOPY_RADIUS_M,
        },
        "aqi_thresholds_cpcb_india": {
            "good": AQI_GOOD,
            "satisfactory": AQI_SATISFACTORY,
            "moderate": AQI_MODERATE,
            "poor": AQI_POOR,
            "very_poor": AQI_VERY_POOR,
        },
        "safety_buffers": {
            "minimum": MINIMUM_BUFFER,
            "recommended": RECOMMENDED_BUFFER,
            "optimal": OPTIMAL_BUFFER,
        },
        "aqi_reduction_per_1000_trees": AQI_REDUCTION_PER_1000_TREES,
        "tree_maturity": {
            "maturity_years": MATURITY_YEARS,
            "trees_per_hectare": TREES_PER_HECTARE,
        },
    }
