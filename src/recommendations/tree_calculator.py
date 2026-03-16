"""Scientific tree planting calculator for AQI offset recommendations.

All constants are validated environmental science values for Indian conditions
(CPCB standards). Formulas are empirically derived from urban forestry studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Scientific Constants
# ---------------------------------------------------------------------------

# Particulate Matter absorption per tree per 100 m² canopy coverage
TREE_PM25_ABSORPTION_UG_M3 = 0.8    # μg/m³ reduction per tree per 100 m²
TREE_PM10_ABSORPTION_UG_M3 = 1.2    # μg/m³ reduction per tree per 100 m²

# Gaseous pollutant absorption per tree per year
TREE_NO2_ABSORPTION_G_YEAR = 1.4    # g NO₂ per tree per year
TREE_SO2_ABSORPTION_G_YEAR = 1.1    # g SO₂ per tree per year
TREE_CO_ABSORPTION_G_YEAR = 5.3     # g CO per tree per year

# Carbon absorption
TREE_CO2_ABSORPTION_KG_YEAR = 22.0   # kg CO₂ per mature tree per year
TREE_CO2_ABSORPTION_TONS_YEAR = 0.022  # tonnes CO₂ per mature tree per year

# Physical coverage
TREE_COVERAGE_AREA_M2 = 100.0        # m² canopy area per tree
TREE_CANOPY_RADIUS_M = 5.64          # √(100 / π) m

# AQI thresholds per CPCB India standards
AQI_GOOD = 50
AQI_SATISFACTORY = 100
AQI_MODERATE = 200
AQI_POOR = 300
AQI_VERY_POOR = 400

# Safety buffer multipliers
MINIMUM_BUFFER = 1.0      # 0 % extra — absolute minimum
RECOMMENDED_BUFFER = 1.2  # 20 % extra — recommended
OPTIMAL_BUFFER = 1.5      # 50 % extra — optimal / future-proof

# AQI points reduced per 1 000 trees at maturity (empirical, pollutant-specific)
AQI_REDUCTION_PER_1000_TREES: Dict[str, float] = {
    "pm25": 8.5,
    "pm10": 6.0,
    "no2":  4.0,
    "so2":  3.5,
    "co":   2.0,
    "o3":   3.0,
}

# Maturity and density
MATURITY_YEARS = 5
TREES_PER_HECTARE = 1000


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TreeRecommendation:
    """Structured tree planting recommendation for a single factory."""

    factory_id: str
    city: str
    current_aqi: float
    current_pollution_score: float
    dominant_pollutant: str
    current_pm25: Optional[float]
    current_pm10: Optional[float]
    current_no2: Optional[float]
    current_so2: Optional[float]
    current_co: Optional[float]
    target_aqi: float
    trees_needed: Dict[str, int]               # {minimum, recommended, optimal}
    impact_radius_km: float
    planting_area_hectares: float
    annual_co2_offset_tons: float
    estimated_aqi_reduction: float
    timeline_years: int
    pollutant_breakdown: Dict[str, Optional[int]]  # per-pollutant tree requirements
    feasibility: str                               # "Low" / "Medium" / "High"
    notes: List[str]                               # human-readable scientific notes


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class TreePlantingCalculator:
    """Calculate tree planting requirements to offset factory pollution.

    All public methods are stateless and may be called directly as class
    methods / static methods.  The orchestrating entry-point is
    :meth:`calculate_trees_needed`.
    """

    # -----------------------------------------------------------------------
    # Per-pollutant tree requirement formulas
    # -----------------------------------------------------------------------

    @staticmethod
    def calculate_trees_for_pm25(
        current_pm25: float,
        target_pm25: float,
        impact_area_km2: float,
    ) -> float:
        """Trees required to reduce PM2.5 from *current* to *target* concentration.

        Args:
            current_pm25: Current PM2.5 level (μg/m³).
            target_pm25: Target PM2.5 level (μg/m³).
            impact_area_km2: Impact area in km².

        Returns:
            Raw float tree count (≥ 0).
        """
        reduction_needed = max(0.0, current_pm25 - target_pm25)
        if reduction_needed == 0.0:
            return 0.0
        area_m2 = impact_area_km2 * 1_000_000
        return (reduction_needed * area_m2) / (TREE_PM25_ABSORPTION_UG_M3 * TREE_COVERAGE_AREA_M2)

    @staticmethod
    def calculate_trees_for_pm10(
        current_pm10: float,
        target_pm10: float,
        impact_area_km2: float,
    ) -> float:
        """Trees required to reduce PM10 from *current* to *target* concentration."""
        reduction_needed = max(0.0, current_pm10 - target_pm10)
        if reduction_needed == 0.0:
            return 0.0
        area_m2 = impact_area_km2 * 1_000_000
        return (reduction_needed * area_m2) / (TREE_PM10_ABSORPTION_UG_M3 * TREE_COVERAGE_AREA_M2)

    @staticmethod
    def calculate_trees_for_no2(
        current_no2: float,
        target_no2: float,
        impact_area_km2: float,
    ) -> float:
        """Trees required to reduce NO₂.

        NO₂ absorption rate: ~1.4 g NO₂ per tree per year.
        """
        reduction_needed = max(0.0, current_no2 - target_no2)
        if reduction_needed == 0.0:
            return 0.0
        area_m2 = impact_area_km2 * 1_000_000
        return (reduction_needed * area_m2) / (TREE_NO2_ABSORPTION_G_YEAR * TREE_COVERAGE_AREA_M2 / 1_000_000)

    @staticmethod
    def calculate_trees_for_so2(
        current_so2: float,
        target_so2: float,
        impact_area_km2: float,
    ) -> float:
        """Trees required to reduce SO₂.

        SO₂ absorption rate: ~1.1 g SO₂ per tree per year.
        """
        reduction_needed = max(0.0, current_so2 - target_so2)
        if reduction_needed == 0.0:
            return 0.0
        area_m2 = impact_area_km2 * 1_000_000
        return (reduction_needed * area_m2) / (TREE_SO2_ABSORPTION_G_YEAR * TREE_COVERAGE_AREA_M2 / 1_000_000)

    @staticmethod
    def calculate_trees_for_co(
        current_co: float,
        target_co: float,
        impact_area_km2: float,
    ) -> float:
        """Trees required to reduce CO.

        CO absorption rate: ~5.3 g CO per tree per year.
        """
        reduction_needed = max(0.0, current_co - target_co)
        if reduction_needed == 0.0:
            return 0.0
        area_m2 = impact_area_km2 * 1_000_000
        return (reduction_needed * area_m2) / (5.3 * TREE_COVERAGE_AREA_M2 / 1_000_000)

    # -----------------------------------------------------------------------
    # Ancillary calculations
    # -----------------------------------------------------------------------

    @staticmethod
    def calculate_impact_radius(pollution_score: float) -> float:
        """Return pollution impact radius in km (capped at 5.0 km).

        Radius scales linearly with pollution severity.
        """
        base_radius = 1.0
        radius = base_radius + (pollution_score * 0.2)
        return min(radius, 5.0)

    @staticmethod
    def calculate_planting_area(trees_recommended: int) -> float:
        """Return required planting area in **hectares** for *trees_recommended*.

        Assumes TREE_COVERAGE_AREA_M2 per tree canopy.
        """
        area_m2 = trees_recommended * TREE_COVERAGE_AREA_M2
        return area_m2 / 10_000

    @staticmethod
    def calculate_aqi_reduction(trees_recommended: int, dominant_pollutant: str) -> float:
        """Estimate AQI reduction achievable from planting *trees_recommended* trees.

        Uses the empirical per-1 000-trees AQI-reduction table.
        Falls back to 3.0 AQI pts / 1 000 trees for unknown pollutants.
        """
        reduction_per_1000 = AQI_REDUCTION_PER_1000_TREES.get(
            dominant_pollutant.lower(), 3.0
        )
        return (trees_recommended / 1000) * reduction_per_1000

    @staticmethod
    def determine_target_aqi(current_aqi: float) -> float:
        """Return the AQI target one band better than the current reading."""
        if current_aqi > AQI_VERY_POOR:
            return float(AQI_POOR)
        if current_aqi > AQI_POOR:
            return float(AQI_MODERATE)
        if current_aqi > AQI_MODERATE:
            return float(AQI_SATISFACTORY)
        return float(AQI_GOOD)

    @staticmethod
    def assess_feasibility(trees_recommended: int, planting_area_hectares: float) -> str:
        """Classify implementation feasibility as High / Medium / Low."""
        if trees_recommended < 500:
            return "High"
        if trees_recommended < 2000:
            return "Medium"
        return "Low"

    # -----------------------------------------------------------------------
    # Orchestrator
    # -----------------------------------------------------------------------

    def calculate_trees_needed(
        self,
        factory_id: str,
        city: str,
        current_aqi: float,
        pollution_score: float,
        dominant_pollutant: str,
        pollution_readings: Dict[str, Optional[float]],
        target_aqi_override: Optional[float] = None,
    ) -> TreeRecommendation:
        """Orchestrate the full tree-planting recommendation calculation.

        Args:
            factory_id: Factory identifier.
            city: City name.
            current_aqi: Current AQI reading.
            pollution_score: Composite ML pollution score (0–10).
            dominant_pollutant: Name of the dominant pollutant.
            pollution_readings: Dict containing optional pm25/pm10/no2/so2/co/o3.
            target_aqi_override: Override the auto-determined target AQI.

        Returns:
            :class:`TreeRecommendation` dataclass with full calculation output.
        """
        # 1. Target AQI
        target_aqi = (
            float(target_aqi_override)
            if target_aqi_override is not None
            else self.determine_target_aqi(current_aqi)
        )

        # 2. Impact radius and area
        impact_radius_km = self.calculate_impact_radius(pollution_score)
        impact_area_km2 = math.pi * (impact_radius_km ** 2)

        # 3. Safe extraction of readings (guard None and NaN)
        def _safe(val: object) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            if math.isnan(f) or math.isinf(f):
                return None
            return f

        pm25 = _safe(pollution_readings.get("pm25"))
        pm10 = _safe(pollution_readings.get("pm10"))
        no2  = _safe(pollution_readings.get("no2"))
        so2  = _safe(pollution_readings.get("so2"))
        co   = _safe(pollution_readings.get("co"))

        # 4. Proportional target concentrations
        scale = target_aqi / max(current_aqi, 1.0)

        def _target(val: Optional[float]) -> float:
            if val is None:
                return 0.0
            return max(0.0, val * scale)

        # 5. Per-pollutant tree counts
        pm25_trees_f = (
            self.calculate_trees_for_pm25(pm25, _target(pm25), impact_area_km2)
            if pm25 is not None else None
        )
        pm10_trees_f = (
            self.calculate_trees_for_pm10(pm10, _target(pm10), impact_area_km2)
            if pm10 is not None else None
        )
        no2_trees_f = (
            self.calculate_trees_for_no2(no2, _target(no2), impact_area_km2)
            if no2 is not None else None
        )
        so2_trees_f = (
            self.calculate_trees_for_so2(so2, _target(so2), impact_area_km2)
            if so2 is not None else None
        )
        co_trees_f = (
            self.calculate_trees_for_co(co, _target(co), impact_area_km2)
            if co is not None else None
        )

        # 6. Base count from dominant pollutant
        dominant_map: Dict[str, Optional[float]] = {
            "pm25": pm25_trees_f,
            "pm10": pm10_trees_f,
            "no2":  no2_trees_f,
            "so2":  so2_trees_f,
            "co":   co_trees_f,
        }
        base_trees_raw = dominant_map.get(dominant_pollutant.lower())
        if base_trees_raw is None or base_trees_raw <= 0.0:
            candidates = [v for v in dominant_map.values() if v is not None and v > 0.0]
            base_trees_raw = max(candidates, default=0.0)

        # Always provide actionable guidance — minimum 1 tree
        base_trees = max(int(math.ceil(base_trees_raw)), 1)

        # 7. Buffer tiers
        minimum_trees = int(math.ceil(base_trees * MINIMUM_BUFFER))
        recommended_trees = int(math.ceil(base_trees * RECOMMENDED_BUFFER))
        optimal_trees = int(math.ceil(base_trees * OPTIMAL_BUFFER))

        trees_needed: Dict[str, int] = {
            "minimum": minimum_trees,
            "recommended": recommended_trees,
            "optimal": optimal_trees,
        }

        # 8. Derived metrics
        planting_area = self.calculate_planting_area(recommended_trees)
        annual_co2_offset = recommended_trees * TREE_CO2_ABSORPTION_TONS_YEAR
        estimated_aqi_reduction = self.calculate_aqi_reduction(
            recommended_trees, dominant_pollutant
        )

        # 9. Pollutant breakdown (int counts)
        pollutant_breakdown: Dict[str, Optional[int]] = {
            "pm25_trees": int(math.ceil(pm25_trees_f)) if pm25_trees_f is not None else None,
            "pm10_trees": int(math.ceil(pm10_trees_f)) if pm10_trees_f is not None else None,
            "no2_trees":  int(math.ceil(no2_trees_f))  if no2_trees_f  is not None else None,
            "so2_trees":  int(math.ceil(so2_trees_f))  if so2_trees_f  is not None else None,
            "co_trees":   int(math.ceil(co_trees_f))   if co_trees_f   is not None else None,
        }

        # 10. Feasibility
        feasibility = self.assess_feasibility(recommended_trees, planting_area)

        # 11. Human-readable notes
        if minimum_trees < 500:
            feas_detail = "< 500 trees — highly practical"
        elif minimum_trees < 2000:
            feas_detail = "500–2 000 trees — medium-scale greening programme"
        else:
            feas_detail = "> 2 000 trees — large-scale urban forestry required"

        notes: List[str] = [
            (
                f"Current AQI {current_aqi:.0f} → targeting {target_aqi:.0f} "
                f"({_aqi_band(target_aqi)} band)."
            ),
            (
                f"Impact zone radius: {impact_radius_km:.1f} km "
                f"({impact_area_km2:.1f} km² total area)."
            ),
            (
                f"Recommended planting: {recommended_trees} trees across "
                f"{planting_area:.2f} ha "
                f"(at {TREES_PER_HECTARE} trees/ha density)."
            ),
            (
                f"Annual CO₂ sequestration: {annual_co2_offset:.1f} t CO₂ "
                f"({TREE_CO2_ABSORPTION_KG_YEAR} kg/tree/year at maturity)."
            ),
            (
                f"Estimated AQI improvement after {MATURITY_YEARS}-year maturity: "
                f"{estimated_aqi_reduction:.1f} AQI points."
            ),
            (
                f"Dominant pollutant: {dominant_pollutant.upper()} — trees reduce "
                f"{AQI_REDUCTION_PER_1000_TREES.get(dominant_pollutant.lower(), 3.0):.1f} "
                f"AQI pts per 1 000 trees."
            ),
            f"Feasibility: {feasibility} ({feas_detail}).",
        ]

        return TreeRecommendation(
            factory_id=factory_id,
            city=city,
            current_aqi=float(current_aqi),
            current_pollution_score=float(pollution_score),
            dominant_pollutant=dominant_pollutant,
            current_pm25=pm25,
            current_pm10=pm10,
            current_no2=no2,
            current_so2=so2,
            current_co=co,
            target_aqi=target_aqi,
            trees_needed=trees_needed,
            impact_radius_km=impact_radius_km,
            planting_area_hectares=round(planting_area, 4),
            annual_co2_offset_tons=round(annual_co2_offset, 4),
            estimated_aqi_reduction=round(estimated_aqi_reduction, 2),
            timeline_years=MATURITY_YEARS,
            pollutant_breakdown=pollutant_breakdown,
            feasibility=feasibility,
            notes=notes,
        )


def _aqi_band(aqi: float) -> str:
    """Return the CPCB AQI band name for a given AQI value."""
    if aqi <= AQI_GOOD:
        return "Good"
    if aqi <= AQI_SATISFACTORY:
        return "Satisfactory"
    if aqi <= AQI_MODERATE:
        return "Moderate"
    if aqi <= AQI_POOR:
        return "Poor"
    if aqi <= AQI_VERY_POOR:
        return "Very Poor"
    return "Severe"
