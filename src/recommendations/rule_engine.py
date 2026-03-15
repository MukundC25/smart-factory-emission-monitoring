"""
RuleEngine: Applies domain expert rules to generate pollution control recommendations.

- Implements pollutant-specific and industry-specific rules
- Returns structured Recommendation dataclass objects
- All config via arguments, no hardcoded paths
- Type hints and Google-style docstrings throughout
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Structured recommendation for pollution control action."""
    category: str
    priority: str
    action: str
    pollutant: str
    estimated_reduction: str
    cost_category: str
    timeline: str

class RuleEngine:
    """Applies expert rules to risk scores to generate recommendations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RuleEngine with optional config."""
        self.config = config or {}

    def apply_rules(self, risk_scores: Dict[str, Any]) -> List[Recommendation]:
        """Apply domain rules to risk scores and return recommendations.

        Args:
            risk_scores: Output from PollutionRiskScorer.compute_factory_risk

        Returns:
            List[Recommendation]: Recommendations for the factory
        """
        recs: List[Recommendation] = []
        # Extract scores
        pm25 = risk_scores.get("pm25_score", 0)
        pm10 = risk_scores.get("pm10_score", 0)
        so2 = risk_scores.get("so2_score", 0)
        no2 = risk_scores.get("no2_score", 0)
        co = risk_scores.get("co_score", 0)
        o3 = risk_scores.get("o3_score", 0)
        composite = risk_scores.get("composite_score", 0)
        risk_level = risk_scores.get("risk_level", "Low")
        industry = (risk_scores.get("industry_type") or "").lower()

        # --- Pollutant-specific rules ---
        if so2 > 6:
            recs.append(Recommendation(
                category="Emission Control",
                priority="Immediate",
                action="Install wet scrubber / flue gas desulfurization (FGD) system",
                pollutant="so2",
                estimated_reduction="60-80% SO2 reduction",
                cost_category="High",
                timeline="3-6 months installation"
            ))
            recs.append(Recommendation(
                category="Emission Control",
                priority="Short-term",
                action="Switch to low-sulfur fuel (< 0.5% sulfur content)",
                pollutant="so2",
                estimated_reduction="20-40% SO2 reduction",
                cost_category="Medium",
                timeline="1-2 months"
            ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Immediate",
                action="Install continuous SO2 emissions monitoring system (CEMS)",
                pollutant="so2",
                estimated_reduction="N/A",
                cost_category="Medium",
                timeline="1-2 months"
            ))
        if pm25 > 6 or pm10 > 6:
            recs.append(Recommendation(
                category="Emission Control",
                priority="Immediate",
                action="Install bag filter / fabric filter system",
                pollutant="pm25/pm10",
                estimated_reduction="70-90% PM reduction",
                cost_category="High",
                timeline="3-6 months installation"
            ))
            recs.append(Recommendation(
                category="Emission Control",
                priority="Short-term",
                action="Electrostatic precipitator (ESP) for high-temperature processes",
                pollutant="pm25/pm10",
                estimated_reduction="60-80% PM reduction",
                cost_category="High",
                timeline="3-6 months installation"
            ))
            if pm10 > 6:
                recs.append(Recommendation(
                    category="Emission Control",
                    priority="Long-term",
                    action="Cyclone separator for coarse particulates (PM10)",
                    pollutant="pm10",
                    estimated_reduction="40-60% PM10 reduction",
                    cost_category="Medium",
                    timeline="6-12 months"
                ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Immediate",
                action="Install real-time particulate matter sensor at stack",
                pollutant="pm25/pm10",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="1 month"
            ))
        if no2 > 6:
            recs.append(Recommendation(
                category="Emission Control",
                priority="Immediate",
                action="Selective catalytic reduction (SCR) system",
                pollutant="no2",
                estimated_reduction="70-90% NOx reduction",
                cost_category="High",
                timeline="6-12 months"
            ))
            recs.append(Recommendation(
                category="Emission Control",
                priority="Short-term",
                action="Low-NOx burner technology retrofit",
                pollutant="no2",
                estimated_reduction="30-50% NOx reduction",
                cost_category="Medium",
                timeline="3-6 months"
            ))
            recs.append(Recommendation(
                category="Emission Control",
                priority="Long-term",
                action="Flue gas recirculation (FGR)",
                pollutant="no2",
                estimated_reduction="10-20% NOx reduction",
                cost_category="Medium",
                timeline="6-12 months"
            ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Immediate",
                action="Install NOx CEMS at combustion sources",
                pollutant="no2",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="1-2 months"
            ))
        if co > 6:
            recs.append(Recommendation(
                category="Process Optimization",
                priority="Immediate",
                action="Optimize combustion air-fuel ratio",
                pollutant="co",
                estimated_reduction="10-30% CO reduction",
                cost_category="Low",
                timeline="1 month"
            ))
            recs.append(Recommendation(
                category="Emission Control",
                priority="Short-term",
                action="Install CO catalytic oxidizer",
                pollutant="co",
                estimated_reduction="60-90% CO reduction",
                cost_category="Medium",
                timeline="3-6 months"
            ))
            recs.append(Recommendation(
                category="Maintenance",
                priority="Long-term",
                action="Regular boiler/furnace maintenance schedule",
                pollutant="co",
                estimated_reduction="10-20% CO reduction",
                cost_category="Low",
                timeline="Ongoing"
            ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Immediate",
                action="CO detector network across facility",
                pollutant="co",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="1 month"
            ))
        if o3 > 6:
            recs.append(Recommendation(
                category="Process Optimization",
                priority="Immediate",
                action="Reduce NOx and VOC precursor emissions",
                pollutant="o3",
                estimated_reduction="10-30% O3 reduction",
                cost_category="Medium",
                timeline="3-6 months"
            ))
            recs.append(Recommendation(
                category="Process Optimization",
                priority="Short-term",
                action="Schedule high-emission operations at night",
                pollutant="o3",
                estimated_reduction="5-10% O3 reduction",
                cost_category="Low",
                timeline="1-2 months"
            ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Immediate",
                action="Install ambient ozone monitoring station",
                pollutant="o3",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="1-2 months"
            ))
        if sum([pm25 > 6, pm10 > 6, so2 > 6, no2 > 6, co > 6, o3 > 6]) > 1 or composite > 6:
            recs.append(Recommendation(
                category="Compliance",
                priority="Immediate",
                action="Comprehensive emission audit within 30 days",
                pollutant="multiple",
                estimated_reduction="N/A",
                cost_category="Medium",
                timeline="1 month"
            ))
            recs.append(Recommendation(
                category="Compliance",
                priority="Short-term",
                action="Apply for pollution control extension if retrofitting",
                pollutant="multiple",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="1-2 months"
            ))
        # --- Risk-level rules ---
        if risk_level == "Medium":
            recs.append(Recommendation(
                category="Maintenance",
                priority="Short-term",
                action="Preventive maintenance schedule for emission control equipment",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Quarterly"
            ))
            recs.append(Recommendation(
                category="Monitoring",
                priority="Short-term",
                action="Quarterly emission stack testing",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Quarterly"
            ))
            recs.append(Recommendation(
                category="Training",
                priority="Short-term",
                action="Employee environmental training program",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Annual"
            ))
            recs.append(Recommendation(
                category="Audit",
                priority="Short-term",
                action="Energy efficiency audit to reduce overall emissions",
                pollutant="all",
                estimated_reduction="5-10% total reduction",
                cost_category="Low",
                timeline="Annual"
            ))
        if risk_level == "Low":
            recs.append(Recommendation(
                category="Compliance",
                priority="Long-term",
                action="Annual compliance review",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Annual"
            ))
            recs.append(Recommendation(
                category="Maintenance",
                priority="Long-term",
                action="Maintain current emission control systems",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Ongoing"
            ))
            recs.append(Recommendation(
                category="Documentation",
                priority="Long-term",
                action="Document best practices for ISO 14001 certification",
                pollutant="all",
                estimated_reduction="N/A",
                cost_category="Low",
                timeline="Annual"
            ))
        # --- Industry-specific rules ---
        if industry in {"steel", "metal", "metal_processing"}:
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize SO2 and PM controls, improve slag handling procedures",
                pollutant="so2/pm",
                estimated_reduction="10-20% total reduction",
                cost_category="Medium",
                timeline="3-6 months"
            ))
        elif industry == "chemical":
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize SO2, NO2, CO, and VOC controls",
                pollutant="so2/no2/co/voc",
                estimated_reduction="15-25% total reduction",
                cost_category="High",
                timeline="6-12 months"
            ))
        elif industry in {"power", "thermal"}:
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize SO2, NOx, PM controls, and carbon capture feasibility",
                pollutant="so2/nox/pm",
                estimated_reduction="20-30% total reduction",
                cost_category="High",
                timeline="12-24 months"
            ))
        elif industry == "cement":
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize PM10, PM2.5, and NOx controls",
                pollutant="pm10/pm25/nox",
                estimated_reduction="15-25% total reduction",
                cost_category="High",
                timeline="6-12 months"
            ))
        elif industry == "textile":
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize CO, VOC, and integrate wastewater co-treatment",
                pollutant="co/voc/wastewater",
                estimated_reduction="10-20% total reduction",
                cost_category="Medium",
                timeline="6-12 months"
            ))
        elif industry == "pharmaceutical":
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Immediate",
                action="Prioritize VOC, CO, and hazardous waste handling",
                pollutant="voc/co/hazardous_waste",
                estimated_reduction="10-20% total reduction",
                cost_category="High",
                timeline="6-12 months"
            ))
        elif industry not in {"steel", "metal", "metal_processing", "chemical", "power", "thermal", "cement", "textile", "pharmaceutical"}:
            recs.append(Recommendation(
                category="Industry-Specific",
                priority="Short-term",
                action="Apply general pollution control measures as per CPCB guidelines",
                pollutant="all",
                estimated_reduction="5-10% total reduction",
                cost_category="Low",
                timeline="6-12 months"
            ))
        return recs
