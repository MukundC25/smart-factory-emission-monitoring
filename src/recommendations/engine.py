"""Generate factory-level pollution control recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common import get_project_root, initialize_environment
from src.recommendations.ml_recommender import MLRecommender
from src.recommendations.risk_scorer import PollutionRiskScorer
from src.recommendations.rule_engine import Recommendation, RuleEngine

LOGGER = logging.getLogger(__name__)


@dataclass
class FactoryReport:
    """Structured recommendation report for a single factory."""

    factory_id: str
    factory_name: str
    industry_type: str
    city: str
    risk_level: str
    composite_score: float
    dominant_pollutant: str
    pollution_scores: Dict[str, float]
    recommendations: List[Recommendation]
    summary: str
    generated_at: datetime


class HybridRecommendationEngine:
    """Hybrid recommendation orchestrator combining rule and ML suggestions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize HybridRecommendationEngine.

        Args:
            config: Optional runtime configuration.
        """
        self.config = config or initialize_environment()
        rec_cfg = self.config.get("recommendations", {})
        self.rule_weight = float(rec_cfg.get("rule_weight", 0.7))
        self.ml_weight = float(rec_cfg.get("ml_weight", 0.3))
        self.max_station_distance_km = float(rec_cfg.get("max_station_distance_km", 100.0))

        self.risk_scorer = PollutionRiskScorer()
        self.rule_engine = RuleEngine(self.config)
        self.ml_recommender = MLRecommender(self.config)

    def _select_pollution_row(self, factory_row: pd.Series, pollution_df: pd.DataFrame) -> pd.Series:
        """Select pollution row for factory using precomputed nearest distance and fallbacks.

        Args:
            factory_row: Factory record.
            pollution_df: Pollution dataset with nearest_factory_distance_km.

        Returns:
            pd.Series: Selected pollution row or fallback aggregate row.
        """
        if pollution_df.empty:
            return pd.Series(dtype="float64")

        city = str(factory_row.get("city", ""))
        has_distance = "nearest_factory_distance_km" in pollution_df.columns

        if has_distance:
            direct = pollution_df[
                (pollution_df.get("city", "") == city)
                & (pollution_df["nearest_factory_distance_km"] <= self.max_station_distance_km)
            ]
            if not direct.empty:
                return direct.sort_values("nearest_factory_distance_km", ascending=True).iloc[0]

        city_rows = pollution_df[pollution_df.get("city", "") == city]
        if not city_rows.empty:
            city_means = city_rows.mean(numeric_only=True)
            city_means["city"] = city
            return city_means

        national_means = pollution_df.mean(numeric_only=True)
        national_means["city"] = city
        return national_means

    @staticmethod
    def _priority_rank(priority: str) -> int:
        """Convert priority label to sortable rank.

        Args:
            priority: Recommendation priority text.

        Returns:
            int: Lower value indicates higher priority.
        """
        order = {"Immediate": 0, "Short-term": 1, "Long-term": 2}
        return order.get(priority, 3)

    def _ml_category_to_recommendation(self, category: str, dominant_pollutant: str) -> Recommendation:
        """Map ML category to a default Recommendation object.

        Args:
            category: ML predicted category.
            dominant_pollutant: Dominant pollutant for context.

        Returns:
            Recommendation: Structured recommendation generated from ML category.
        """
        return Recommendation(
            category="ML Suggested",
            priority="Short-term",
            action=f"Apply data-driven intervention plan for category: {category}",
            pollutant=dominant_pollutant,
            estimated_reduction="Data-driven estimate pending site audit",
            cost_category="Medium",
            timeline="2-6 months",
        )

    def _merge_recommendations(
        self,
        rule_recs: List[Recommendation],
        ml_categories: List[str],
        dominant_pollutant: str,
    ) -> List[Recommendation]:
        """Merge and deduplicate rule-based and ML recommendations.

        Args:
            rule_recs: Rule-based recommendations.
            ml_categories: ML category predictions.
            dominant_pollutant: Dominant pollutant from risk scoring.

        Returns:
            List[Recommendation]: Ranked and deduplicated recommendations.
        """
        weighted_rows: List[tuple[float, Recommendation]] = []
        for rec in rule_recs:
            rank = self._priority_rank(rec.priority)
            weighted_rows.append((self.rule_weight * (10 - rank), rec))

        for category in ml_categories:
            ml_rec = self._ml_category_to_recommendation(category, dominant_pollutant)
            rank = self._priority_rank(ml_rec.priority)
            weighted_rows.append((self.ml_weight * (10 - rank), ml_rec))

        weighted_rows.sort(key=lambda row: (-row[0], self._priority_rank(row[1].priority), row[1].action))

        deduped: List[Recommendation] = []
        seen: set[tuple[str, str, str]] = set()
        for _, rec in weighted_rows:
            key = (rec.category, rec.priority, rec.action)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(rec)
        return deduped

    def generate_recommendations(self, factory_row: pd.Series, pollution_row: pd.Series) -> FactoryReport:
        """Generate recommendation report for one factory.

        Args:
            factory_row: Factory record.
            pollution_row: Matched pollution record.

        Returns:
            FactoryReport: Generated recommendation report.
        """
        risk_scores = self.risk_scorer.compute_factory_risk(factory_row, pollution_row)
        rule_recs = self.rule_engine.apply_rules(risk_scores)
        ml_categories = self.ml_recommender.predict_recommendations(risk_scores)
        merged = self._merge_recommendations(rule_recs, ml_categories, str(risk_scores.get("dominant_pollutant", "all")))

        pollution_scores = {
            "pm25_score": float(risk_scores.get("pm25_score", 0.0) or 0.0),
            "pm10_score": float(risk_scores.get("pm10_score", 0.0) or 0.0),
            "so2_score": float(risk_scores.get("so2_score", 0.0) or 0.0),
            "no2_score": float(risk_scores.get("no2_score", 0.0) or 0.0),
            "co_score": float(risk_scores.get("co_score", 0.0) or 0.0),
            "o3_score": float(risk_scores.get("o3_score", 0.0) or 0.0),
        }

        top_action = merged[0].action if merged else "No specific recommendation available"
        summary = (
            f"{risk_scores.get('factory_name', 'Factory')} is classified as {risk_scores.get('risk_level', 'Unknown')} "
            f"risk with composite score {float(risk_scores.get('composite_score', 0.0) or 0.0):.2f}. "
            f"Top action: {top_action}."
        )

        return FactoryReport(
            factory_id=str(risk_scores.get("factory_id", "")),
            factory_name=str(risk_scores.get("factory_name", "")),
            industry_type=str(risk_scores.get("industry_type", "")),
            city=str(risk_scores.get("city", "")),
            risk_level=str(risk_scores.get("risk_level", "Low")),
            composite_score=float(risk_scores.get("composite_score", 0.0) or 0.0),
            dominant_pollutant=str(risk_scores.get("dominant_pollutant", "")),
            pollution_scores=pollution_scores,
            recommendations=merged,
            summary=summary,
            generated_at=datetime.now(timezone.utc),
        )

    def generate_all(self, factories_df: pd.DataFrame, pollution_df: pd.DataFrame) -> List[FactoryReport]:
        """Generate recommendation reports for all factories.

        Args:
            factories_df: Factory dataframe.
            pollution_df: Pollution dataframe.

        Returns:
            List[FactoryReport]: Reports sorted by composite score descending.
        """
        if factories_df.empty:
            LOGGER.warning("No factory records available for recommendation generation")
            return []
        if pollution_df.empty:
            LOGGER.warning("No pollution records available; generating reports with fallback defaults")

        reports: List[FactoryReport] = []
        for index, (_, factory_row) in enumerate(factories_df.iterrows(), start=1):
            pollution_row = self._select_pollution_row(factory_row, pollution_df)
            report = self.generate_recommendations(factory_row, pollution_row)
            reports.append(report)

            if index % 10 == 0:
                LOGGER.info("Generated recommendations for %d factories", index)

        reports.sort(key=lambda report: report.composite_score, reverse=True)
        return reports


def _risk_level(score: float, config: Dict[str, Any]) -> str:
    """Map score to risk category.

    Args:
        score: Pollution impact score.
        config: Runtime configuration.

    Returns:
        str: Risk category.
    """
    if score <= float(config["risk_bands"]["low_max"]):
        return "Low"
    if score <= float(config["risk_bands"]["medium_max"]):
        return "Medium"
    return "High"


def _recommendation_text(risk_level: str, industry_type: str) -> str:
    """Generate plain-English recommendation text.

    Args:
        risk_level: Risk level label.
        industry_type: Factory industry type.

    Returns:
        str: Recommendation text.
    """
    if risk_level == "High":
        return (
            f"{industry_type.title()} site is high-risk. Install SO2 scrubbers and ESP filters, "
            "upgrade baghouse maintenance cadence to weekly, and deploy continuous emissions "
            "monitoring with automated alerts."
        )
    if risk_level == "Medium":
        return (
            f"{industry_type.title()} site is medium-risk. Increase stack testing to bi-weekly, "
            "perform preventive burner tuning, and add leak-detection walkthroughs each shift."
        )
    return (
        f"{industry_type.title()} site is low-risk. Maintain compliance logs, keep monthly "
        "calibration checks, and sustain preventive maintenance schedules."
    )


def generate_recommendations(config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Create factory-level risk scoring and recommendations.

    Args:
        config: Optional runtime configuration.

    Returns:
        pd.DataFrame: Recommendation dataset.
    """
    runtime_config = config or initialize_environment()
    root = get_project_root()
    processed_path = root / runtime_config["paths"]["processed_dataset"]

    dataset = pd.read_parquet(processed_path)
    target_col = runtime_config["ml"]["target_column"]

    factory_view = (
        dataset.groupby(
            ["factory_id", "factory_name", "industry_type", "latitude", "longitude", "city", "state", "country"],
            as_index=False,
        )
        .agg(
            pollution_impact_score=(target_col, "mean"),
            latest_pm25=("pm25", "mean"),
            latest_pm10=("pm10", "mean"),
        )
        .sort_values("pollution_impact_score", ascending=False)
    )

    factory_view["risk_level"] = factory_view["pollution_impact_score"].apply(
        lambda score: _risk_level(float(score), runtime_config)
    )
    factory_view["recommendation"] = factory_view.apply(
        lambda row: _recommendation_text(row["risk_level"], row["industry_type"]),
        axis=1,
    )

    output_path = root / runtime_config["paths"]["recommendations"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    factory_view.to_csv(output_path, index=False)
    LOGGER.info("Recommendations written to %s (%s rows)", output_path, len(factory_view))
    return factory_view


def main() -> None:
    """Run recommendation engine standalone."""
    config = initialize_environment()
    generate_recommendations(config)


if __name__ == "__main__":
    main()
