"""Root test fixtures and global pytest setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.recommendations.ml_recommender import MLRecommender

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("HEATMAP_INLINE_ASSETS", "0")


@pytest.fixture
def sample_factories_df() -> pd.DataFrame:
	"""Factory dataframe aligned with the production schema."""
	return pd.DataFrame(
		[
			{
				"factory_id": "FAC001",
				"factory_name": "Pune Steel Works",
				"industry_type": "steel",
				"latitude": 18.5204,
				"longitude": 73.8567,
				"city": "Pune",
				"state": "Maharashtra",
				"country": "India",
				"source": "OpenStreetMap",
				"osm_id": "node_1001",
				"last_updated": "2026-03-16",
			},
			{
				"factory_id": "FAC002",
				"factory_name": "Mumbai Chemical Plant",
				"industry_type": "chemical",
				"latitude": 19.076,
				"longitude": 72.8777,
				"city": "Mumbai",
				"state": "Maharashtra",
				"country": "India",
				"source": "OpenStreetMap",
				"osm_id": "way_1002",
				"last_updated": "2026-03-16",
			},
			{
				"factory_id": "FAC003",
				"factory_name": "Bengaluru Auto Works",
				"industry_type": "automotive",
				"latitude": 12.9716,
				"longitude": 77.5946,
				"city": "Bengaluru",
				"state": "Karnataka",
				"country": "India",
				"source": "OpenStreetMap",
				"osm_id": "relation_1003",
				"last_updated": "2026-03-16",
			},
			{
				"factory_id": "FAC004",
				"factory_name": "Chennai Textile Mill",
				"industry_type": "textile",
				"latitude": 13.0827,
				"longitude": 80.2707,
				"city": "Chennai",
				"state": "Tamil Nadu",
				"country": "India",
				"source": "OpenStreetMap",
				"osm_id": "node_1004",
				"last_updated": "2026-03-16",
			},
			{
				"factory_id": "FAC005",
				"factory_name": "Hyderabad Pharma Labs",
				"industry_type": "pharmaceutical",
				"latitude": 17.385,
				"longitude": 78.4867,
				"city": "Hyderabad",
				"state": "Telangana",
				"country": "India",
				"source": "OpenStreetMap",
				"osm_id": "node_1005",
				"last_updated": "2026-03-16",
			},
		]
	)


@pytest.fixture
def sample_pollution_df() -> pd.DataFrame:
	"""Pollution dataframe aligned with the production schema."""
	return pd.DataFrame(
		[
			{
				"pm25": 82.0,
				"pm10": 160.0,
				"co": 1.2,
				"no2": 64.0,
				"so2": 78.0,
				"o3": 42.0,
				"aqi_index": 198.0,
				"timestamp": "2026-03-14T10:00:00Z",
				"station_name": "Pune CPCB",
				"station_lat": 18.52,
				"station_lon": 73.85,
				"city": "Pune",
				"country": "India",
				"source": "synthetic",
				"nearest_factory_distance_km": 3.2,
			},
			{
				"pm25": 70.0,
				"pm10": 145.0,
				"co": 1.0,
				"no2": 55.0,
				"so2": 60.0,
				"o3": 39.0,
				"aqi_index": 170.0,
				"timestamp": "2026-03-14T10:05:00Z",
				"station_name": "Mumbai Chembur",
				"station_lat": 19.08,
				"station_lon": 72.88,
				"city": "Mumbai",
				"country": "India",
				"source": "synthetic",
				"nearest_factory_distance_km": 4.1,
			},
			{
				"pm25": 42.0,
				"pm10": 82.0,
				"co": 0.7,
				"no2": 34.0,
				"so2": 22.0,
				"o3": 25.0,
				"aqi_index": 110.0,
				"timestamp": "2026-03-14T10:10:00Z",
				"station_name": "Bengaluru BTM",
				"station_lat": 12.97,
				"station_lon": 77.59,
				"city": "Bengaluru",
				"country": "India",
				"source": "synthetic",
				"nearest_factory_distance_km": 2.7,
			},
			{
				"pm25": 28.0,
				"pm10": 64.0,
				"co": 0.4,
				"no2": 20.0,
				"so2": 12.0,
				"o3": 21.0,
				"aqi_index": 74.0,
				"timestamp": "2026-03-14T10:20:00Z",
				"station_name": "Chennai Central",
				"station_lat": 13.08,
				"station_lon": 80.27,
				"city": "Chennai",
				"country": "India",
				"source": "synthetic",
				"nearest_factory_distance_km": 2.9,
			},
			{
				"pm25": 58.0,
				"pm10": 102.0,
				"co": 0.8,
				"no2": 44.0,
				"so2": 36.0,
				"o3": 30.0,
				"aqi_index": 142.0,
				"timestamp": "2026-03-14T10:30:00Z",
				"station_name": "Hyderabad Balanagar",
				"station_lat": 17.39,
				"station_lon": 78.48,
				"city": "Hyderabad",
				"country": "India",
				"source": "synthetic",
				"nearest_factory_distance_km": 3.6,
			},
		]
	)


@pytest.fixture
def sample_recommendations() -> list[dict]:
	"""Recommendation report payload examples aligned to API schema."""
	return [
		{
			"factory_id": "FAC001",
			"factory_name": "Pune Steel Works",
			"industry_type": "steel",
			"city": "Pune",
			"risk_level": "Critical",
			"composite_score": 8.4,
			"dominant_pollutant": "so2",
			"pollution_scores": {
				"pm25_score": 7.2,
				"pm10_score": 7.8,
				"so2_score": 8.9,
				"no2_score": 6.4,
				"co_score": 2.1,
				"o3_score": 3.0,
			},
			"summary": "Critical risk with dominant SO2.",
			"generated_at": "2026-03-16T00:00:00+00:00",
			"recommendations": [
				{
					"category": "Emission Control",
					"priority": "Immediate",
					"action": "Install wet scrubber",
					"pollutant": "so2",
					"estimated_reduction": "60-80% SO2 reduction",
					"cost_category": "High",
					"timeline": "3-6 months",
				}
			],
		},
		{
			"factory_id": "FAC002",
			"factory_name": "Mumbai Chemical Plant",
			"industry_type": "chemical",
			"city": "Mumbai",
			"risk_level": "High",
			"composite_score": 7.1,
			"dominant_pollutant": "pm25",
			"pollution_scores": {
				"pm25_score": 8.0,
				"pm10_score": 6.8,
				"so2_score": 6.2,
				"no2_score": 5.5,
				"co_score": 2.3,
				"o3_score": 2.8,
			},
			"summary": "High risk due to PM2.5.",
			"generated_at": "2026-03-16T00:00:00+00:00",
			"recommendations": [],
		},
		{
			"factory_id": "FAC003",
			"factory_name": "Bengaluru Auto Works",
			"industry_type": "automotive",
			"city": "Bengaluru",
			"risk_level": "Low",
			"composite_score": 2.6,
			"dominant_pollutant": "pm10",
			"pollution_scores": {
				"pm25_score": 2.0,
				"pm10_score": 2.8,
				"so2_score": 1.7,
				"no2_score": 2.1,
				"co_score": 1.2,
				"o3_score": 1.9,
			},
			"summary": "Low risk profile.",
			"generated_at": "2026-03-16T00:00:00+00:00",
			"recommendations": [],
		},
	]


@pytest.fixture
def mock_config(tmp_path: Path) -> dict:
	"""Runtime config with all output paths redirected to tmp_path."""
	return {
		"paths": {
			"processed_dataset": str((tmp_path / "ml_dataset.parquet").as_posix()),
			"recommendations": str((tmp_path / "recommendations.csv").as_posix()),
			"model": str((tmp_path / "pollution_impact_model.pkl").as_posix()),
			"scaler": str((tmp_path / "scaler.pkl").as_posix()),
			"model_report": str((tmp_path / "model_report.json").as_posix()),
			"factories_raw": str((tmp_path / "factories_raw.csv").as_posix()),
			"factories_clean": str((tmp_path / "factories.csv").as_posix()),
			"factories_processed": str((tmp_path / "factories_processed.csv").as_posix()),
			"pollution_raw": str((tmp_path / "pollution_readings.csv").as_posix()),
			"pollution_processed": str((tmp_path / "pollution_clean.csv").as_posix()),
		},
		"recommendations": {
			"rule_weight": 0.7,
			"ml_weight": 0.3,
			"confidence_threshold": 0.4,
			"max_station_distance_km": 100,
			"output_csv": str((tmp_path / "recommendations.csv").as_posix()),
			"output_json": str((tmp_path / "recommendations.json").as_posix()),
			"model_path": str((tmp_path / "recommendation_model.pkl").as_posix()),
			"encoder_path": str((tmp_path / "recommendation_label_encoder.pkl").as_posix()),
		},
		"ml": {
			"random_state": 42,
			"n_estimators": 50,
			"target_column": "pollution_impact_score",
			"test_size": 0.2,
			"val_size": 0.2,
			"pollution_weights": {
				"pm25": 0.35,
				"pm10": 0.2,
				"no2": 0.15,
				"so2": 0.1,
				"co": 0.1,
				"o3": 0.1,
			},
		},
		"risk_bands": {"low_max": 3.0, "medium_max": 6.0},
	}


@pytest.fixture
def trained_recommender(tmp_path: Path, mock_config: dict) -> MLRecommender:
	"""Return an MLRecommender trained and persisted fully under tmp_path."""
	config = dict(mock_config)
	config["recommendations"] = dict(mock_config["recommendations"])
	config["recommendations"]["model_path"] = str((tmp_path / "recommendation_model.pkl").as_posix())
	config["recommendations"]["encoder_path"] = str((tmp_path / "recommendation_label_encoder.pkl").as_posix())
	recommender = MLRecommender(config)
	recommender.train()
	return recommender
