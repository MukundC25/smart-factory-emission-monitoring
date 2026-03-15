from __future__ import annotations

from csv import DictReader
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
RECOMMENDATIONS_FILE = ROOT_DIR / "data" / "output" / "recommendations.csv"
FACTORIES_FILE = ROOT_DIR / "data" / "raw" / "factories" / "factories.csv"
POLLUTION_FILE = ROOT_DIR / "data" / "raw" / "pollution" / "pollution_readings.csv"


def _parse_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _normalize_score(value: Any) -> int:
	score = _parse_float(value, 55.0)
	if score <= 10:
		return round(score * 10)
	return round(score)


def _read_csv(path: Path) -> list[dict[str, str]]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8", newline="") as handle:
		return list(DictReader(handle))


def load_recommendations(city: str | None = None, limit: int = 250) -> list[dict[str, Any]]:
	rows = _read_csv(RECOMMENDATIONS_FILE)
	items: list[dict[str, Any]] = []
	for row in rows:
		row_city = (row.get("city") or "").strip()
		if city and row_city.lower() != city.lower():
			continue

		item = {
			"factory_id": row.get("factory_id", ""),
			"factory_name": row.get("factory_name", "Unknown Factory"),
			"industry_type": row.get("industry_type", "industrial"),
			"city": row_city,
			"state": row.get("state", ""),
			"country": row.get("country", "India"),
			"latitude": _parse_float(row.get("latitude")),
			"longitude": _parse_float(row.get("longitude")),
			"pollution_score": _normalize_score(row.get("pollution_impact_score")),
			"risk_level": row.get("risk_level") or "Moderate",
			"primary_pollutant": "PM2.5",
			"latest_pm25": _parse_float(row.get("latest_pm25"), 0.0),
			"latest_pm10": _parse_float(row.get("latest_pm10"), 0.0),
			"recommendation": row.get("recommendation")
			or "Install scrubber systems and strengthen continuous monitoring.",
		}
		items.append(item)
		if len(items) >= limit:
			break

	return items


def load_factories(city: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
	rows = _read_csv(FACTORIES_FILE)
	items: list[dict[str, Any]] = []
	for row in rows:
		row_city = (row.get("city") or "").strip()
		if city and row_city.lower() != city.lower():
			continue
		items.append(
			{
				"factory_id": row.get("factory_id", ""),
				"factory_name": row.get("factory_name", "Unknown Factory"),
				"industry_type": row.get("industry_type", "industrial"),
				"city": row_city,
				"state": row.get("state", ""),
				"country": row.get("country", "India"),
				"source": row.get("source", ""),
				"latitude": _parse_float(row.get("latitude")),
				"longitude": _parse_float(row.get("longitude")),
			}
		)
		if len(items) >= limit:
			break
	return items


def load_factory_catalog(city: str | None = None, limit: int = 350) -> list[dict[str, Any]]:
	recommendations = load_recommendations(city=city, limit=limit)
	if recommendations:
		return recommendations

	factories = load_factories(city=city, limit=limit)
	fallback: list[dict[str, Any]] = []
	for index, item in enumerate(factories):
		score = 30 + ((index * 17) % 65)
		fallback.append(
			{
				**item,
				"pollution_score": score,
				"risk_level": "High" if score >= 70 else "Moderate" if score >= 40 else "Low",
				"primary_pollutant": "PM2.5" if score >= 70 else "NO2" if score >= 40 else "CO",
				"latest_pm25": round(18 + score * 1.2, 1),
				"latest_pm10": round(32 + score * 1.5, 1),
				"recommendation": "Install scrubber systems and deploy continuous emissions monitoring.",
			}
		)
	return fallback


def load_pollution_readings(city: str | None = None, limit: int = 350) -> list[dict[str, Any]]:
	rows = _read_csv(POLLUTION_FILE)
	items: list[dict[str, Any]] = []
	for row in rows:
		row_city = (row.get("city") or "").strip()
		if city and row_city.lower() != city.lower():
			continue
		items.append(
			{
				"timestamp": row.get("timestamp", ""),
				"station_name": row.get("station_name", ""),
				"city": row_city,
				"country": row.get("country", "India"),
				"pm25": _parse_float(row.get("pm25")),
				"pm10": _parse_float(row.get("pm10")),
				"co": _parse_float(row.get("co")),
				"no2": _parse_float(row.get("no2")),
				"so2": _parse_float(row.get("so2")),
				"o3": _parse_float(row.get("o3")),
				"aqi_index": _parse_float(row.get("aqi_index")),
				"station_lat": _parse_float(row.get("station_lat")),
				"station_lon": _parse_float(row.get("station_lon")),
			}
		)
		if len(items) >= limit:
			break
	return items


def get_recommendation(factory_id: str) -> dict[str, Any] | None:
	for item in load_factory_catalog(limit=2000):
		if item.get("factory_id") == factory_id:
			return item
	return None
