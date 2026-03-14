"""Generate interactive Folium dashboard for factory emissions."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from src.common import get_project_root, initialize_environment

LOGGER = logging.getLogger(__name__)

RISK_COLORS = {"High": "red", "Medium": "orange", "Low": "green"}


def _load_inputs(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load recommendations and pollution datasets.

    Args:
        config: Runtime configuration.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Recommendation and pollution dataframes.
    """
    root = get_project_root()
    recommendation_path = root / config["paths"]["recommendations"]
    pollution_path = root / config["paths"]["pollution_raw"]
    recommendations = pd.read_csv(recommendation_path)
    pollution = pd.read_csv(pollution_path)
    return recommendations, pollution


def build_dashboard(config: Optional[Dict[str, Any]] = None) -> str:
    """Build a standalone HTML dashboard with map overlays.

    Args:
        config: Optional runtime configuration.

    Returns:
        str: Path to generated dashboard HTML file.
    """
    runtime_config = config or initialize_environment()
    recommendations, pollution = _load_inputs(runtime_config)

    center_lat = float(recommendations["latitude"].mean()) if not recommendations.empty else 20.5937
    center_lon = float(recommendations["longitude"].mean()) if not recommendations.empty else 78.9629

    dashboard_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, control_scale=True)

    factory_layer = folium.FeatureGroup(name="Factories", show=True)
    station_layer = folium.FeatureGroup(name="Pollution Stations", show=True)

    factory_cluster = MarkerCluster(name="Factory Clusters")
    station_cluster = MarkerCluster(name="Station Clusters")

    for _, row in recommendations.iterrows():
        risk = row.get("risk_level", "Low")
        color = RISK_COLORS.get(risk, "blue")
        popup = folium.Popup(
            html=(
                f"<b>{row['factory_name']}</b><br>"
                f"Industry: {row['industry_type']}<br>"
                f"Risk Score: {row['pollution_impact_score']:.2f}<br>"
                f"Risk Level: {risk}<br>"
                f"Recommendation: {row['recommendation']}"
            ),
            max_width=350,
        )
        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        )
        marker.add_to(factory_cluster)

        influence_radius = 5000 if risk == "Low" else 10000 if risk == "Medium" else 15000
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=influence_radius,
            color=color,
            weight=1,
            fill=False,
        ).add_to(factory_layer)

    for _, station in pollution.drop_duplicates(subset=["station_name", "station_lat", "station_lon"]).iterrows():
        popup = folium.Popup(
            html=(
                f"<b>{station['station_name']}</b><br>"
                f"PM2.5: {station.get('pm25', 'NA')}<br>"
                f"PM10: {station.get('pm10', 'NA')}<br>"
                f"City: {station.get('city', 'NA')}"
            ),
            max_width=260,
        )
        folium.Marker(
            location=[station["station_lat"], station["station_lon"]],
            popup=popup,
            icon=folium.Icon(color="blue", icon="cloud"),
        ).add_to(station_cluster)

    heat_points = pollution[["station_lat", "station_lon", "pm25"]].dropna().values.tolist()
    if heat_points:
        HeatMap(heat_points, name="PM2.5 Heatmap", radius=18, blur=14, min_opacity=0.3).add_to(
            dashboard_map
        )

    factory_cluster.add_to(factory_layer)
    station_cluster.add_to(station_layer)
    factory_layer.add_to(dashboard_map)
    station_layer.add_to(dashboard_map)
    folium.LayerControl(collapsed=False).add_to(dashboard_map)

    output_path = get_project_root() / runtime_config["paths"]["dashboard"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_map.save(str(output_path))
    LOGGER.info("Dashboard written to %s", output_path)
    return str(output_path)


def main() -> None:
    """Run dashboard builder standalone."""
    config = initialize_environment()
    build_dashboard(config)


if __name__ == "__main__":
    main()
