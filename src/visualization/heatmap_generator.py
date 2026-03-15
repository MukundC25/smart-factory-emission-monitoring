"""Folium-based heatmap generator for pollution data."""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import folium
import pandas as pd
import requests
from bs4 import BeautifulSoup
from folium import Element
from folium.plugins import Fullscreen, HeatMap

LOGGER = logging.getLogger(__name__)


def make_html_self_contained(html_path: Path) -> None:
    """Inline external JS/CSS assets into a saved Folium HTML document.

    This utility scans script/link tags and replaces external HTTP(S) URLs
    with inline script/style elements. Failed fetches are logged and left
    unchanged to avoid breaking output generation.

    Args:
        html_path: Path to generated HTML file.
    """
    if not html_path.exists():
        LOGGER.warning("HTML file not found for inlining: %s", html_path)
        return

    text = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")
    session = requests.Session()
    timeout_seconds = 20

    for script_tag in list(soup.find_all("script", src=True)):
        src = script_tag.get("src", "")
        if not isinstance(src, str) or not src.startswith(("http://", "https://")):
            continue
        try:
            response = session.get(src, timeout=timeout_seconds)
            response.raise_for_status()
            inline_tag = soup.new_tag("script")
            inline_tag.string = response.text
            script_tag.replace_with(inline_tag)
            LOGGER.info("Inlined script asset: %s", src)
        except Exception as exc:
            LOGGER.warning("Could not inline script asset %s: %s", src, exc)

    for link_tag in list(soup.find_all("link", href=True)):
        rel_values = link_tag.get("rel") or []
        href = link_tag.get("href", "")
        if "stylesheet" not in rel_values:
            continue
        if not isinstance(href, str) or not href.startswith(("http://", "https://")):
            continue
        try:
            response = session.get(href, timeout=timeout_seconds)
            response.raise_for_status()
            style_tag = soup.new_tag("style")
            style_tag.string = response.text
            link_tag.replace_with(style_tag)
            LOGGER.info("Inlined stylesheet asset: %s", href)
        except Exception as exc:
            LOGGER.warning("Could not inline stylesheet asset %s: %s", href, exc)

    html_path.write_text(str(soup), encoding="utf-8")


class HeatmapGenerator:
    """Generate production-ready pollution heatmaps with overlays."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize heatmap rendering configuration.

        Args:
            config: Heatmap config dictionary.
        """
        self.tile_provider: str = str(config.get("tile_provider", "CartoDB positron"))
        self.zoom_start: int = int(config.get("zoom_start", 6))
        self.heatmap_radius: int = int(config.get("radius", 25))
        self.heatmap_blur: int = int(config.get("blur", 15))
        self.heatmap_min_opacity: float = float(config.get("min_opacity", 0.4))
        self.heatmap_max_zoom: int = int(config.get("max_zoom", 18))
        gradient_cfg = config.get(
            "gradient",
            {
                "0.2": "#00FF00",
                "0.4": "#FFFF00",
                "0.6": "#FFA500",
                "0.8": "#FF0000",
                "1.0": "#800080",
            },
        )
        self.gradient: Dict[float, str] = {float(k): str(v) for k, v in gradient_cfg.items()}
        self.intensity_label: str = "N/A"
        self.timestamp_range_label: str = "N/A"

    def create_base_map(self, center: Tuple[float, float], zoom: int | None = None) -> folium.Map:
        """Create base map with fullscreen and layer controls.

        Args:
            center: Initial map center (lat, lon).
            zoom: Optional zoom override.

        Returns:
            Configured folium map.
        """
        fmap = folium.Map(
            location=[center[0], center[1]],
            zoom_start=zoom if zoom is not None else self.zoom_start,
            tiles=self.tile_provider,
            control_scale=True,
        )
        Fullscreen(position="topright", title="Fullscreen", title_cancel="Exit fullscreen").add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)
        return fmap

    def add_heatmap_layer(
        self,
        fmap: folium.Map,
        points: List[List[float]],
        intensity_col_name: str,
    ) -> folium.Map:
        """Add pollution heatmap layer to the map.

        Args:
            fmap: Folium map instance.
            points: Heatmap points [lat, lon, intensity].
            intensity_col_name: Intensity column used.

        Returns:
            Updated folium map.
        """
        layer_name = f"Pollution Heatmap ({intensity_col_name})"
        if points:
            HeatMap(
                data=points,
                name=layer_name,
                min_opacity=self.heatmap_min_opacity,
                max_zoom=self.heatmap_max_zoom,
                radius=self.heatmap_radius,
                blur=self.heatmap_blur,
                gradient=self.gradient,
                show=True,
            ).add_to(fmap)
        else:
            folium.FeatureGroup(name=layer_name, show=True).add_to(fmap)
        return fmap

    def add_pollution_station_markers(self, fmap: folium.Map, df: pd.DataFrame) -> folium.Map:
        """Add AQI-coded station circle markers.

        Args:
            fmap: Folium map instance.
            df: Pollution dataframe.

        Returns:
            Updated folium map.
        """
        station_layer = folium.FeatureGroup(name="Pollution Stations", show=True)
        lat_col = "station_lat" if "station_lat" in df.columns else "latitude"
        lon_col = "station_lon" if "station_lon" in df.columns else "longitude"

        for _, row in df.iterrows():
            lat = pd.to_numeric(row.get(lat_col), errors="coerce")
            lon = pd.to_numeric(row.get(lon_col), errors="coerce")
            if pd.isna(lat) or pd.isna(lon):
                continue

            aqi_value = pd.to_numeric(row.get("aqi_index"), errors="coerce")
            marker_color, aqi_band = self._aqi_style(float(aqi_value)) if pd.notna(aqi_value) else ("#808080", "Unknown")

            station_name = self._safe_text(row.get("station_name"))
            city = self._safe_text(row.get("city"))
            pm25 = self._safe_num(row.get("pm25"))
            pm10 = self._safe_num(row.get("pm10"))
            aqi = self._safe_num(row.get("aqi_index"))
            timestamp = self._safe_text(row.get("timestamp"))

            popup_html = (
                f"<b>{station_name}</b><br>"
                f"City: {city}<br>"
                f"PM2.5: {pm25}<br>"
                f"PM10: {pm10}<br>"
                f"AQI: {aqi}<br>"
                f"Band: {aqi_band}<br>"
                f"Timestamp: {timestamp}"
            )
            tooltip = f"{station_name} | AQI: {aqi}"

            folium.CircleMarker(
                location=[float(lat), float(lon)],
                radius=5,
                color=marker_color,
                weight=1,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                tooltip=tooltip,
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(station_layer)

        station_layer.add_to(fmap)
        return fmap

    def add_legend(self, fmap: folium.Map) -> folium.Map:
        """Inject AQI legend HTML into the map.

        Args:
            fmap: Folium map instance.

        Returns:
            Updated folium map.
        """
        legend_html = f"""
        <div style="
            position: fixed;
            bottom: 40px;
            right: 20px;
            z-index: 9999;
            background: white;
            border: 1px solid #888;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            line-height: 1.4;
        ">
            <div style="font-weight: 700; margin-bottom: 6px;">AQI Legend</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#00e400;margin-right:6px;"></span>Good (0-50)</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#ffff00;margin-right:6px;"></span>Satisfactory (51-100)</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#ff7e00;margin-right:6px;"></span>Moderate (101-200)</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#ff0000;margin-right:6px;"></span>Poor (201-300)</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#99004c;margin-right:6px;"></span>Very Poor (301-400)</div>
            <div><span style="display:inline-block;width:10px;height:10px;background:#7e0023;margin-right:6px;"></span>Severe (400+)</div>
            <hr style="margin: 8px 0;">
            <div><b>Intensity:</b> {html.escape(self.intensity_label)}</div>
            <div><b>Time range:</b> {html.escape(self.timestamp_range_label)}</div>
        </div>
        """
        fmap.get_root().html.add_child(Element(legend_html))
        return fmap

    def get_timestamp_range_label(self, df: pd.DataFrame) -> str:
        """Return a compact timestamp range label for legend display.

        Args:
            df: Pollution dataframe.

        Returns:
            Date range label or N/A.
        """
        return self._timestamp_range(df)

    def add_city_labels(self, fmap: folium.Map, df: pd.DataFrame) -> folium.Map:
        """Add labels for top 10 most polluted cities by average AQI.

        Args:
            fmap: Folium map instance.
            df: Pollution dataframe.

        Returns:
            Updated folium map.
        """
        if "city" not in df.columns or "aqi_index" not in df.columns:
            return fmap

        label_layer = folium.FeatureGroup(name="Top Polluted Cities", show=True)
        city_df = df.copy()
        lat_col = "station_lat" if "station_lat" in city_df.columns else "latitude"
        lon_col = "station_lon" if "station_lon" in city_df.columns else "longitude"
        city_df["aqi_index"] = pd.to_numeric(city_df["aqi_index"], errors="coerce")
        city_df[lat_col] = pd.to_numeric(city_df.get(lat_col), errors="coerce")
        city_df[lon_col] = pd.to_numeric(city_df.get(lon_col), errors="coerce")

        grouped = (
            city_df.dropna(subset=["city", "aqi_index", lat_col, lon_col])
            .groupby("city", as_index=False)
            .agg(avg_aqi=("aqi_index", "mean"), lat=(lat_col, "median"), lon=(lon_col, "median"))
            .sort_values("avg_aqi", ascending=False)
            .head(10)
        )

        for _, row in grouped.iterrows():
            label = f"{row['city']} | AQI {row['avg_aqi']:.1f}"
            folium.map.Marker(
                [float(row["lat"]), float(row["lon"])],
                icon=folium.DivIcon(
                    html=(
                        '<div style="font-size: 11px; font-weight: 600; color: #1f2937; '
                        'background: rgba(255,255,255,0.8); border: 1px solid #d1d5db; '
                        'padding: 2px 4px; border-radius: 4px; white-space: nowrap;">'
                        f"{html.escape(label)}"
                        "</div>"
                    )
                ),
            ).add_to(label_layer)

        label_layer.add_to(fmap)
        return fmap

    def build_full_map(self, df: pd.DataFrame, intensity_col: str, output_path: Path) -> Path:
        """Build and save the complete heatmap artifact.

        Args:
            df: Prepared pollution dataframe.
            intensity_col: Intensity source column.
            output_path: Destination HTML path.

        Returns:
            Path to saved HTML map.
        """
        center = self._resolve_center(df)
        points = self._build_points(df)

        self.intensity_label = intensity_col
        self.timestamp_range_label = self._timestamp_range(df)

        fmap = self.create_base_map(center=center, zoom=self.zoom_start)
        self.add_heatmap_layer(fmap, points, intensity_col)
        self.add_pollution_station_markers(fmap, df)
        self.add_city_labels(fmap, df)
        self.add_legend(fmap)

        title_html = """
        <div style="
            position: fixed;
            top: 16px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 9999;
            background: rgba(255,255,255,0.92);
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 16px;
            font-weight: 700;
            color: #111827;
        ">Smart Factory Pollution Heatmap</div>
        """
        source_note_html = """
        <div style="
            position: fixed;
            bottom: 18px;
            left: 18px;
            z-index: 9999;
            background: rgba(255,255,255,0.9);
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 11px;
            color: #374151;
        ">Data source: Smart Factory Emission Monitoring pipeline</div>
        """
        fmap.get_root().html.add_child(Element(title_html))
        fmap.get_root().html.add_child(Element(source_note_html))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(output_path))
        make_html_self_contained(output_path)
        LOGGER.info("Heatmap saved to %s — %d data points", output_path, len(points))
        return output_path

    def _resolve_center(self, df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty:
            return (20.5937, 78.9629)
        lat_col = "station_lat" if "station_lat" in df.columns else "latitude"
        lon_col = "station_lon" if "station_lon" in df.columns else "longitude"
        lat = pd.to_numeric(df.get(lat_col), errors="coerce").dropna()
        lon = pd.to_numeric(df.get(lon_col), errors="coerce").dropna()
        if lat.empty or lon.empty:
            return (20.5937, 78.9629)
        return (float(lat.median()), float(lon.median()))

    def _build_points(self, df: pd.DataFrame) -> List[List[float]]:
        if "intensity_normalized" not in df.columns or df.empty:
            return []
        lat_col = "station_lat" if "station_lat" in df.columns else "latitude"
        lon_col = "station_lon" if "station_lon" in df.columns else "longitude"
        points_df = pd.DataFrame(
            {
                "lat": pd.to_numeric(df.get(lat_col), errors="coerce"),
                "lon": pd.to_numeric(df.get(lon_col), errors="coerce"),
                "intensity": pd.to_numeric(df.get("intensity_normalized"), errors="coerce"),
            }
        ).dropna(subset=["lat", "lon", "intensity"])
        return points_df[["lat", "lon", "intensity"]].astype(float).values.tolist()

    def _timestamp_range(self, df: pd.DataFrame) -> str:
        if "timestamp" not in df.columns or df.empty:
            return "N/A"
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()
        if ts.empty:
            return "N/A"
        return f"{ts.min().date().isoformat()} to {ts.max().date().isoformat()}"

    def _aqi_style(self, aqi: float) -> Tuple[str, str]:
        if aqi <= 50:
            return ("#00e400", "Good")
        if aqi <= 100:
            return ("#ffff00", "Satisfactory")
        if aqi <= 200:
            return ("#ff7e00", "Moderate")
        if aqi <= 300:
            return ("#ff0000", "Poor")
        if aqi <= 400:
            return ("#99004c", "Very Poor")
        return ("#7e0023", "Severe")

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return "N/A"
        text = str(value).strip()
        return text if text else "N/A"

    def _safe_num(self, value: Any) -> str:
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return "N/A"
        return f"{float(numeric):.2f}"
