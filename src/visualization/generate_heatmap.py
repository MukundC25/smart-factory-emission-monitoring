"""Standalone script to generate pollution heatmap HTML output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import get_project_root, initialize_environment
from src.visualization.heatmap_data_prep import HeatmapDataPreparator
from src.visualization.heatmap_generator import HeatmapGenerator

LOGGER = logging.getLogger(__name__)


def _resolve_pollution_path(config: Dict[str, Any], root: Path) -> Path:
    """Resolve pollution input path with fallback candidates.

    Args:
        config: Runtime config dictionary.
        root: Project root path.

    Returns:
        Existing pollution file path.

    Raises:
        FileNotFoundError: If no suitable pollution file exists.
    """
    configured_processed = config.get("paths", {}).get("pollution_processed")
    configured_raw = config.get("paths", {}).get("pollution_raw")

    candidates = []
    if configured_processed:
        candidates.append(root / configured_processed)
    if configured_raw:
        candidates.append(root / configured_raw)
    candidates.append(root / "data/processed/pollution_clean.csv")
    candidates.append(root / "data/processed/ml_dataset.parquet")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No pollution dataset found. Checked configured and fallback paths: "
        + ", ".join(str(c) for c in candidates)
    )


def generate_pollution_heatmap(config: Dict[str, Any] | None = None) -> Tuple[Path, int, str]:
    """Generate pollution heatmap and save HTML output.

    Args:
        config: Optional runtime config override.

    Returns:
        Tuple of (output_path, points_count, intensity_column).
    """
    runtime_config = config or initialize_environment()
    root = get_project_root()

    preparator = HeatmapDataPreparator()
    pollution_path = _resolve_pollution_path(runtime_config, root)

    df = preparator.load_pollution_data(pollution_path)
    df = preparator.validate_coordinates(df)
    intensity_col = preparator.resolve_intensity_column(df)
    df = preparator.normalize_intensity(df, intensity_col)
    points = preparator.get_heatmap_points(df)
    city_center = preparator.get_city_center(df)
    LOGGER.info("Using map center lat=%.4f lon=%.4f", city_center[0], city_center[1])

    heatmap_cfg = runtime_config.get("heatmap", {})
    generator = HeatmapGenerator(heatmap_cfg)

    output_rel = runtime_config.get("paths", {}).get(
        "pollution_heatmap", "data/output/pollution_heatmap.html"
    )
    output_path = root / output_rel
    generator.build_full_map(df=df, intensity_col=intensity_col, output_path=output_path)

    return output_path, len(points), intensity_col


def main() -> None:
    """Run the heatmap generation pipeline and print summary output."""
    output_path, point_count, intensity_col = generate_pollution_heatmap()
    print("✅ Heatmap generated successfully")
    print(f"📍 Data points plotted: {point_count}")
    print(f"🎯 Intensity metric: {intensity_col}")
    print(f"🗺️  Output: {output_path}")
    print(f"🌐 Open in browser: file://{output_path.resolve()}")


if __name__ == "__main__":
    main()
