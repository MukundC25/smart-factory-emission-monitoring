"""Shared utilities for the smart factory emission monitoring pipeline."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path: Absolute path to the project root.
    """
    return PROJECT_ROOT


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load application configuration from YAML.

    Args:
        config_path: Optional path to config file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    target = config_path or (PROJECT_ROOT / "config.yaml")
    with target.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    """Configure root logger with file and console handlers.

    Args:
        log_file: Path where logs should be written.
        level: Logging level name.

    Returns:
        logging.Logger: Configured root logger.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def safe_request_json(
    method: str,
    url: str,
    timeout: int,
    max_retries: int,
    backoff_base_seconds: float,
    rate_limit_seconds: float,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Execute HTTP request with retry, backoff, and rate limiting.

    Args:
        method: HTTP method.
        url: Target URL.
        timeout: Request timeout in seconds.
        max_retries: Max retry attempts.
        backoff_base_seconds: Base duration for exponential backoff.
        rate_limit_seconds: Delay between requests.
        **kwargs: Additional request arguments.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON response when available.
    """
    logger = logging.getLogger(__name__)
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code >= 500:
                raise requests.HTTPError(
                    f"Server error status code: {response.status_code}",
                    response=response,
                )
            if response.status_code == 429:
                raise requests.HTTPError("Rate limit exceeded", response=response)
            if 400 <= response.status_code < 500:
                logger.error(
                    "Client error status code %s for %s; skipping retries",
                    response.status_code,
                    url,
                )
                return None
            response.raise_for_status()
            time.sleep(rate_limit_seconds)
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type or response.text.strip().startswith("{"):
                return response.json()
            return json.loads(response.text)
        except (requests.RequestException, json.JSONDecodeError) as error:
            wait_seconds = backoff_base_seconds ** attempt
            logger.warning(
                "Request failed (%s/%s) for %s: %s. Retrying in %.2fs",
                attempt,
                max_retries,
                url,
                error,
                wait_seconds,
            )
            time.sleep(wait_seconds)
    logger.error("Request failed permanently for URL: %s", url)
    return None


def haversine_distance_km(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Calculate Haversine distance in kilometers between two coordinates.

    Args:
        latitude_1: First latitude.
        longitude_1: First longitude.
        latitude_2: Second latitude.
        longitude_2: Second longitude.

    Returns:
        float: Distance in kilometers.
    """
    earth_radius_km = 6371.0
    lat1_rad = math.radians(latitude_1)
    lon1_rad = math.radians(longitude_1)
    lat2_rad = math.radians(latitude_2)
    lon2_rad = math.radians(longitude_2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    arc = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * earth_radius_km * math.atan2(math.sqrt(arc), math.sqrt(1 - arc))


def initialize_environment() -> Dict[str, Any]:
    """Load environment variables and configuration.

    Returns:
        Dict[str, Any]: Runtime configuration.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.example", override=False)
    config = load_config()
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logging(PROJECT_ROOT / config["paths"]["log_file"], log_level)
    logger.debug("Environment initialized")
    return config
