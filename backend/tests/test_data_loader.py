"""Unit tests for DataLoader behavior and cache semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.dependencies import get_data_loader
from backend.utils import data_loader as loader_module
from backend.utils.data_loader import DataLoader


@pytest.fixture
def tmp_settings(tmp_path: Path):  # type: ignore[no-untyped-def]
    class _S:
        FACTORIES_CSV = tmp_path / "factories.csv"
        RAW_POLLUTION_CSV = tmp_path / "pollution_raw.csv"
        PROCESSED_POLLUTION_FILE = tmp_path / "pollution_clean.csv"
        POLLUTION_CSV = None
        RECOMMENDATIONS_CSV = tmp_path / "recommendations.csv"
        CACHE_TTL_SECONDS = 3600

    return _S()


def test_load_factories_returns_dataframe(tmp_settings) -> None:  # type: ignore[no-untyped-def]
    pd.DataFrame({"factory_id": ["FAC1"]}).to_csv(tmp_settings.FACTORIES_CSV, index=False)
    loader = DataLoader(settings=tmp_settings)
    df = loader.load_factories()
    assert isinstance(df, pd.DataFrame)


def test_load_factories_empty_file_returns_empty_df_not_crash(tmp_settings) -> None:  # type: ignore[no-untyped-def]
    pd.DataFrame(columns=["factory_id"]).to_csv(tmp_settings.FACTORIES_CSV, index=False)
    loader = DataLoader(settings=tmp_settings)
    assert loader.load_factories().empty


def test_load_pollution_returns_dataframe(tmp_settings) -> None:  # type: ignore[no-untyped-def]
    pd.DataFrame({"city": ["Pune"], "pm25": [10.0]}).to_csv(tmp_settings.RAW_POLLUTION_CSV, index=False)
    loader = DataLoader(settings=tmp_settings)
    df = loader.load_pollution()
    assert isinstance(df, pd.DataFrame)


def test_load_pollution_missing_file_returns_empty_df_not_crash(tmp_settings) -> None:  # type: ignore[no-untyped-def]
    loader = DataLoader(settings=tmp_settings)
    assert loader.load_pollution().empty


def test_load_recommendation_reports_returns_list(tmp_settings, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(loader_module, "_CONFIG_CACHE", {"recommendations": {"output_json": "recommendations.json"}})
    monkeypatch.setattr(loader_module, "_CONFIG_PATH", tmp_settings.RECOMMENDATIONS_CSV.parent / "config.yaml")
    payload = {
        "count": 1,
        "reports": [{"factory_id": "FAC1", "generated_at": "2026-01-01T00:00:00+00:00"}],
    }
    (tmp_settings.RECOMMENDATIONS_CSV.parent / "recommendations.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    loader = DataLoader(settings=tmp_settings)
    reports = loader.load_recommendation_reports()
    assert isinstance(reports, list)


def test_load_recommendation_reports_missing_file_returns_empty_list(tmp_settings, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(loader_module, "_CONFIG_CACHE", {"recommendations": {"output_json": "does_not_exist.json"}})
    monkeypatch.setattr(loader_module, "_CONFIG_PATH", tmp_settings.RECOMMENDATIONS_CSV.parent / "config.yaml")
    loader = DataLoader(settings=tmp_settings)
    assert loader.load_recommendation_reports() == []


def test_dataloader_cache_returns_same_instance() -> None:
    first = get_data_loader()
    second = get_data_loader()
    assert first is second


def test_dataset_info_returns_counts_dict(tmp_settings) -> None:  # type: ignore[no-untyped-def]
    pd.DataFrame({"factory_id": ["FAC1"]}).to_csv(tmp_settings.FACTORIES_CSV, index=False)
    pd.DataFrame({"city": ["Pune"]}).to_csv(tmp_settings.RAW_POLLUTION_CSV, index=False)
    pd.DataFrame({"factory_id": ["FAC1"]}).to_csv(tmp_settings.RECOMMENDATIONS_CSV, index=False)
    loader = DataLoader(settings=tmp_settings)
    info = loader.dataset_info()
    assert set(info.keys()) == {"factories", "pollution", "recommendations"}
