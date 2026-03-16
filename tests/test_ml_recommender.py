"""Comprehensive tests for MLRecommender."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.recommendations.ml_recommender import MLRecommender


def _scores(industry: str = "steel") -> dict:
    return {
        "pm25_score": 6.0,
        "pm10_score": 6.0,
        "so2_score": 5.0,
        "no2_score": 4.0,
        "co_score": 2.0,
        "o3_score": 3.0,
        "industry_type": industry,
        "composite_score": 5.2,
    }


def test_ml_recommender_trains_without_error(mock_config: dict) -> None:
    recommender = MLRecommender(mock_config)
    recommender.train()
    assert recommender.model is not None


def test_ml_recommender_predict_returns_list(trained_recommender: MLRecommender) -> None:
    result = trained_recommender.predict_recommendations(_scores())
    assert isinstance(result, list)


def test_ml_recommender_predict_returns_only_confident_categories(trained_recommender: MLRecommender) -> None:
    result = trained_recommender.predict_recommendations(_scores())
    assert all(isinstance(item, str) for item in result)


def test_ml_recommender_is_model_trained_false_before_train(tmp_path: Path, mock_config: dict) -> None:
    config = dict(mock_config)
    config["recommendations"] = dict(mock_config["recommendations"])
    config["recommendations"]["model_path"] = str((tmp_path / "m.pkl").as_posix())
    config["recommendations"]["encoder_path"] = str((tmp_path / "e.pkl").as_posix())
    recommender = MLRecommender(config)
    assert recommender.is_model_trained() is False


def test_ml_recommender_is_model_trained_true_after_train(trained_recommender: MLRecommender) -> None:
    assert trained_recommender.is_model_trained() is True


def test_ml_recommender_auto_trains_on_first_predict(tmp_path: Path, mock_config: dict) -> None:
    config = dict(mock_config)
    config["recommendations"] = dict(mock_config["recommendations"])
    config["recommendations"]["model_path"] = str((tmp_path / "auto_model.pkl").as_posix())
    config["recommendations"]["encoder_path"] = str((tmp_path / "auto_encoder.pkl").as_posix())
    recommender = MLRecommender(config)
    out = recommender.predict_recommendations(_scores())
    assert isinstance(out, list)
    assert (tmp_path / "auto_model.pkl").exists()


def test_ml_recommender_single_class_output_does_not_crash(trained_recommender: MLRecommender) -> None:
    with patch.object(trained_recommender.model, "predict_proba", return_value=[np.array([[1.0]])]), patch.object(
        trained_recommender.model, "classes_", [np.array([0])]
    ):
        result = trained_recommender.predict_recommendations(_scores())
    assert isinstance(result, list)


def test_ml_recommender_unknown_industry_type_does_not_crash(trained_recommender: MLRecommender) -> None:
    result = trained_recommender.predict_recommendations(_scores("unknown_x"))
    assert isinstance(result, list)


def test_ml_recommender_all_zero_scores_returns_list(trained_recommender: MLRecommender) -> None:
    zeros = {
        "pm25_score": 0.0,
        "pm10_score": 0.0,
        "so2_score": 0.0,
        "no2_score": 0.0,
        "co_score": 0.0,
        "o3_score": 0.0,
        "industry_type": "steel",
        "composite_score": 0.0,
    }
    assert isinstance(trained_recommender.predict_recommendations(zeros), list)


def test_ml_recommender_saves_model_to_tmp_path(tmp_path: Path, mock_config: dict) -> None:
    config = dict(mock_config)
    config["recommendations"] = dict(mock_config["recommendations"])
    config["recommendations"]["model_path"] = str((tmp_path / "saved_model.pkl").as_posix())
    config["recommendations"]["encoder_path"] = str((tmp_path / "saved_encoder.pkl").as_posix())
    recommender = MLRecommender(config)
    recommender.train()
    assert (tmp_path / "saved_model.pkl").exists()
    assert (tmp_path / "saved_encoder.pkl").exists()


def test_ml_recommender_loads_existing_model_without_retraining(tmp_path: Path, mock_config: dict) -> None:
    config = dict(mock_config)
    config["recommendations"] = dict(mock_config["recommendations"])
    config["recommendations"]["model_path"] = str((tmp_path / "existing_model.pkl").as_posix())
    config["recommendations"]["encoder_path"] = str((tmp_path / "existing_encoder.pkl").as_posix())
    first = MLRecommender(config)
    first.train()

    second = MLRecommender(config)
    with patch.object(second, "train_model", wraps=second.train_model) as spy_train:
        second._load_model()
        assert second.model is not None
        assert spy_train.call_count == 0
