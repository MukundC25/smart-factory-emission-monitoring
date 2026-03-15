"""
MLRecommender: Machine learning-based recommendation engine for pollution control actions.

- Uses scikit-learn RandomForestClassifier (multi-label)
- Auto-generates synthetic data if real samples < 100
- Saves model and label encoder to models/
- Confidence threshold from config
- Auto-trains if model files missing
- Type hints and Google-style docstrings throughout
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

LOGGER = logging.getLogger(__name__)

class MLRecommender:
    """ML-based recommender for pollution control actions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MLRecommender with config.

        Args:
            config: Runtime configuration dict.
        """
        self.config = config or {}
        self.model_path = Path(self.config.get("recommendations", {}).get("model_path", "models/recommendation_model.pkl"))
        self.encoder_path = Path(self.config.get("recommendations", {}).get("encoder_path", "models/recommendation_label_encoder.pkl"))
        self.confidence_threshold = float(self.config.get("recommendations", {}).get("confidence_threshold", 0.4))
        self.model: Optional[RandomForestClassifier] = None
        self.label_binarizer: Optional[MultiLabelBinarizer] = None
        self.industry_encoder: Optional[LabelEncoder] = None

    def is_model_trained(self) -> bool:
        """Check if model and encoder files exist."""
        return self.model_path.exists() and self.encoder_path.exists()

    def _load_model(self) -> None:
        """Load model and encoder from disk."""
        if not self.is_model_trained():
            self.train_model()
        self.model = joblib.load(self.model_path)
        encoder_payload = joblib.load(self.encoder_path)
        if isinstance(encoder_payload, dict):
            self.label_binarizer = encoder_payload["label_binarizer"]
            self.industry_encoder = encoder_payload["industry_encoder"]
            return

        LOGGER.warning("Legacy encoder artifact detected; retraining model artifacts")
        self.train_model()

    def train_model(self) -> None:
        """Train the ML recommender, using real or synthetic data as needed."""
        # Load processed dataset
        dataset_path = Path(self.config.get("paths", {}).get("processed_dataset", "data/processed/ml_dataset.parquet"))
        if dataset_path.exists():
            if dataset_path.suffix == ".parquet":
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_csv(dataset_path)
        else:
            LOGGER.warning("Processed dataset not found at %s. Generating synthetic training data.", dataset_path)
            df = pd.DataFrame()
        # Features
        features = ["pm25_score", "pm10_score", "so2_score", "no2_score", "co_score", "industry_type", "composite_score"]
        # Target: recommendation_category (multi-label, pipe-separated)
        missing_features = [feature for feature in features if feature not in df.columns]
        if "recommendation_category" not in df.columns or len(df) < 100 or missing_features:
            LOGGER.warning("Insufficient real data (<100 samples) or missing target. Generating synthetic training data.")
            df = self.generate_synthetic_training_data(n_samples=200)
        X = df[features].copy()
        # Encode industry_type
        industry_encoder = LabelEncoder()
        X["industry_type_encoded"] = industry_encoder.fit_transform(X["industry_type"].astype(str))
        X = X.drop(columns=["industry_type"])
        # Multi-label target
        y_raw = df["recommendation_category"].fillna("").apply(lambda x: x.split("|") if x else [])
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y_raw)
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None)
        LOGGER.info("MLRecommender trained. Accuracy: %.3f, F1 per label: %s", acc, f1)
        # Save model and encoder
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(
            {"label_binarizer": mlb, "industry_encoder": industry_encoder},
            self.encoder_path,
        )
        self.model = model
        self.label_binarizer = mlb
        self.industry_encoder = industry_encoder

    def train(self) -> None:
        """Backward-compatible train entrypoint for tests and callers."""
        self.train_model()

    def generate_synthetic_training_data(self, n_samples: int = 200) -> pd.DataFrame:
        """Generate synthetic training data for ML recommender.

        Args:
            n_samples: Number of samples to generate.
        Returns:
            pd.DataFrame: Synthetic dataset.
        """
        np.random.seed(42)
        industry_types = ["steel", "chemical", "power", "cement", "textile", "pharmaceutical", "industrial", "works"]
        rec_categories = [
            "SO2_Control", "PM_Control", "NOx_Control", "CO_Control", "O3_Control",
            "Audit", "Monitoring", "Compliance", "Maintenance", "Training"
        ]
        data = []
        for _ in range(n_samples):
            row = {
                "pm25_score": np.random.uniform(0, 10),
                "pm10_score": np.random.uniform(0, 10),
                "so2_score": np.random.uniform(0, 10),
                "no2_score": np.random.uniform(0, 10),
                "co_score": np.random.uniform(0, 10),
                "industry_type": np.random.choice(industry_types),
                "composite_score": np.random.uniform(0, 10),
            }
            # Assign 1-3 random categories based on scores
            cats = []
            if row["so2_score"] > 6: cats.append("SO2_Control")
            if row["pm25_score"] > 6 or row["pm10_score"] > 6: cats.append("PM_Control")
            if row["no2_score"] > 6: cats.append("NOx_Control")
            if row["co_score"] > 6: cats.append("CO_Control")
            if row["composite_score"] > 6: cats.append("Audit")
            if not cats: cats.append(np.random.choice(rec_categories))
            row["recommendation_category"] = "|".join(set(cats))
            data.append(row)
        return pd.DataFrame(data)

    def predict_recommendations(self, risk_scores: Dict[str, Any]) -> List[str]:
        """Predict recommended action categories for a factory.

        Args:
            risk_scores: Output from PollutionRiskScorer.compute_factory_risk
        Returns:
            List[str]: Recommended categories (filtered by confidence)
        """
        if self.model is None or self.label_binarizer is None or self.industry_encoder is None:
            self._load_model()
        # Prepare input
        X = pd.DataFrame([{k: risk_scores.get(k, 0) for k in [
            "pm25_score", "pm10_score", "so2_score", "no2_score", "co_score", "composite_score"]}])
        # Encode industry_type
        industry_type = str(risk_scores.get("industry_type", "industrial"))
        known_classes = set(self.industry_encoder.classes_.tolist())
        if industry_type in known_classes:
            X["industry_type_encoded"] = self.industry_encoder.transform([industry_type])[0]
        else:
            LOGGER.debug("Unknown industry_type '%s' at inference; using fallback class", industry_type)
            X["industry_type_encoded"] = self.industry_encoder.transform([self.industry_encoder.classes_[0]])[0]

        ordered_features = [
            "pm25_score",
            "pm10_score",
            "so2_score",
            "no2_score",
            "co_score",
            "composite_score",
            "industry_type_encoded",
        ]
        X = X[ordered_features]
        # Predict probabilities
        probs = self.model.predict_proba(X)
        cats = []
        model_classes = getattr(self.model, "classes_", None)
        for idx, class_proba in enumerate(probs):
            if isinstance(model_classes, list):
                output_classes = np.asarray(model_classes[idx])
            else:
                output_classes = np.asarray(model_classes) if model_classes is not None else None
            if output_classes is None:
                continue
            if output_classes.shape[0] == 1:
                positive_prob = float(class_proba[0][0]) if int(output_classes[0]) == 1 else 0.0
            else:
                positive_indices = np.where(output_classes == 1)[0]
                if positive_indices.size == 0:
                    continue
                positive_prob = float(class_proba[0][int(positive_indices[0])])
            if positive_prob > self.confidence_threshold:
                cats.append(str(self.label_binarizer.classes_[idx]))
        return cats
