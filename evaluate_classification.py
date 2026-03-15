"""Classification evaluation for pollution impact model with detailed metrics."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.common import get_project_root, initialize_environment

logger = logging.getLogger(__name__)

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class PollutionImpactClassifier:
    """Classification wrapper for regression-based pollution impact model."""

    def __init__(self, model_path: Path, config_path: Path):
        """Initialize classifier with trained model.

        Args:
            model_path: Path to trained model
            config_path: Path to model configuration
        """
        self.model = joblib.load(model_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Risk thresholds from config
        # Prefer top-level 'risk_bands' (e.g. from config.yaml), but fall back
        # to nested 'config.risk_bands' if present, and finally to defaults.
        risk_bands = self.config.get('risk_bands')
        if risk_bands is None:
            risk_bands = self.config.get('config', {}).get('risk_bands', {})
        self.low_max = risk_bands.get('low_max', 3.0)
        self.medium_max = risk_bands.get('medium_max', 6.0)

        self.class_names = ['Low', 'Medium', 'High']
        self.class_labels = [0, 1, 2]

    def score_to_class(self, score: float) -> int:
        """Convert regression score to class label.

        Args:
            score: Pollution impact score (0-10)

        Returns:
            int: Class label (0=Low, 1=Medium, 2=High)
        """
        if score <= self.low_max:
            return 0  # Low
        elif score <= self.medium_max:
            return 1  # Medium
        else:
            return 2  # High

    def predict_classes(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class labels from features.

        Args:
            features: Input features

        Returns:
            np.ndarray: Predicted class labels
        """
        scores = self.model.predict(features)
        return np.array([self.score_to_class(score) for score in scores])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities (simplified from regression scores).

        Args:
            features: Input features

        Returns:
            np.ndarray: Class probabilities [n_samples, n_classes]
        """
        scores = self.model.predict(features)
        probabilities = []

        for score in scores:
            if score <= self.low_max:
                # Low risk
                prob_low = 1.0 - (score / self.low_max) * 0.3
                prob_med = (score / self.low_max) * 0.3
                prob_high = 0.0
            elif score <= self.medium_max:
                # Medium risk
                prob_low = 0.0
                prob_med = 1.0 - ((score - self.low_max) / (self.medium_max - self.low_max)) * 0.4
                prob_high = ((score - self.low_max) / (self.medium_max - self.low_max)) * 0.4
            else:
                # High risk
                prob_low = 0.0
                prob_med = 1.0 - ((score - self.medium_max) / (10.0 - self.medium_max)) * 0.3
                prob_high = ((score - self.medium_max) / (10.0 - self.medium_max)) * 0.3

            # Normalize probabilities
            total = prob_low + prob_med + prob_high
            probabilities.append([prob_low/total, prob_med/total, prob_high/total])

        return np.array(probabilities)


def load_test_data(config: Dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Load test dataset for evaluation.

    Args:
        config: Runtime configuration

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and true labels
    """
    root = get_project_root()
    dataset_path = root / config["paths"]["processed_dataset"]

    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    dataset = pd.read_parquet(dataset_path)
    test_data = dataset[dataset["split"] == "test"].copy()

    target_col = config["ml"]["target_column"]
    feature_cols = [
        col for col in test_data.columns
        if col not in [target_col, "split", "timestamp", "last_updated"]
    ]

    X_test = test_data[feature_cols]
    y_true_scores = test_data[target_col]

    return X_test, y_true_scores


def evaluate_classification_performance(
    classifier: PollutionImpactClassifier,
    X_test: pd.DataFrame,
    y_true_scores: pd.Series
) -> Dict:
    """Evaluate classification performance with comprehensive metrics.

    Args:
        classifier: Trained classifier
        X_test: Test features
        y_true_scores: True pollution impact scores

    Returns:
        Dict: Comprehensive evaluation results
    """
    # Convert true scores to class labels
    y_true_classes = np.array([classifier.score_to_class(score) for score in y_true_scores])

    # Get unique classes present in the data
    unique_classes = np.unique(y_true_classes)
    class_names_present = [classifier.class_names[i] for i in unique_classes]
    class_labels_present = unique_classes.tolist()

    # Get predictions
    y_pred_classes = classifier.predict_classes(X_test)
    y_pred_proba = classifier.predict_proba(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)

    # Detailed classification report (only for present classes)
    report = classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=class_names_present,
        labels=class_labels_present,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes, labels=class_labels_present)

    # Per-class metrics (only for present classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average=None, labels=class_labels_present
    )

    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='macro', labels=class_labels_present
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted', labels=class_labels_present
    )

    # Class distribution (only for present classes)
    class_distribution = pd.Series(y_true_classes).value_counts().sort_index()
    class_distribution.index = [classifier.class_names[i] for i in class_distribution.index]

    # Prediction confidence analysis
    max_probs = np.max(y_pred_proba, axis=1)
    confidence_stats = {
        'mean': float(np.mean(max_probs)),
        'std': float(np.std(max_probs)),
        'min': float(np.min(max_probs)),
        'max': float(np.max(max_probs)),
        'median': float(np.median(max_probs))
    }

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'support': support.tolist()
        },
        'macro_averages': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'weighted_averages': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1
        },
        'class_distribution': class_distribution.to_dict(),
        'prediction_confidence': confidence_stats,
        'thresholds': {
            'low_max': classifier.low_max,
            'medium_max': classifier.medium_max
        },
        'classes_present': class_names_present,
        'n_classes': len(class_names_present)
    }


def plot_evaluation_results(results: Dict, output_dir: Path):
    """Create comprehensive evaluation plots.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the classes that are actually present
    classes_present = results['classes_present']
    n_classes = results['n_classes']

    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    if n_classes == 2:
        # 2-class layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    else:
        # 3-class layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes_present,
        yticklabels=classes_present,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # Per-class Metrics
    metrics_df = pd.DataFrame({
        'Precision': results['per_class_metrics']['precision'],
        'Recall': results['per_class_metrics']['recall'],
        'F1-Score': results['per_class_metrics']['f1_score']
    }, index=classes_present)

    metrics_df.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # Class Distribution
    class_dist = results['class_distribution']
    colors = ['green', 'orange', 'red'][:n_classes]  # Adjust colors based on number of classes
    axes[1, 0].bar(class_dist.keys(), class_dist.values(), color=colors)
    axes[1, 0].set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].set_xlabel('Risk Level')

    # Add value labels on bars
    for i, v in enumerate(class_dist.values()):
        axes[1, 0].text(i, v + 0.5, str(v), ha='center', va='bottom')

    # Overall Metrics Summary
    overall_metrics = {
        'Accuracy': results['accuracy'],
        'Macro F1': results['macro_averages']['f1_score'],
        'Weighted F1': results['weighted_averages']['f1_score']
    }

    axes[1, 1].bar(overall_metrics.keys(), overall_metrics.values(),
                   color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1, 1].set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])

    # Add value labels
    for i, v in enumerate(overall_metrics.values()):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / 'classification_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detailed Classification Report Table
    fig, ax = plt.subplots(figsize=(12, 4))

    # Create table data
    table_data = []
    for i, class_name in enumerate(classes_present):
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            table_data.append([
                class_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{metrics['support']:.0f}"
            ])

    # Add macro and weighted averages
    table_data.extend([
        ['Macro Avg', '', '', f"{results['macro_averages']['f1_score']:.3f}", ''],
        ['Weighted Avg', '', '', f"{results['weighted_averages']['f1_score']:.3f}", '']
    ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
        loc='center',
        cellLoc='center',
        colColours=['lightblue'] * 5
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    ax.axis('off')
    ax.set_title('Detailed Classification Report', fontsize=16, fontweight='bold', pad=20)

    plt.savefig(output_dir / 'classification_report_table.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Evaluation plots saved to {output_dir}")


def main():
    """Run comprehensive classification evaluation."""
    print("=" * 80)
    print("POLLUTION IMPACT CLASSIFICATION EVALUATION")
    print("=" * 80)
    print()

    # Initialize environment
    config = initialize_environment()
    root = get_project_root()

    print("Environment initialized")
    print(f"Config loaded from: {root / 'config.yaml'}")
    print()

    # Load classifier
    model_path = root / config["paths"]["model"]
    config_path = root / config["paths"]["model_report"]

    if not model_path.exists():
        print("❌ Error: Trained model not found!")
        print(f"   Expected at: {model_path}")
        print("   Please run training first: python train_model.py")
        return

    print("Loading trained model...")
    classifier = PollutionImpactClassifier(model_path, config_path)
    print("Model loaded successfully")
    print()

    # Load test data
    print("Loading test dataset...")
    try:
        X_test, y_true_scores = load_test_data(config)
        print(f"Test dataset loaded: {len(X_test)} samples")
        print()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run feature engineering first:")
        print("   python -m src.processing.feature_engineering")
        return

    # Run evaluation
    print("Running classification evaluation...")
    print("-" * 80)

    results = evaluate_classification_performance(classifier, X_test, y_true_scores)

    # Display results
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Macro F1-Score:    {results['macro_averages']['f1_score']:.4f}")
    print(f"Weighted F1-Score: {results['weighted_averages']['f1_score']:.4f}")
    print()

    print("Per-Class Performance:")
    print("-" * 80)
    classes_present = results['classes_present']
    for i, class_name in enumerate(classes_present):
        precision = results['per_class_metrics']['precision'][i]
        recall = results['per_class_metrics']['recall'][i]
        f1 = results['per_class_metrics']['f1_score'][i]
        support = results['per_class_metrics']['support'][i]
        print(f"{class_name:>8}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} (n={support})")
    print()

    print("Class Distribution:")
    print("-" * 80)
    for class_name, count in results['class_distribution'].items():
        print(f"{class_name:>8}: {count} samples")
    print()

    print("Risk Thresholds:")
    print("-" * 80)
    print(f"Low Risk:    Score <= {results['thresholds']['low_max']}")
    print(f"Medium Risk: {results['thresholds']['low_max']} < Score <= {results['thresholds']['medium_max']}")
    print(f"High Risk:   Score > {results['thresholds']['medium_max']}")
    print()

    # Save detailed results
    output_dir = root / "models"
    results_path = output_dir / "classification_evaluation.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to: {results_path}")

    # Generate plots
    print("Generating evaluation plots...")
    plot_evaluation_results(results, output_dir)
    print("Evaluation plots saved to models/ directory")
    print()

    # Print classification report
    print("Detailed Classification Report:")
    print("-" * 80)
    y_true_classes = np.array([classifier.score_to_class(score) for score in y_true_scores])
    y_pred_classes = classifier.predict_classes(X_test)

    from sklearn.metrics import classification_report
    print(classification_report(
        y_true_classes,
        y_pred_classes,
        target_names=classes_present,
        labels=[classifier.class_names.index(cls) for cls in classes_present]
    ))
    print()

    print("=" * 80)
    print("CLASSIFICATION EVALUATION COMPLETE!")
    print("=" * 80)

    # Performance assessment
    accuracy = results['accuracy']
    macro_f1 = results['macro_averages']['f1_score']

    if accuracy >= 0.95 and macro_f1 >= 0.95:
        print("EXCELLENT PERFORMANCE! Model shows outstanding classification accuracy.")
    elif accuracy >= 0.90 and macro_f1 >= 0.90:
        print("VERY GOOD PERFORMANCE! Model demonstrates strong classification capabilities.")
    elif accuracy >= 0.85 and macro_f1 >= 0.85:
        print("GOOD PERFORMANCE! Model performs well for production use.")
    else:
        print("MODERATE PERFORMANCE! Consider model improvements or additional training data.")

    print()
    print("Next Steps:")
    print("- Start API server: cd backend && uvicorn backend.main:app --reload")
    print("- View dashboard: open data/output/dashboard.html")
    print("- Check model artifacts in models/ directory")


if __name__ == "__main__":
    main()