"""Script to run model training and display results."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ml.train import train_models
from src.common import initialize_environment, get_project_root

if __name__ == "__main__":
    print("=" * 70)
    print("SMART FACTORY EMISSION MONITORING - ML MODEL TRAINING")
    print("=" * 70)
    print()
    
    config = initialize_environment()
    print(f"✓ Environment initialized")
    print(f"✓ Config loaded from: {get_project_root() / 'config.yaml'}")
    print()
    
    print("Starting model training...")
    print("-" * 70)
    
    result = train_models(config)
    
    print("-" * 70)
    print()
    print("TRAINING COMPLETE!")
    print()
    print(f"Best Model Selected: {result['selected_model'].upper()}")
    print()
    
    print("Model Performance Metrics:")
    print("-" * 70)
    for model_name, metrics in result['metrics'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}")
        print(f"  MAE  (Mean Absolute Error):     {metrics['mae']:.4f}")
        print(f"  MAPE (Mean Absolute % Error):   {metrics['mape']:.4f}")
        print(f"  R²   (Coefficient of Determination): {metrics['r2']:.4f}")
        print(f"  CV-R² Mean (5-fold Cross-Val):  {metrics['cv_r2_mean']:.4f}")
        print(f"  CV-R² Std:                      {metrics['cv_r2_std']:.4f}")
    
    print()
    print("=" * 70)
    print("Training Configuration:")
    print("-" * 70)
    print(f"  Target Column: {result['config']['target_column']}")
    print(f"  Features: {result['config']['feature_count']}")
    print(f"  Training Samples: {result['config']['train_rows']}")
    print(f"  Test Samples: {result['config']['test_rows']}")
    print(f"  Total Samples: {result['config']['total_rows']}")
    
    print()
    print("=" * 70)
    print("Saved Artifacts:")
    print("-" * 70)
    for artifact_name, artifact_path in result['artifacts'].items():
        print(f"  {artifact_name.replace('_', ' ').title()}: {artifact_path}")
    
    print()
    print("=" * 70)
    print("Next Steps:")
    print("-" * 70)
    print("1. Run recommendations generation:")
    print("   python -m src.recommendations.engine")
    print()
    print("2. Build the dashboard:")
    print("   python -m src.visualization.dashboard")
    print()
    print("3. Start the FastAPI backend:")
    print("   cd backend && uvicorn backend.main:app --reload")
    print()
    print("=" * 70)
