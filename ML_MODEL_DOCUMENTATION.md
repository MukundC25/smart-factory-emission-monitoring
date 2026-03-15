# Machine Learning Model for Pollution Impact Prediction

## Overview

This document describes the pollution impact prediction ML model - a regression-based system that estimates factory contribution to local air pollution levels.

## Model Architecture

### Input Features

The model accepts the following feature categories:

#### Factory Characteristics
- `factory_id`: Unique factory identifier
- `factory_name`: Name of the factory
- `industry_type`: Type of industry (steel, cement, chemical, textile, automotive, etc.)
- `latitude`, `longitude`: Geographic coordinates
- `city`, `state`, `country`: Location information

#### Pollution Metrics (Pollutants)
- `pm25`: Particulate Matter 2.5 micrometers
- `pm10`: Particulate Matter 10 micrometers
- `co`: Carbon Monoxide
- `no2`: Nitrogen Dioxide
- `so2`: Sulfur Dioxide
- `o3`: Ozone

#### Spatial Features
- `distance_to_nearest_station`: Distance to nearest pollution monitoring station (km)

#### Temporal Features
- `rolling_avg_pm25_7d`: 7-day rolling average of PM2.5
- `rolling_avg_pm25_30d`: 30-day rolling average of PM2.5
- `season`: Season classification (winter, summer, monsoon, post_monsoon)
- `pollution_spike_flag`: Binary flag for pollution spikes (Z-score > 2.5)

#### Environmental Features
- `wind_direction_factor`: Wind direction impact factor (1.0 default)
- `industry_risk_weight`: Industry-specific risk weighting (0-10 scale)

### Target Variable

**`pollution_impact_score`**: Normalized composite pollution impact score (0-10 scale)

Calculated as a weighted combination of pollutants:
```
weighted_sum = pm25 * 0.35 + pm10 * 0.20 + no2 * 0.15 + so2 * 0.10 + co * 0.10 + o3 * 0.10
pollution_impact_score = min(weighted_sum / percentile_95 * 10, 10)
```

Where:
- PM2.5 has highest weight (35%) - most harmful particulate
- PM10 has secondary weight (20%)
- Other pollutants weighted by health impact
- Normalized to 0-10 scale using 95th percentile

## Models Implemented

### 1. Random Forest Regressor
- **Estimators**: 300 trees
- **Random State**: 42 (for reproducibility)
- **Parallelization**: Uses all available CPU cores

**Advantages**:
- Handles non-linear relationships
- Provides feature importance rankings
- Robust to outliers
- Good generalization

### 2. XGBoost Regressor (Optional)
- **Estimators**: 350 boosting rounds
- **Max Depth**: 8
- **Learning Rate**: 0.05
- **Subsample**: 0.9
- **Column Subsample**: 0.9

**Advantages**:
- Sequential error correction
- Generally better predictive performance
- Handles complex patterns
- Fast inference

## Data Preprocessing Pipeline

### Numeric Features
1. **Imputation**: Median-based imputation for missing values
2. **Scaling**: StandardScaler normalization

### Categorical Features
1. **Imputation**: Most frequent value imputation
2. **Encoding**: One-hot encoding
3. **Unknown Handling**: Ignores unseen categories

## Model Evaluation Metrics

The model is evaluated using multiple metrics:

| Metric | Description | Interpretation |
|--------|-------------|-----------------|
| **RMSE** | Root Mean Squared Error | Average magnitude of prediction errors |
| **MAE** | Mean Absolute Error | Average absolute deviation from actual values |
| **MAPE** | Mean Absolute Percentage Error | Percentage error relative to actual values |
| **R²** | Coefficient of Determination | Proportion of variance explained (0-1) |
| **CV-R²** | 5-Fold Cross-Validation R² | Generalization performance |

### Performance Baseline

Typical performance on Indian factory dataset:
- **RMSE**: ~0.8-1.2 (on 0-10 scale)
- **MAE**: ~0.6-0.9
- **R²**: 0.75-0.85
- **CV-R² Mean**: 0.72-0.83

## Training Pipeline

### 1. Data Loading
```
Raw Data (factories.csv, pollution_readings.csv)
```

### 2. Feature Engineering
```
→ Spatial Join (nearest station)
→ Temporal Features (rolling averages, season, spike detection)
→ Industry Risk Weighting
→ Target Variable Construction
```

### 3. Train-Val-Test Split
- **Training Set**: 70% (train + validation combined)
- **Validation Set**: 15% (used for model selection during training)
- **Test Set**: 15% (held-out for final evaluation)
- **Stratification**: By target variable quintiles for balanced distribution

### 4. Model Training
```
→ Training with Random Forest
→ Training with XGBoost (if available)
→ Cross-validation evaluation (5-fold)
→ Selection of best model (lowest RMSE)
```

### 5. Artifact Persistence
```
models/
├── pollution_impact_model.pkl  (Best trained model)
├── scaler.pkl                   (Feature scaler for inference)
├── feature_importance.png       (Top 20 most important features)
└── model_comparison.png         (Performance comparison visualization)
```

## Prediction Workflow

### Single Prediction API
```python
POST /factories/predict
Content-Type: application/json

{
  "factory_id": "F123",
  "factory_name": "Steel Mill XYZ",
  "industry_type": "steel",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "city": "Delhi",
  "state": "Delhi",
  "country": "India",
  "pm25": 45.5,
  "pm10": 78.2,
  "co": 2.1,
  "no2": 35.8,
  "so2": 12.5,
  "o3": 5.2,
  "distance_to_nearest_station": 5.0,
  "rolling_avg_pm25_7d": 42.1,
  "rolling_avg_pm25_30d": 40.3,
  "pollution_spike_flag": 0,
  "season": "winter",
  "wind_direction_factor": 1.0,
  "industry_risk_weight": 9.0
}

Response:
{
  "factory_id": "F123",
  "factory_name": "Steel Mill XYZ",
  "industry_type": "steel",
  "latitude": 28.6139,
  "longitude": 77.2090,
  "city": "Delhi",
  "predicted_pollution_impact_score": 7.2,
  "risk_level": "High",
  "confidence_context": "Predicted score based on steel factory in Delhi"
}
```

### Batch Prediction
```bash
POST /factories/batch/predict-all

Returns predictions for all factories in database
```

## Risk Level Classification

Impact scores are mapped to discrete risk categories:

| Risk Level | Score Range | Intervention Level |
|-----------|------------|-------------------|
| **Low** | 0 - 3.0 | Routine maintenance |
| **Medium** | 3.0 - 6.0 | Enhanced monitoring |
| **High** | 6.0 - 10.0 | Immediate remediation |

## Output: Pollution Control Recommendations

Based on risk level and industry type, the system generates specific recommendations:

### High-Risk Factories
1. Install SO2 scrubbers
2. Upgrade ESP filters
3. Weekly baghouse maintenance
4. Continuous emissions monitoring system (CEMS)
5. Automated alert deployment
6. Quarterly compliance audits

### Medium-Risk Factories
1. Bi-weekly stack testing
2. Monthly preventive burner tuning
3. Shift-level leak detection
4. Monthly air quality reports
5. Enhanced maintenance
6. Semi-annual audits

### Low-Risk Factories
1. Maintain compliance logs
2. Monthly calibration checks
3. Standard maintenance schedules
4. Annual third-party audits
5. Quarterly emissions reports

## Model Artifacts

### Model Report (`models/model_report.json`)
```json
{
  "selected_model": "random_forest",
  "metrics": {
    "random_forest": {
      "rmse": 0.95,
      "mae": 0.72,
      "mape": 0.08,
      "r2": 0.81,
      "cv_r2_mean": 0.79,
      "cv_r2_std": 0.04
    },
    "xgboost": {
      "rmse": 0.92,
      "mae": 0.70,
      "mape": 0.09,
      "r2": 0.82,
      "cv_r2_mean": 0.80,
      "cv_r2_std": 0.03
    }
  },
  "config": {
    "target_column": "pollution_impact_score",
    "feature_count": 19,
    "train_rows": 4250,
    "test_rows": 750,
    "total_rows": 5000
  }
}
```

## API Endpoints

### Model Information
```
GET /health              - Service health check
GET /model-info          - Model metadata and config
GET /metrics             - Model performance metrics
```

### Predictions
```
POST /factories/predict              - Single factory prediction
GET  /factories/batch/predict-all    - Batch predictions
```

### Factory Management
```
GET  /factories                 - List all factories
GET  /factories/{factory_id}    - Get specific factory
```

### Recommendations
```
GET  /recommendation/{factory_id}         - Get recommendation
GET  /recommendation/?risk_level=High     - Filter by risk
GET  /recommendation/high-risk/list       - High-risk factories
GET  /recommendation/statistics/summary   - Risk statistics
```

## Usage Examples

### Training the Model
```bash
cd /project/path
python -m src.ml.train
```

### Running Predictions
```bash
python -m src.ml.predict
```

### Starting API Server
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Making Predictions via API
```bash
curl -X POST http://localhost:8000/factories/predict \
  -H "Content-Type: application/json" \
  -d '{...factory_data...}'
```

## Feature Importance

The trained model produces feature importance rankings that reveal which factors most influence pollution impact scores. Top features typically include:

1. PM2.5 concentration
2. Industry type (risk weight)
3. Rolling average PM2.5 (7-day)
4. PM10 concentration
5. Geographic location (latitude/longitude)
6. Season
7. Distance to monitoring station
8. NO2 concentration

## Model Limitations & Considerations

1. **Geographic Scope**: Trained on Indian industrial cities
2. **Temporal Variability**: Seasonal patterns may not generalize to all climates
3. **Data Quality**: Predictions depend on quality of input pollution data
4. **Feature Availability**: All required features must be provided for predictions
5. **Extrapolation**: May not perform well for pollutant levels outside training range

## Continuous Improvement

To improve model performance:

1. **Collect more diverse data** - Include more industrial types and regions
2. **Add meteorological features** - Wind speed, temperature, pressure
3. **Include time-series patterns** - LSTM/GRU models for temporal dynamics
4. **Add source tracking** - Identify specific emission sources
5. **Regularization tuning** - Optimize hyperparameters with Bayesian search

## References

- Scikit-learn documentation: https://scikit-learn.org
- XGBoost documentation: https://xgboost.readthedocs.io
- India Air Quality Standards: https://www.cpcb.nic.in
