# Smart Factory Emission Monitoring and Pollution Control

Production-grade Python pipeline for collecting industrial geospatial data, building a pollution impact model, generating factory recommendations, and exporting an interactive map dashboard.

## Architecture

```
                  +--------------------------+
                  |      Overpass / OSM      |
                  +------------+-------------+
                               |
                  +------------v-------------+
                  |      OpenAQ / CPCB       |
                  +------------+-------------+
                               |
                 +-------------v--------------+
                 |  Ingestion (Factory + Air) |
                 +-------------+--------------+
                               |
                 +-------------v--------------+
                 |     Data Validation         |
                 |  Imputation + Range Check   |
                 +-------------+--------------+
                               |
                 +-------------v--------------+
                 |     Feature Engineering     |
                 | Spatial Join + Rolling FE   |
                 +-------------+--------------+
                               |
                 +-------------v--------------+
                 |   ML Training (RF/XGBoost)  |
                 +-------------+--------------+
                               |
            +------------------+------------------+
            |                                     |
 +----------v-----------+             +-----------v-----------+
 | Recommendation Engine|             | Folium Dashboard      |
 | risk band + controls |             | markers + heatmap     |
 +----------------------+             +-----------------------+
```

## Repository Layout

```
smart-factory-emission-monitoring/
├── data/
│   ├── raw/
│   │   ├── factories/
│   │   └── pollution/
│   ├── processed/
│   └── output/
├── src/
│   ├── ingestion/
│   │   ├── factory_collector.py
│   │   └── pollution_collector.py
│   ├── processing/
│   │   ├── data_validator.py
│   │   └── feature_engineering.py
│   ├── ml/
│   │   ├── train.py
│   │   └── predict.py
│   ├── recommendations/
│   │   └── engine.py
│   └── visualization/
│       └── dashboard.py
├── notebooks/
│   └── eda.ipynb
├── models/
├── main.py
├── config.yaml
├── requirements.txt
├── .env.example
└── backend/
```

## Setup

1. Create virtual environment:

```bash
python -m venv .venv
```

2. Activate environment:

```bash
# PowerShell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

4. Create runtime env file:

```bash
copy .env.example .env
```

5. Add API keys in `.env` if available:

```env
OPENAQ_API_KEY=
GOOGLE_PLACES_API_KEY=
LOG_LEVEL=INFO
```

## Secrets Setup (Safe For GitHub)

1. Add keys only in the local file `.env` at project root:

```env
OPENAQ_API_KEY=your_openaq_key_here
GOOGLE_PLACES_API_KEY=your_google_places_key_here
DEFAULT_COUNTRY=India
LOG_LEVEL=INFO
```

2. Do not put keys in `.env.example`. Keep that file with empty values only.
3. `.gitignore` already excludes `.env` and `.env.*`, and keeps `.env.example` tracked.

Verify secret protection:

```powershell
git status
git check-ignore -v .env
```

Expected: `.env` is ignored and does not appear in staged/untracked changes.

## Run Full Pipeline

```bash
python main.py
```

## Test And Verify

Run unit tests:

```powershell
pytest tests -q
```

Run full pipeline and verify outputs:

```powershell
python main.py
```

Check generated files:

```powershell
Get-ChildItem data/output
Get-Content data/output/pipeline.log -Tail 30
```

Open dashboard:

```powershell
Start-Process data/output/dashboard.html
```

Run API and test endpoints:

```powershell
uvicorn backend.main:app --reload
```

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
Invoke-RestMethod http://127.0.0.1:8000/factories/
Invoke-RestMethod http://127.0.0.1:8000/pollution/
```

## Outputs

- `data/raw/factories/factories.csv`
- `data/raw/pollution/pollution_readings.csv`
- `data/processed/ml_dataset.parquet`
- `models/pollution_impact_model.pkl`
- `models/scaler.pkl`
- `models/model_report.json`
- `data/output/recommendations.csv`
- `data/output/dashboard.html`
- `data/output/pipeline.log`

## API (Existing Backend Skeleton)

Run backend API:

```bash
uvicorn backend.main:app --reload
```

Available routes:

- `GET /`
- `GET /factories`
- `GET /factory/{factory_id}`
- `GET /pollution`
- `GET /pollution/stats`
- `GET /health`

## Notes

- Factory ingestion uses Overpass first, then Google Places only when key exists.
- Pollution ingestion uses OpenAQ first, then CPCB placeholder, then synthetic fallback.
- Missing pollutant values are imputed by station median, then global median.
- Out-of-range pollutant values are clipped to configured bounds in `config.yaml`.
```

Example tasks:

* Distance between **factory and pollution monitoring station**
* Pollution spread mapping
* Heatmap generation

---

# 7️⃣ Data Pipeline

```
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
ML Model Training
   ↓
Prediction API
```

Store processed data in:

```
data/processed
```

---

# 8️⃣ Deployment (Optional but impressive)

If you deploy this, it becomes **very strong project**.

Use:

Frontend

```
Vercel
```

Backend

```
Render
```

Database

```
Supabase
```

---

# 🚀 Final Architecture Diagram

```
             USER
               │
               ▼
        React Frontend
   (Map + Dashboard + UI)
               │
               ▼
        FastAPI Backend
       (API + Data Logic)
               │
       ┌───────┴────────┐
       ▼                ▼
   ML Prediction     Database
 (Pollution Impact)  (Factories +
                     Pollution Data)
       │
       ▼
Recommendation Engine
       │
       ▼
Results → Frontend Map
```

---


## Setup

Clone repo:
git clone https://github.com/username/smart-factory-emission-monitoring.git


Install dependencies:
pip install -r requirements.txt


Run project:
python main.py


---

## Future Scope
- Real-time pollution monitoring
- Satellite data integration
- Government dashboard



