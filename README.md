# 🤖 AutoML MLOps System

**Auto-selects and tunes the best classifier → serves predictions via API → monitors production data drift → Docker deployable to Render/Railway.**

---

## Architecture

```
Phase 1 — Core AutoML
  CSV dataset → feature engineering → Optuna hyperparameter search
  (7 models × N trials) → best model → model card

Phase 2 — FastAPI REST API
  POST /train   → upload CSV, triggers AutoML in background
  POST /predict → serve predictions (single & batch)
  GET  /model/info → metadata, model card, results
  GET  /drift/status → latest drift report

Phase 3 — Drift Monitoring
  Evidently-based PSI + KS drift detection
  Streamlit dashboard with historical reports
  Retraining alerts when drift threshold exceeded

Phase 4 — Docker + Cloud Deploy
  Dockerfile → docker-compose → Render / Railway / HuggingFace Spaces
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| AutoML | scikit-learn, XGBoost, LightGBM, Optuna (TPE sampler) |
| API | FastAPI, Uvicorn, Pydantic |
| Drift | PSI + KS test (scipy), Streamlit, Plotly |
| Deploy | Docker, Render, Railway |
| CI/CD | GitHub Actions |

---

## Phase 1 — Core AutoML

### Setup

```bash
# Clone and navigate
git clone https://github.com/YOUR_USERNAME/automl-mlops.git
cd automl-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows Git Bash

# Install dependencies
pip install -r requirements.txt
```

### Run on built-in datasets

```bash
# Breast cancer (binary classification)
cd phase1_core
python run_phase1.py

# Or run the trainer directly on any CSV:
python -m automl.trainer \
  --data data/breast_cancer.csv \
  --target target \
  --trials 30 \
  --cv 5 \
  --metric f1_weighted \
  --output models

# Multi-class (iris, wine)
python -c "
from data.data_utils import load_sklearn_dataset
path, target = load_sklearn_dataset('iris')
print(f'Dataset saved to: {path}, target: {target}')
"
python -m automl.trainer --data data/iris.csv --target target --trials 20
```

### What gets saved to `models/`

```
models/
├── best_model.pkl        # Sklearn Pipeline (scaler + best classifier)
├── label_encoder.pkl     # Target label encoder (if string labels)
├── metadata.json         # All model comparison results + best params
├── training_stats.json   # Feature statistics (reference for drift)
├── MODEL_CARD.md         # Auto-generated model card
└── prediction_log.jsonl  # Appended with every prediction (for drift)
```

---

## Phase 2 — FastAPI

### Run locally

```bash
cd automl-mlops

# Start API
uvicorn phase2_api.api.main:app --reload --port 8000

# Open docs
open http://localhost:8000/docs
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Upload and train (background job)
curl -X POST http://localhost:8000/train \
  -F "file=@data/breast_cancer.csv" \
  -F "target_column=target" \
  -F "n_trials=20" \
  -F "cv_folds=5" \
  -F "metric=f1_weighted"

# Poll training status
curl http://localhost:8000/train/status

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"mean radius": 14.0, "mean texture": 20.0, "mean perimeter": 90.0}}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records": [{"mean radius": 14.0}, {"mean radius": 20.0}]}'

# Model info
curl http://localhost:8000/model/info

# Model card
curl http://localhost:8000/model/card/raw

# All model comparison results
curl http://localhost:8000/model/results

# Drift status
curl http://localhost:8000/drift/status

# Upload production data for drift check
curl -X POST http://localhost:8000/drift/check \
  -F "file=@data/production.csv"
```

---

## Phase 3 — Drift Monitoring

### Generate test production data

```bash
cd automl-mlops

# No drift (normal production data)
python -c "
from phase1_core.data.data_utils import simulate_production_data
simulate_production_data('data/breast_cancer.csv', 'target', n_samples=300, drift=False)
print('Production data saved to data/production.csv')
"

# With drift (simulated)
python -c "
from phase1_core.data.data_utils import simulate_production_data
simulate_production_data('data/breast_cancer.csv', 'target', n_samples=300, drift=True, output_path='data/drifted.csv')
print('Drifted data saved to data/drifted.csv')
"
```

### Run drift check

```bash
# CLI drift check
python -m phase3_drift.drift.monitor models/ data/production.csv

# With drifted data
python -c "
import sys; sys.path.insert(0, '.')
from phase3_drift.drift.monitor import DriftMonitor
monitor = DriftMonitor(model_dir='models', reports_dir='drift_reports')
report = monitor.check_drift('data/drifted.csv')
print(f'Drift detected: {report[\"dataset_drift_detected\"]}')
print(f'Drifted features: {report[\"drifted_features\"]}')
"

# Check prediction log for drift
python -c "
import sys; sys.path.insert(0, '.')
from phase3_drift.drift.monitor import DriftMonitor
monitor = DriftMonitor(model_dir='models')
report = monitor.watch_prediction_log('models/prediction_log.jsonl', window=200)
print(report)
"
```

### Launch Streamlit dashboard

```bash
streamlit run phase3_drift/dashboard/app.py

# Open http://localhost:8501
```

---

## Phase 4 — Docker + Deploy

### Docker locally

```bash
# Build
docker build -t automl-mlops .

# Run API
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  automl-mlops api

# Run dashboard
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  automl-mlops dashboard

# Run both with docker-compose
docker-compose up --build

# Stop
docker-compose down
```

---

## Deploy to Render (Free Tier)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "feat: AutoML MLOps system"
git remote add origin https://github.com/YOUR_USERNAME/automl-mlops.git
git push -u origin main

# 2. Go to https://render.com → New → Web Service
# 3. Connect your GitHub repo
# 4. Settings:
#    Build Command: pip install -r requirements.txt
#    Start Command: uvicorn phase2_api.api.main:app --host 0.0.0.0 --port $PORT
#    Environment: PYTHON_VERSION=3.11.0
# 5. Add a Disk: mount at /opt/render/project/src/models (1 GB)
# 6. Click Deploy

# render.yaml is included — Render auto-detects it for Blueprint deployments
```

---

## Deploy to Railway

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create project
railway init

# 4. Deploy (railway.toml is auto-detected)
railway up

# 5. Get URL
railway open
```

---

## Deploy to HuggingFace Spaces (Streamlit dashboard)

```bash
# 1. Create a new Space at https://huggingface.co/spaces
#    Framework: Streamlit | SDK: Docker

# 2. Create HF-specific Dockerfile
cat > Dockerfile.hf << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models data drift_reports
CMD streamlit run phase3_drift/dashboard/app.py --server.port 7860 --server.address 0.0.0.0
EOF

# 3. Push to HuggingFace
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/automl-mlops
git push hf main
```

---

## Project Structure

```
automl-mlops/
├── phase1_core/
│   ├── automl/
│   │   ├── trainer.py       ← AutoML + Optuna tuning
│   │   ├── predictor.py     ← Inference + prediction logging
│   │   └── __init__.py
│   ├── data/
│   │   └── data_utils.py    ← Dataset loaders + production simulator
│   └── run_phase1.py        ← Quick-start script
│
├── phase2_api/
│   └── api/
│       └── main.py          ← FastAPI REST API
│
├── phase3_drift/
│   ├── drift/
│   │   └── monitor.py       ← PSI + KS drift detection
│   └── dashboard/
│       └── app.py           ← Streamlit monitoring dashboard
│
├── phase4_deploy/
│   └── docker/
│       └── Dockerfile       ← Multi-stage Docker build
│
├── .github/
│   └── workflows/
│       └── ci.yml           ← GitHub Actions CI/CD
│
├── Dockerfile               ← Root Dockerfile (for cloud deploy)
├── docker-compose.yml       ← Local dev orchestration
├── render.yaml              ← Render Blueprint config
├── railway.toml             ← Railway deploy config
├── requirements.txt         ← Combined dependencies
├── entrypoint.sh            ← API / Dashboard / Train switch
└── README.md
```

---

## Models Available

| Model | Notes |
|-------|-------|
| `random_forest` | Robust baseline, interpretable feature importance |
| `gradient_boosting` | Sklearn GBM, good for medium datasets |
| `extra_trees` | Faster than RF, similar accuracy |
| `logistic_regression` | Fast, interpretable linear baseline |
| `xgboost` | Production-grade boosting |
| `lightgbm` | Fast boosting, good for large datasets |
| `svm` | Effective on smaller datasets |

Restrict models with `--models xgboost lightgbm random_forest` for faster iteration.

---

## Drift Detection Logic

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| PSI | > 0.2 | Feature distribution shifted significantly |
| KS p-value | < 0.05 | Statistically significant distribution change |
| Drift share | ≥ 30% features | Dataset-level drift → retraining alert |

---

## For Your Resume / Portfolio

**Key talking points:**
- Designed an end-to-end AutoML system that automatically selects and tunes 7 classifier families using Optuna's TPE sampler
- Implemented production drift monitoring using PSI + Kolmogorov-Smirnov tests with automated retraining triggers
- Built a complete MLOps pipeline: REST API (FastAPI) → containerized deployment (Docker) → cloud hosting (Render/Railway)
- Prediction logging enables continuous drift monitoring without external dependencies
- Auto-generates model cards with full training metadata for model governance
