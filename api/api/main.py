"""
FastAPI Application
Run from repo root: uvicorn api.api.main:app --reload --port 8000
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.automl.trainer import AutoMLTrainer
from core.automl.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AutoML MLOps API",
    description="Upload any tabular CSV, auto-tune best classifier, serve predictions, monitor drift",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Optional[Predictor] = None
_training_status: Dict[str, Any] = {"status": "idle", "message": "No training started yet."}


def _try_load_predictor():
    global _predictor
    try:
        _predictor = Predictor(model_dir=str(MODEL_DIR))
        logger.info("Predictor loaded on startup.")
    except FileNotFoundError:
        logger.info("No model found. Train first via POST /train")


@app.on_event("startup")
async def startup():
    _try_load_predictor()


class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictBatchRequest(BaseModel):
    records: List[Dict[str, Any]]


def _run_training(data_path, target_column, n_trials, cv_folds, metric, models_to_try):
    global _predictor, _training_status
    _training_status = {"status": "running", "message": "Training in progress..."}
    try:
        trainer = AutoMLTrainer(
            target_column=target_column,
            n_trials=n_trials,
            cv_folds=cv_folds,
            metric=metric,
            output_dir=str(MODEL_DIR),
            models_to_try=models_to_try,
        )
        trainer.fit(data_path)
        _predictor = Predictor(model_dir=str(MODEL_DIR))
        _training_status = {
            "status": "complete",
            "message": f"Training complete. Best: {trainer.best_model_name}",
            "best_model": trainer.best_model_name,
            "best_score": trainer.best_score,
            "metric": trainer.metric,
        }
    except Exception as e:
        _training_status = {"status": "error", "message": str(e)}
        logger.error(f"Training failed: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body style="font-family:sans-serif;padding:2rem">
    <h1>AutoML MLOps API</h1>
    <ul>
      <li><a href="/docs">Swagger UI</a></li>
      <li><a href="/health">Health</a></li>
      <li><a href="/model/info">Model Info</a></li>
      <li><a href="/model/results">All Model Results</a></li>
      <li><a href="/drift/status">Drift Status</a></li>
    </ul>
    <p><b>POST /train</b> to upload a CSV and start AutoML.</p>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _predictor is not None,
        "training_status": _training_status.get("status"),
    }


@app.post("/train")
async def train(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    n_trials: int = Form(20),
    cv_folds: int = Form(5),
    metric: str = Form("f1_weighted"),
    models_to_try: Optional[str] = Form(None),
):
    tmp_path = DATA_DIR / f"uploaded_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    models = [m.strip() for m in models_to_try.split(",")] if models_to_try else None
    background_tasks.add_task(
        _run_training, str(tmp_path), target_column, n_trials, cv_folds, metric, models)
    return {"message": "Training started.", "poll": "GET /train/status"}


@app.get("/train/status")
async def train_status():
    return _training_status


@app.post("/predict")
async def predict(request: PredictRequest):
    if _predictor is None:
        raise HTTPException(503, "No model loaded. Train first via POST /train")
    try:
        return _predictor.predict(request.features)
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/predict/batch")
async def predict_batch(request: PredictBatchRequest):
    if _predictor is None:
        raise HTTPException(503, "No model loaded.")
    results = _predictor.predict_batch(request.records)
    return {"predictions": results, "count": len(results)}


@app.get("/model/info")
async def model_info():
    if _predictor is None:
        raise HTTPException(404, "No model loaded.")
    return _predictor.get_model_info()


@app.get("/model/results")
async def model_results():
    meta_path = MODEL_DIR / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, "No metadata found. Train first.")
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "best_model": meta.get("best_model"),
        "all_results": meta.get("all_model_results", {}),
        "metric": meta.get("metric"),
    }


@app.get("/model/card")
async def model_card():
    card_path = MODEL_DIR / "MODEL_CARD.md"
    if not card_path.exists():
        raise HTTPException(404, "Model card not found. Train first.")
    return {"content": card_path.read_text()}


@app.get("/drift/status")
async def drift_status():
    drift_path = MODEL_DIR / "drift_report.json"
    if not drift_path.exists():
        return {"drift_detected": None, "message": "No drift report yet. Run POST /drift/check"}
    with open(drift_path) as f:
        return json.load(f)


@app.post("/drift/check")
async def trigger_drift_check(
    file: UploadFile = File(..., description="Production CSV without target column")
):
    try:
        from drift.drift.monitor import DriftMonitor
        tmp = DATA_DIR / "prod_upload.csv"
        with open(tmp, "wb") as f:
            shutil.copyfileobj(file.file, f)
        monitor = DriftMonitor(model_dir=str(MODEL_DIR))
        return monitor.check_drift(str(tmp))
    except ImportError:
        raise HTTPException(501, "Drift monitor not available.")
    except Exception as e:
        raise HTTPException(400, str(e))
