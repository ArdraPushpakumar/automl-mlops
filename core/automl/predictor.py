"""
Predictor
Loads saved model and serves predictions. Logs inputs for drift monitoring.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_dir="models", log_predictions=True):
        self.model_dir = Path(model_dir)
        self.log_predictions = log_predictions
        self.pipeline = None
        self.label_encoder = None
        self.metadata = {}
        self.feature_names = []
        self.prediction_log_path = self.model_dir / "prediction_log.jsonl"
        self._load()

    def _load(self):
        model_path = self.model_dir / "best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Run: python core/run_core.py")
        self.pipeline = joblib.load(model_path)
        le_path = self.model_dir / "label_encoder.pkl"
        if le_path.exists():
            self.label_encoder = joblib.load(le_path)
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_names = self.metadata.get("feature_names", [])
        logger.info(f"Model loaded: {self.metadata.get('best_model')}")

    def _validate_input(self, data):
        df = pd.DataFrame([data])
        stats = self.metadata.get("training_data_stats", {})
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = stats.get(col, {}).get("mean", 0)
        extra = [c for c in df.columns if c not in self.feature_names]
        if extra:
            df = df.drop(columns=extra)
        return df[self.feature_names]

    def predict(self, data):
        df = self._validate_input(data)
        pred_raw = self.pipeline.predict(df.values)[0]
        try:
            proba = self.pipeline.predict_proba(df.values)[0]
            confidence = float(np.max(proba))
            proba_dict = {str(i): float(p) for i, p in enumerate(proba)}
        except Exception:
            confidence = None
            proba_dict = None

        if self.label_encoder is not None:
            pred_label = str(self.label_encoder.inverse_transform([pred_raw])[0])
        else:
            pred_label = str(pred_raw)

        result = {
            "prediction": pred_label,
            "prediction_raw": int(pred_raw),
            "confidence": confidence,
            "probabilities": proba_dict,
            "model": self.metadata.get("best_model", "unknown"),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        if self.log_predictions:
            self._log_prediction(data, result)
        return result

    def predict_batch(self, records):
        return [self.predict(r) for r in records]

    def _log_prediction(self, input_data, result):
        entry = {
            "timestamp": result["timestamp"],
            "input": input_data,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        }
        with open(self.prediction_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_model_info(self):
        return {
            "model": self.metadata.get("best_model"),
            "metric": self.metadata.get("metric"),
            "best_cv_score": self.metadata.get("best_score"),
            "features": self.feature_names,
            "target_column": self.metadata.get("target_column"),
            "trained_at": self.metadata.get("trained_at"),
            "eval_metrics": self.metadata.get("eval_metrics"),
        }
