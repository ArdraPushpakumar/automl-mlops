#!/usr/bin/env python3
"""
Quick-start script. Run from repo root: python core/run_core.py
"""
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from core.data.data_utils import load_sklearn_dataset, simulate_production_data
from core.automl.trainer import AutoMLTrainer
from core.automl.predictor import Predictor

DATASET   = "breast_cancer"
TRIALS    = 15
CV_FOLDS  = 3
MODEL_DIR = "models"

data_path, target = load_sklearn_dataset(DATASET, output_dir="core/data")

trainer = AutoMLTrainer(
    target_column=target,
    n_trials=TRIALS,
    cv_folds=CV_FOLDS,
    metric="f1_weighted",
    output_dir=MODEL_DIR,
)
trainer.fit(data_path)

print("\n" + "=" * 55)
predictor = Predictor(model_dir=MODEL_DIR)
info = predictor.get_model_info()
print(f"Best model : {info['model']}")
print(f"CV score   : {info['best_cv_score']:.4f}")

sample = {f: v["mean"] for f, v in trainer.training_data_stats.items()}
result = predictor.predict(sample)
print(f"Prediction : {result['prediction']}  (confidence={result['confidence']:.3f})")
print("=" * 55)
print("Artifacts  : models/")
print("Model card : models/MODEL_CARD.md")

simulate_production_data(data_path, target, n_samples=300, drift=False,
                         output_path="data/production_normal.csv")
simulate_production_data(data_path, target, n_samples=300, drift=True,
                         output_path="data/production_drifted.csv")
print("Sim data   : data/production_normal.csv | data/production_drifted.csv")
