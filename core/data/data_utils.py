"""
Data utilities — dataset loading, splitting, production data simulation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

logger = logging.getLogger(__name__)

SAMPLE_DATASETS = {
    "breast_cancer": {"loader": load_breast_cancer, "target": "target"},
    "iris":          {"loader": load_iris,          "target": "target"},
    "wine":          {"loader": load_wine,           "target": "target"},
}


def load_sklearn_dataset(name, output_dir="core/data"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if name not in SAMPLE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(SAMPLE_DATASETS.keys())}")
    info = SAMPLE_DATASETS[name]
    bunch = info["loader"]()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {name} to {path} | shape={df.shape}")
    return str(path), info["target"]


def simulate_production_data(reference_path, target, n_samples=200,
                              drift=False, output_path="data/production.csv"):
    df = pd.read_csv(reference_path)
    feature_cols = [c for c in df.columns if c != target]
    prod_data = {}
    for col in feature_cols:
        col_mean = df[col].mean()
        col_std  = df[col].std()
        samples  = np.random.normal(col_mean, col_std, n_samples)
        if drift and np.random.rand() > 0.7:
            samples += col_std * np.random.uniform(1.5, 3.0)
        prod_data[col] = samples
    prod_df = pd.DataFrame(prod_data)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    prod_df.to_csv(output_path, index=False)
    logger.info(f"Production data (drift={drift}) saved to {output_path}")
    return output_path
