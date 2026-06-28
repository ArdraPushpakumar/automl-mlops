import json, time, warnings, logging, inspect
from pathlib import Path
import numpy as np, pandas as pd, optuna, joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
    "logistic_regression": LogisticRegression,
    "xgboost": XGBClassifier,
    "lightgbm": LGBMClassifier,
    "svm": SVC,
}

def get_params_for_model(model_name, trial):
    if model_name == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 3, 15), "min_samples_split": trial.suggest_int("min_samples_split", 2, 10), "random_state": 42}
    elif model_name == "gradient_boosting":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 8), "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True), "random_state": 42}
    elif model_name == "extra_trees":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 3, 15), "random_state": 42}
    elif model_name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-4, 100.0, log=True), "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]), "max_iter": 2000, "random_state": 42}
    elif model_name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 8), "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True), "eval_metric": "logloss", "random_state": 42, "verbosity": 0}
    elif model_name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300), "max_depth": trial.suggest_int("max_depth", 2, 10), "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True), "num_leaves": trial.suggest_int("num_leaves", 20, 100), "random_state": 42, "verbose": -1}
    elif model_name == "svm":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True), "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]), "probability": True, "random_state": 42}
    return {}

class AutoMLTrainer:
    def __init__(self, target_column, n_trials=30, cv_folds=5, metric="f1_weighted", output_dir="models", models_to_try=None):
        self.target_column = target_column
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.metric = metric
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_to_try = models_to_try or list(CANDIDATE_MODELS.keys())
        self.best_pipeline = None
        self.best_score = -float("inf")
        self.best_model_name = None
        self.best_params = None
        self.results = {}
        self.feature_names = None
        self.label_encoder = None
        self.training_data_stats = {}
        self.eval_metrics = {}

    def load_data(self, path):
        logger.info(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        logger.info(f"Shape: {df.shape}")
        if self.target_column not in df.columns:
            raise ValueError(f"Target not found. Available: {list(df.columns)}")
        df = df.dropna(subset=[self.target_column])
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        for col in X.select_dtypes(include=["object","category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median(numeric_only=True))
        if y.dtype == object or str(y.dtype) == "category":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
        else:
            y = y.values
        self.feature_names = list(X.columns)
        self.training_data_stats = {col: {"mean": float(X[col].mean()), "std": float(X[col].std()), "min": float(X[col].min()), "max": float(X[col].max()), "q25": float(X[col].quantile(0.25)), "q75": float(X[col].quantile(0.75))} for col in X.columns}
        logger.info(f"Features: {len(self.feature_names)} | Classes: {np.unique(y)}")
        return X.values, y

    def tune_model(self, model_name, X, y):
        logger.info(f"Tuning {model_name} ({self.n_trials} trials)...")
        def objective(trial):
            params = get_params_for_model(model_name, trial)
            try:
                clf = CANDIDATE_MODELS[model_name](**params)
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                return cross_val_score(pipe, X, y, cv=cv, scoring=self.metric, n_jobs=-1).mean()
            except Exception as e:
                return -1.0
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = get_params_for_model(study.best_trial, None) if False else {k: v for k, v in study.best_trial.params.items()}
        # Reconstruct clean params by calling get_params_for_model with a fixed-value trial
        clean = get_params_for_model(model_name, study.best_trial)
        return study.best_value, clean

    def fit(self, data_path):
        start = time.time()
        X, y = self.load_data(data_path)
        logger.info(f"AutoML: {len(self.models_to_try)} models x {self.n_trials} trials")
        for model_name in self.models_to_try:
            try:
                score, params = self.tune_model(model_name, X, y)
                self.results[model_name] = {"score": round(score, 4), "params": params}
                logger.info(f"  {model_name}: {self.metric} = {score:.4f}")
                if score > self.best_score:
                    self.best_score = score
                    self.best_model_name = model_name
                    self.best_params = params
            except Exception as e:
                logger.error(f"  {model_name} failed: {e}")
                self.results[model_name] = {"score": None, "params": None, "error": str(e)}
        logger.info(f"Best: {self.best_model_name} ({self.best_score:.4f})")
        logger.info(f"Best params: {self.best_params}")
        best_clf = CANDIDATE_MODELS[self.best_model_name](**self.best_params)
        self.best_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", best_clf)])
        self.best_pipeline.fit(X, y)
        y_pred = self.best_pipeline.predict(X)
        try:
            y_proba = self.best_pipeline.predict_proba(X)
            n_classes = len(np.unique(y))
            roc = roc_auc_score(y, y_proba if n_classes > 2 else y_proba[:,1], multi_class="ovr" if n_classes > 2 else "raise")
        except Exception:
            roc = None
        self.eval_metrics = {"accuracy": float(accuracy_score(y, y_pred)), "f1_weighted": float(f1_score(y, y_pred, average="weighted")), "roc_auc": float(roc) if roc else None, "classification_report": classification_report(y, y_pred, output_dict=True)}
        duration = round(time.time() - start, 2)
        logger.info(f"Done in {duration}s | Accuracy: {self.eval_metrics['accuracy']:.4f}")
        self._save_artifacts(data_path, duration)
        self._generate_model_card(data_path, duration)
        return self

    def _save_artifacts(self, data_path, duration):
        joblib.dump(self.best_pipeline, self.output_dir / "best_model.pkl")
        joblib.dump(self.label_encoder, self.output_dir / "label_encoder.pkl")
        metadata = {"best_model": self.best_model_name, "best_score": self.best_score, "metric": self.metric, "best_params": self.best_params, "feature_names": self.feature_names, "target_column": self.target_column, "eval_metrics": self.eval_metrics, "all_model_results": self.results, "training_data_stats": self.training_data_stats, "training_duration_sec": duration, "trained_at": pd.Timestamp.now().isoformat(), "data_source": str(data_path)}
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(self.training_data_stats, f, indent=2)
        logger.info(f"Artifacts saved to {self.output_dir}")

    def _generate_model_card(self, data_path, duration):
        sorted_results = sorted([(k,v) for k,v in self.results.items() if v.get("score") is not None], key=lambda x: x[1]["score"], reverse=True)
        rows = "\n".join(f"| {k} | {v['score']} |" for k,v in sorted_results)
        card = f"""# Model Card\n\n| Field | Value |\n|-------|-------|\n| Algorithm | {self.best_model_name} |\n| Metric | {self.metric} |\n| CV Score | {self.best_score:.4f} |\n| Duration | {duration}s |\n\n## Features\n{", ".join(self.feature_names)}\n\n## Model Comparison\n| Model | Score |\n|-------|-------|\n{rows}\n"""
        with open(self.output_dir / "MODEL_CARD.md", "w") as f:
            f.write(card)
        logger.info("Model card saved")
