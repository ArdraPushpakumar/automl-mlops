"""
Drift Monitor
PSI + KS drift detection. Run from repo root:
  python drift/drift/monitor.py models/ data/production_drifted.csv
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DriftMonitor:
    PSI_THRESHOLD         = 0.2
    KS_PVALUE_THRESHOLD   = 0.05
    DRIFT_SHARE_THRESHOLD = 0.3

    def __init__(self, model_dir="models", reports_dir="drift_reports"):
        self.model_dir   = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names  = []
        self.training_stats = {}
        self._load_reference()

    def _load_reference(self):
        stats_path = self.model_dir / "training_stats.json"
        meta_path  = self.model_dir / "metadata.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"training_stats.json not found in {self.model_dir}. "
                "Run training first: python core/run_core.py")
        with open(stats_path) as f:
            self.training_stats = json.load(f)
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", list(self.training_stats.keys()))
        else:
            self.feature_names = list(self.training_stats.keys())
        logger.info(f"Reference loaded: {len(self.feature_names)} features")

    def _compute_psi(self, reference, production, bins=10):
        ref_min, ref_max = np.min(reference), np.max(reference)
        if ref_max == ref_min:
            return 0.0
        edges = np.linspace(ref_min, ref_max, bins + 1)
        ref_counts,  _ = np.histogram(reference,  bins=edges)
        prod_counts, _ = np.histogram(production, bins=edges)
        ref_pct  = (ref_counts  + 1e-6) / (len(reference)  + 1e-6 * bins)
        prod_pct = (prod_counts + 1e-6) / (len(production) + 1e-6 * bins)
        return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

    def _ks_test(self, reference, production):
        from scipy.stats import ks_2samp
        stat, pvalue = ks_2samp(reference, production)
        return float(stat), float(pvalue)

    def _generate_reference_samples(self, n=1000):
        data = {}
        for feat in self.feature_names:
            if feat in self.training_stats:
                s = self.training_stats[feat]
                data[feat] = np.random.normal(s["mean"], max(s["std"], 1e-6), n)
        return pd.DataFrame(data)[self.feature_names]

    def check_drift(self, production_path):
        logger.info(f"Checking drift: {production_path}")
        prod_df   = pd.read_csv(production_path)
        available = [f for f in self.feature_names if f in prod_df.columns]
        missing   = [f for f in self.feature_names if f not in prod_df.columns]
        if missing:
            logger.warning(f"Missing features (skipping): {missing}")

        reference_df     = self._generate_reference_samples(n=max(1000, len(prod_df)))
        feature_results  = {}
        drifted_features = []

        for feat in available:
            ref_vals  = reference_df[feat].values
            prod_vals = prod_df[feat].fillna(prod_df[feat].median()).values
            psi              = self._compute_psi(ref_vals, prod_vals)
            ks_stat, ks_pval = self._ks_test(ref_vals, prod_vals)
            is_drifted       = psi > self.PSI_THRESHOLD or ks_pval < self.KS_PVALUE_THRESHOLD
            if is_drifted:
                drifted_features.append(feat)
            feature_results[feat] = {
                "psi":          round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue":    round(ks_pval, 4),
                "drifted":      is_drifted,
                "ref_mean":     round(self.training_stats[feat]["mean"], 4),
                "prod_mean":    round(float(prod_vals.mean()), 4),
                "ref_std":      round(self.training_stats[feat]["std"], 4),
                "prod_std":     round(float(prod_vals.std()), 4),
            }

        drift_share     = len(drifted_features) / len(available) if available else 0
        dataset_drifted = drift_share >= self.DRIFT_SHARE_THRESHOLD

        report = {
            "timestamp":              datetime.now().isoformat(),
            "production_file":        production_path,
            "n_production_samples":   len(prod_df),
            "n_features_checked":     len(available),
            "n_features_drifted":     len(drifted_features),
            "drift_share":            round(drift_share, 4),
            "dataset_drift_detected": dataset_drifted,
            "retraining_recommended": dataset_drifted,
            "drifted_features":       drifted_features,
            "thresholds": {
                "psi":         self.PSI_THRESHOLD,
                "ks_pvalue":   self.KS_PVALUE_THRESHOLD,
                "drift_share": self.DRIFT_SHARE_THRESHOLD,
            },
            "feature_results": feature_results,
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.reports_dir / f"drift_report_{ts}.json", "w") as f:
            json.dump(report, f, indent=2)
        with open(self.model_dir / "drift_report.json", "w") as f:
            json.dump(report, f, indent=2)
        self._generate_html_report(report, ts)

        if dataset_drifted:
            logger.warning(
                f"DRIFT DETECTED: {len(drifted_features)}/{len(available)} features "
                f"({drift_share:.1%}). Retraining recommended.")
        else:
            logger.info(f"No significant drift ({drift_share:.1%} features drifted).")
        return report

    def watch_prediction_log(self, log_path, window=200):
        log_file = Path(log_path)
        if not log_file.exists():
            return {"error": "Prediction log not found", "drift_detected": None}
        rows = []
        with open(log_file) as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()).get("input", {}))
                except Exception:
                    continue
        if len(rows) < 10:
            return {"message": f"Only {len(rows)} predictions logged — need at least 10."}
        prod_df  = pd.DataFrame(rows[-window:])
        tmp_path = Path("data/prediction_log_window.csv")
        tmp_path.parent.mkdir(exist_ok=True)
        prod_df.to_csv(tmp_path, index=False)
        return self.check_drift(str(tmp_path))

    def _generate_html_report(self, report, timestamp):
        rows = ""
        for feat, r in report["feature_results"].items():
            color = "#ffcccc" if r["drifted"] else "#ccffcc"
            rows += (
                f"<tr style='background:{color}'>"
                f"<td>{feat}</td><td>{r['psi']}</td><td>{r['ks_pvalue']}</td>"
                f"<td>{r['ref_mean']} +/- {r['ref_std']}</td>"
                f"<td>{r['prod_mean']} +/- {r['prod_std']}</td>"
                f"<td>{'YES' if r['drifted'] else 'NO'}</td></tr>"
            )
        badge = (
            "<span style='color:red'>DRIFT DETECTED - Retraining Recommended</span>"
            if report["dataset_drift_detected"]
            else "<span style='color:green'>No Significant Drift</span>"
        )
        html = f"""<!DOCTYPE html>
<html><head><title>Drift Report {timestamp}</title>
<style>
  body{{font-family:Arial,sans-serif;padding:2rem;background:#f5f5f5}}
  .card{{background:white;padding:1.5rem;border-radius:8px;margin-bottom:1rem;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#333;color:white;padding:8px;text-align:left}}
  td{{padding:8px;border-bottom:1px solid #ddd}}
</style></head>
<body>
  <h1>Data Drift Report</h1>
  <p>Generated: {report['timestamp']} | Samples: {report['n_production_samples']}</p>
  <div class="card"><h2>{badge}</h2>
    <p>Features drifted: <b>{report['n_features_drifted']}/{report['n_features_checked']}</b> ({report['drift_share']:.1%})</p>
  </div>
  <div class="card">
    <h2>Feature Results</h2>
    <table>
      <tr><th>Feature</th><th>PSI</th><th>KS p-value</th><th>Ref Mean +/- Std</th><th>Prod Mean +/- Std</th><th>Drifted?</th></tr>
      {rows}
    </table>
  </div>
  <div class="card"><h3>Thresholds</h3>
    <ul>
      <li>PSI &gt; {report['thresholds']['psi']} = feature drift</li>
      <li>KS p-value &lt; {report['thresholds']['ks_pvalue']} = feature drift</li>
      <li>30%+ features drifted = dataset alert + retraining</li>
    </ul>
  </div>
</body></html>"""
        html_path = self.reports_dir / f"drift_report_{timestamp}.html"
        html_path.write_text(html)
        logger.info(f"HTML report saved: {html_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    prod_path = sys.argv[2] if len(sys.argv) > 2 else "data/production_drifted.csv"
    monitor   = DriftMonitor(model_dir=model_dir)
    report    = monitor.check_drift(prod_path)
    summary   = {k: v for k, v in report.items() if k != "feature_results"}
    print(json.dumps(summary, indent=2))
