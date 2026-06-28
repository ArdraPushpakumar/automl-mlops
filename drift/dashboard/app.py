"""
Drift Monitoring Dashboard — Phase 3
Streamlit dashboard showing drift reports, model info, and retraining alerts.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

MODEL_DIR = Path("models")
REPORTS_DIR = Path("drift_reports")

st.set_page_config(
    page_title="AutoML MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AutoML MLOps Dashboard")
st.caption("Model monitoring, drift detection, and retraining alerts")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio("View", ["📊 Model Overview", "🔍 Drift Monitor", "📁 Upload & Check", "📋 Model Card"])

    st.divider()
    if (MODEL_DIR / "metadata.json").exists():
        with open(MODEL_DIR / "metadata.json") as f:
            meta = json.load(f)
        st.metric("Best Model", meta.get("best_model", "N/A"))
        st.metric("CV Score", f"{meta.get('best_score', 0):.4f}")
        st.metric("Metric", meta.get("metric", "N/A"))
    else:
        st.warning("No model trained yet.")


# ── Page: Model Overview ──────────────────────────────────────────────────────
if page == "📊 Model Overview":
    st.header("Model Overview")

    if not (MODEL_DIR / "metadata.json").exists():
        st.info("No model trained yet. Use the API to train via POST /train")
        st.stop()

    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Best Model", meta["best_model"])
    with col2: st.metric("CV Score", f"{meta['best_score']:.4f}")
    with col3: st.metric("Features", len(meta.get("feature_names", [])))
    with col4: st.metric("Training Time", f"{meta.get('training_duration_sec', 0):.1f}s")

    # Model comparison chart
    st.subheader("Model Comparison")
    results = meta.get("all_model_results", {})
    if results:
        df_results = pd.DataFrame([
            {"Model": k, "Score": v.get("score") or 0}
            for k, v in results.items()
            if v.get("score") is not None
        ]).sort_values("Score", ascending=True)

        fig = px.bar(
            df_results, x="Score", y="Model", orientation="h",
            title=f"All Models — {meta['metric']}",
            color="Score", color_continuous_scale="viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Best hyperparameters
    st.subheader("Best Hyperparameters")
    st.json(meta.get("best_params", {}))

    # Evaluation metrics
    st.subheader("Evaluation Metrics")
    eval_m = meta.get("eval_metrics", {})
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Accuracy", f"{eval_m.get('accuracy', 0):.4f}")
    with col2: st.metric("F1 Weighted", f"{eval_m.get('f1_weighted', 0):.4f}")
    with col3: st.metric("ROC-AUC", f"{eval_m.get('roc_auc', 'N/A')}")


# ── Page: Drift Monitor ───────────────────────────────────────────────────────
elif page == "🔍 Drift Monitor":
    st.header("Drift Monitoring")

    drift_path = MODEL_DIR / "drift_report.json"
    if not drift_path.exists():
        st.info("No drift report yet. Upload production data to run a check.")
        st.stop()

    with open(drift_path) as f:
        report = json.load(f)

    # Summary banner
    if report.get("dataset_drift_detected"):
        st.error(f"⚠️ **DRIFT DETECTED** — {report['n_features_drifted']} features drifted. Retraining recommended!")
    else:
        st.success(f"✅ No significant drift detected ({report['drift_share']:.1%} features drifted)")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Production Samples", report["n_production_samples"])
    with col2: st.metric("Features Checked", report["n_features_checked"])
    with col3: st.metric("Features Drifted", report["n_features_drifted"])
    with col4: st.metric("Drift Share", f"{report['drift_share']:.1%}")

    st.caption(f"Report generated: {report['timestamp']}")

    # Feature-level drift table
    st.subheader("Feature-Level Drift Results")
    feat_data = []
    for feat, r in report.get("feature_results", {}).items():
        feat_data.append({
            "Feature": feat,
            "PSI": r["psi"],
            "KS p-value": r["ks_pvalue"],
            "Ref Mean": r["ref_mean"],
            "Prod Mean": r["prod_mean"],
            "Drifted": "⚠️ YES" if r["drifted"] else "✅ NO",
        })

    df_feat = pd.DataFrame(feat_data)
    st.dataframe(
        df_feat.style.apply(
            lambda row: ["background-color: #ffeeee" if "YES" in str(row["Drifted"]) else "" for _ in row],
            axis=1,
        ),
        use_container_width=True,
    )

    # PSI chart
    st.subheader("PSI by Feature")
    fig = px.bar(
        df_feat.sort_values("PSI", ascending=False),
        x="Feature", y="PSI",
        color="PSI",
        color_continuous_scale="RdYlGn_r",
        title="Population Stability Index (PSI) — threshold: 0.2",
    )
    fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="PSI Threshold (0.2)")
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Mean shift chart
    st.subheader("Mean Shift: Reference vs Production")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Reference Mean", x=df_feat["Feature"], y=df_feat["Ref Mean"]))
    fig2.add_trace(go.Bar(name="Production Mean", x=df_feat["Feature"], y=df_feat["Prod Mean"]))
    fig2.update_layout(barmode="group", height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    # Historical reports
    st.subheader("Historical Reports")
    report_files = sorted(REPORTS_DIR.glob("drift_report_*.json"), reverse=True)
    if report_files:
        history = []
        for rp in report_files[:20]:
            with open(rp) as f:
                r = json.load(f)
            history.append({
                "Timestamp": r["timestamp"],
                "Samples": r["n_production_samples"],
                "Drifted Features": r["n_features_drifted"],
                "Drift Share": f"{r['drift_share']:.1%}",
                "Alert": "⚠️" if r["dataset_drift_detected"] else "✅",
            })
        st.dataframe(pd.DataFrame(history), use_container_width=True)


# ── Page: Upload & Check ──────────────────────────────────────────────────────
elif page == "📁 Upload & Check":
    st.header("Upload Production Data & Check Drift")

    st.info(
        "Upload a CSV with the same feature columns as your training data (no target column needed). "
        "Drift will be checked against the training reference distribution."
    )

    uploaded = st.file_uploader("Production CSV", type=["csv"])

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Max samples to use", 50, 5000, 500)
    with col2:
        simulate_drift = st.checkbox("Simulate drift (for testing)", value=False)

    if st.button("Run Drift Check", type="primary") and uploaded:
        import tempfile, sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from drift.monitor import DriftMonitor

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        prod_df = pd.read_csv(tmp_path).head(n_samples)

        if simulate_drift:
            for col in prod_df.select_dtypes(include="number").columns[:3]:
                prod_df[col] += prod_df[col].std() * 2.5
            st.warning("Drift simulation applied to first 3 numeric features.")

        prod_df.to_csv(tmp_path, index=False)

        with st.spinner("Running drift analysis..."):
            try:
                monitor = DriftMonitor(model_dir=str(MODEL_DIR), reports_dir=str(REPORTS_DIR))
                report = monitor.check_drift(tmp_path)

                if report.get("dataset_drift_detected"):
                    st.error(f"⚠️ Drift detected in {report['n_features_drifted']} features!")
                else:
                    st.success("✅ No significant drift detected.")

                st.json({k: v for k, v in report.items() if k != "feature_results"})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ── Page: Model Card ──────────────────────────────────────────────────────────
elif page == "📋 Model Card":
    st.header("Model Card")
    card_path = MODEL_DIR / "MODEL_CARD.md"
    if not card_path.exists():
        st.info("No model card found. Train a model first.")
    else:
        st.markdown(card_path.read_text())
