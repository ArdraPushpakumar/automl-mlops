#!/bin/bash
# entrypoint.sh — starts API or Dashboard based on CMD

set -e

case "$1" in
  api)
    echo "Starting FastAPI server..."
    cd /app
    exec uvicorn phase2_api.api.main:app \
      --host 0.0.0.0 \
      --port ${PORT:-8000} \
      --workers 2
    ;;
  dashboard)
    echo "Starting Streamlit dashboard..."
    cd /app
    exec streamlit run phase3_drift/dashboard/app.py \
      --server.port ${PORT:-8501} \
      --server.address 0.0.0.0 \
      --server.headless true
    ;;
  train)
    echo "Running AutoML training..."
    cd /app
    python -m phase1_core.run_phase1
    ;;
  *)
    echo "Usage: $0 {api|dashboard|train}"
    exit 1
    ;;
esac
