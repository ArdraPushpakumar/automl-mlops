# AutoML MLOps — Production Dockerfile
# Multi-stage build: deps layer cached separately from app code

FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless requirements changes) ─────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Application layer ─────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy source code
COPY phase1_core/ ./phase1_core/
COPY phase2_api/  ./phase2_api/
COPY phase3_drift/ ./phase3_drift/
COPY entrypoint.sh .

# Create runtime directories
RUN mkdir -p models data drift_reports

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Default: start FastAPI. Override CMD to run Streamlit.
ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
