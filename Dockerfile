FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements_hf.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
RUN mkdir -p drift_reports data models

CMD streamlit run drift/dashboard/app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
