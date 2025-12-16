FROM python:3.11-slim AS base

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps (minimal) and Python deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Default values (override via env or compose)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    REQUEST_TIMEOUT_SECONDS=15 \
    TOP_N_ASSETS=50 \
    DATA_DIR=/app/data \
    PARQUET_FILENAME=market_snapshots.parquet \
    API_BASE_URL=http://localhost:8000

# Expose both API and dashboard ports (choose entrypoint via command)
EXPOSE 8000 8501

# Entrypoints are defined per service in docker-compose.yml
