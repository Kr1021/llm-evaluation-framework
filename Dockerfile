# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps needed by sentence-transformers / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create outputs directory so reports have somewhere to land
RUN mkdir -p outputs

# ── Runtime ────────────────────────────────────────────────────────────────
EXPOSE 8000

# Default: run the FastAPI server
# Override CMD to run the CI pipeline instead:
#   docker run ... python pipelines/ci_eval_pipeline.py --input ... --output ...
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
