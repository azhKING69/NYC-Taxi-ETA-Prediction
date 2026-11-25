FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (libgomp is required for LightGBM OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy model artifact and serving application
COPY model_artifact/ /app/model_artifact/
COPY serving/ /app/serving/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
