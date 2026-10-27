# ============================================================================
# SINYAL — Production / Competition Demo Dockerfile
# Compatible with Hugging Face Spaces (default port 7860), Render, Railway,
# Fly.io, and Google Cloud Run (via $PORT environment variable).
# ============================================================================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy application source, frozen Phase 2 model artifacts, reports, and web assets
COPY src/ /app/src/
COPY web/ /app/web/
COPY models/ /app/models/
COPY reports/phase2/ /app/reports/phase2/
COPY README.md /app/README.md

# Create non-root user (required by Hugging Face Spaces and security best practices)
RUN useradd -m -u 1000 sinyaluser && chown -R sinyaluser:sinyaluser /app
USER sinyaluser

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

