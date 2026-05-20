FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY fixtures ./fixtures
COPY README.md ARCHITECTURE.md IMPLEMENTATION_PLAN.md ./

ENV PYTHONPATH=/app/src \
    TRIBE_CACHE_DIR=/cache \
    TRIBE_JOBS_DIR=/jobs \
    TRIBE_FORCE_CPU=true

RUN useradd -m -u 1000 appuser \
    && mkdir -p /cache /jobs \
    && chown -R appuser:appuser /app /cache /jobs

USER appuser

EXPOSE 8000

CMD ["uvicorn", "tribev2_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
