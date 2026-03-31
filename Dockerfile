# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for the production RAG API

# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
WORKDIR /app

# System deps needed by PyMuPDF, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Builder ───────────────────────────────────────────────────────────────────
FROM base AS builder
COPY pyproject.toml README.md ./
COPY src/ src/

# Install hatchling + all deps into a prefix (include [ui] for streamlit)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps hatchling hatch-vcs && \
    pip install --prefix=/install ".[ui]"

# ── Production ────────────────────────────────────────────────────────────────
FROM base AS production

# Non-root user
RUN useradd --create-home --shell /bin/bash rag && \
    mkdir -p /uploads && \
    chmod 777 /uploads
USER rag
WORKDIR /home/rag/app

# Copy installed packages
COPY --from=builder /install /usr/local
# Copy source (needed for editable-like imports)
COPY --chown=rag:rag src/ src/
COPY --chown=rag:rag ui/ ui/

ENV PYTHONPATH=/home/rag/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "production_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
