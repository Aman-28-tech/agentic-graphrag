# ══════════════════════════════════════════════════════════════════════════════
#  Agentic GraphRAG — Production Dockerfile
#  Multi-stage build: builder (deps) → runtime (slim)
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for PyMuPDF and building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Amandeep"
LABEL description="Agentic GraphRAG — Self-Correcting Multi-Strategy RAG with Knowledge Graph"
LABEL version="2.0.0"

WORKDIR /app

# System runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/docs/ai_papers \
             /app/qdrant_storage \
             /app/logs \
             /app/model_cache

# Set environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/model_cache \
    TRANSFORMERS_CACHE=/app/model_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LLM_PROVIDER=local \
    LOCAL_MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
    QDRANT_MODE=local \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Pre-download models to bake them into the Docker image
# We temporarily unset offline mode just for this RUN command
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose ports: FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Default: run FastAPI backend
CMD ["python", "api.py"]
