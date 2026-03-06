# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Installs all Python dependencies into a virtual environment.
# This stage is discarded — only the venv is copied to the final image.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv (fast installer)
RUN pip install --no-cache-dir uv==0.3.5

# Copy dependency manifest first (layer cache: only re-run if deps change)
COPY pyproject.toml .
COPY src/ src/

# Create venv and install production dependencies only
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e . --no-dev


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
# Minimal production image — no build tools, no dev dependencies.
FROM python:3.11-slim AS runtime

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy only the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY --chown=appuser:appuser src/       ./src/
COPY --chown=appuser:appuser configs/   ./configs/

# Create writable directories for data, models, and logs
RUN mkdir -p data/raw data/processed data/interim models logs mlruns && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check — polls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Default: start the prediction API
CMD ["uvicorn", "ames_housing.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
