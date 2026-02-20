# Stage 1: dependency builder 
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps separately so they are cached independently
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: lean runtime image 
FROM python:3.11-slim AS runtime

# Metadata labels (OCI standard)
LABEL org.opencontainers.image.title="urdu-trigram-api" \
      org.opencontainers.image.description="Urdu Trigram Language Model – FastAPI microservice" \
      org.opencontainers.image.source="https://github.com/$GITHUB_REPOSITORY"

# Security: run as non-root user
RUN groupadd --system appgroup && \
    useradd  --system --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application files
COPY phase4_microservice.py .
COPY trigram_model.json     .

# Ownership hand-off
RUN chown -R appuser:appgroup /app

USER appuser

# Tell Docker which port the app listens on
EXPOSE 8000

# Lightweight health-check (curl is available in slim; fallback to python)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Environment variable defaults (can be overridden at runtime)
ENV MODEL_PATH="./trigram_model.json" \
    MAX_TOKENS=150 \
    TEMPERATURE=0.85 \
    PYTHONUNBUFFERED=1

# Start the server (uvicorn is the entrypoint; CMD can be overridden)
ENTRYPOINT ["uvicorn", "phase4_microservice:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
