# Multi-stage build: compile wheels in one image, ship only the runtime.
# Keeps the final image free of build toolchains it will never use.

FROM python:3.12-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


FROM python:3.12-slim AS runtime

# Don't run as root. If the container is ever compromised, the blast radius
# is one unprivileged account rather than the whole filesystem.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval/ ./eval/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Bake the embedding model into the image so the first request isn't a
# 90 MB download. Trades image size for predictable cold-start latency.
ENV HF_HOME=/home/appuser/.cache/huggingface \
    CHROMA_DIR=/app/.chroma

# appuser has to own every path it writes at runtime. WORKDIR creates /app as
# root, and the COPY --chown lines above only cover the files copied into it --
# so Chroma could not mkdir its persistence directory and startup died with
# PermissionError on /app/.chroma. Create it and hand over /app before dropping
# privileges.
RUN mkdir -p "$HF_HOME" "$CHROMA_DIR" \
    && chown -R appuser:appuser /home/appuser /app

USER appuser
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The vector store lives here; mount a volume to keep an index across restarts.
VOLUME ["/app/.chroma"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
