FROM python:3.11-slim AS builder

WORKDIR /build
COPY api_design_env/server/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY api_design_env/ /app/api_design_env/
COPY inference.py /app/inference.py

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn api_design_env.server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --timeout-keep-alive 120"]
