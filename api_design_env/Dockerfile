FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/api_design_env/

ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=4
ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn api_design_env.server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS}
