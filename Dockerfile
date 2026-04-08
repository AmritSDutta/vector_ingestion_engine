# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    STREAMLIT_PORT=8501 \
    STREAMLIT_SCRIPT=/app/ui.py

WORKDIR /app

# minimal system deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# install Python deps (expect requirements.txt at repo root)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy source
COPY . /app

# expose only streamlit
EXPOSE ${STREAMLIT_PORT}

# start uvicorn in background, short warm-up, then run streamlit in foreground
CMD ["bash", "-lc", "\
  set -euo pipefail; \
  uvicorn_cmd=\"uvicorn app.main:app --host 127.0.0.1 --port ${PORT} --log-level info\"; \
  echo \"[container] starting FastAPI: $uvicorn_cmd\"; \
  $uvicorn_cmd & \
  UVICORN_PID=$!; \
  sleep 0.6; \
  echo \"[container] starting Streamlit on ${STREAMLIT_PORT}\"; \
  exec streamlit run app/ui.py --server.port ${STREAMLIT_PORT} --server.headless true --server.enableCORS false --server.runOnSave false \
"]
