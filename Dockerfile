# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables for Python and Port
# PYTHONUNBUFFERED=1 ensures logs are streamed in real-time
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose the port (Cloud Run sets the PORT env var at runtime)
EXPOSE 8000

# Default command to run the FastAPI application
# Using shell form to allow environment variable expansion for $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
