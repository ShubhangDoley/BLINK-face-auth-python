# Production-grade Dockerfile for BLINK Biometric FastAPI Backend
FROM python:3.10-slim

# Set environment defaults
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies required by OpenCV (image/video processing tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Expose server port
EXPOSE 8000

WORKDIR /app/src

# Run uvicorn server
CMD uvicorn server:app --host 0.0.0.0 --port 8000
