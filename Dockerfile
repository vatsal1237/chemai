# ---- RAG API Server ----
# Multi-stage-friendly Dockerfile for the FastAPI RAG application.

FROM python:3.13-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level build dependencies and poppler-utils for pdf2image
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the image
COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
