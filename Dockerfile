# ============================================================
# Dual-Engine Banking Chatbot — Backend Docker Image
# Defaults to serving the FastAPI REST API on port 8000
# Also supports running the Streamlit UI on port 8501
# ============================================================

FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr and writing pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch and application requirements
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy codebase
COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Healthcheck targeting the FastAPI liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command: Run the FastAPI backend
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
