FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Python project
COPY pyproject.toml ./
COPY src/ ./src/
COPY config.yaml ./
COPY data/ ./data/

# Install Python dependencies including backend
RUN pip install --no-cache-dir -e ".[dev]"
RUN pip install --no-cache-dir -r backend-requirements.txt

EXPOSE 8000

# Start backend
CMD ["python", "-m", "ai_quant_nautilus.backend"]
