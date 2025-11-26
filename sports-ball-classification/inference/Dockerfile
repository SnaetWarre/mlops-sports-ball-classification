FROM python:3.10-slim

WORKDIR /app

# Install system deps for Pillow and TensorFlow basic runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create directory for SQLite database
RUN mkdir -p /app/data

EXPOSE 8000

ENV PORT=8000
ENV DATABASE_URL=sqlite:////app/data/predictions.db

CMD ["python", "main.py"]
