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

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

COPY . /app

EXPOSE 8000

ENV PORT=8000

CMD ["python", "main.py"]
