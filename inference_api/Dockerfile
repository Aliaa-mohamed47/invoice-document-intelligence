# inference_api/Dockerfile
FROM python:3.11-slim

# Install system dependencies (Tesseract + Poppler for pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference service code
COPY main.py .

# Copy AI config (needed for label imports)
COPY ../ai/finetuning/config.py ./ai/finetuning/config.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]