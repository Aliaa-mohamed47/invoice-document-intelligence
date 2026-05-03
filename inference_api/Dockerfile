FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY inference_api/requirements.txt .
RUN pip install --no-cache-dir --timeout 300 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt
    
COPY inference_api/main.py .
COPY ai/finetuning/config.py ./ai/finetuning/config.py
COPY inference_api/start.sh .
RUN chmod +x start.sh

ENV MODEL_PATH=/app/model/saved_model
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["./start.sh"]