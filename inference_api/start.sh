#!/bin/bash
echo "Downloading model from S3..."
aws s3 cp s3://invoice-intelligence-storage-2026/ai/model/saved_model \
    /app/model/saved_model --recursive --region eu-north-1
echo "Model ready — starting server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000