import os
import uuid
import httpx
import logging
import boto3
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum  # المحول لـ AWS Lambda
from pydantic import BaseModel
from typing import Optional

from database import save_invoice_to_db, get_all_invoices_from_db, get_invoice_by_id

# إعدادات اللوج والبيئة
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice-api")

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
INFERENCE_URL = os.getenv("INFERENCE_API_URL")

s3_client = boto3.client('s3')
app = FastAPI(title="Invoice Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# المعالج الخاص بـ AWS Lambda
handler = Mangum(app)

class ConfirmRequest(BaseModel):
    company: Optional[str] = None
    date: Optional[str] = None
    total: Optional[str] = None
    address: Optional[str] = None

@app.get("/health")
def health_check():
    return {"status": "alive", "engine": "serverless", "database": "dynamodb"}

@app.post("/api/invoices/upload")
async def upload_invoice(file: UploadFile = File(...)):
    if not S3_BUCKET or not INFERENCE_URL:
        raise HTTPException(status_code=500, detail="Cloud environment not configured")

    file_bytes = await file.read()
    invoice_id = str(uuid.uuid4())
    
    # 1. رفع الصورة لـ S3 (Free Tier: 5GB)[cite: 4]
    s3_key = f"invoices/{invoice_id}_{file.filename}"
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file_bytes, ContentType=file.content_type)
    except Exception as e:
        logger.error(f"S3 Upload Failed: {e}")
        raise HTTPException(status_code=500, detail="Storage error")

    # 2. إرسال الصورة لمحرك الـ AI[cite: 4]
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{INFERENCE_URL}/extract",
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            response.raise_for_status()
            inference_result = response.json()
    except Exception as e:
        logger.error(f"Inference API Error: {e}")
        raise HTTPException(status_code=502, detail="AI Processing failed")

    # 3. تنظيم البيانات وحفظها في DynamoDB[cite: 1]
    fields = inference_result.get("extracted_fields", {})
    invoice_data = {
        "id": invoice_id,
        "filename": file.filename,
        "s3_key": s3_key,
        "company": fields.get("company"),
        "date": fields.get("date"),
        "total": fields.get("total"),
        "address": fields.get("address"),
        "status": "PENDING",
        "created_at": datetime.now().isoformat()
    }
    
    if save_invoice_to_db(invoice_data):
        return invoice_data
    raise HTTPException(status_code=500, detail="Database save failed")

@app.get("/api/invoices")
def list_invoices():
    return get_all_invoices_from_db()

@app.get("/api/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    invoice = get_invoice_by_id(invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="Not found")
    return invoice