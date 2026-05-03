import os
import uuid
import logging
import boto3
import httpx
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from typing import Optional

from database import save_invoice_to_db, get_all_invoices_from_db, get_invoice_by_id, delete_invoice_from_db
import logging
logger = logging.getLogger(__name__)
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL")  # URL الـ inference service

s3_client = boto3.client('s3')
app = FastAPI(title="Invoice Intelligence API")


handler = Mangum(app)

class ConfirmRequest(BaseModel):
    company: Optional[str] = None
    date: Optional[str] = None
    total: Optional[str] = None
    address: Optional[str] = None

@app.get("/health")
def health_check():
    return {
        "status": "alive",
        "engine": "LayoutLM-v1",
        "database": "DynamoDB",
        "inference_api": INFERENCE_API_URL or "NOT CONFIGURED"
    }

@app.post("/api/invoices/upload")
async def upload_invoice(file: UploadFile = File(...)):
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3 Bucket not configured")
    if not INFERENCE_API_URL:
        raise HTTPException(status_code=500, detail="Inference API not configured")

    file_bytes = await file.read()
    invoice_id = str(uuid.uuid4())

    # 1. رفع لـ S3
    s3_key = f"invoices/{invoice_id}_{file.filename}"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ContentType=file.content_type
        )
        logger.info(f"Uploaded to S3: {s3_key}")
    except Exception as e:
        logger.error(f"S3 Upload Failed: {e}")

    # 2. بعت الصورة لـ inference_api
    try:
        logger.info("Calling Inference API...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{INFERENCE_API_URL}/extract",
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            response.raise_for_status()
            inference_result = response.json()
        logger.info(f"Inference result: {inference_result}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference API timeout")
    except Exception as e:
        logger.error(f"Inference API Error: {e}")
        raise HTTPException(status_code=502, detail=f"AI Processing failed: {str(e)}")

    # 3. حفظ في DynamoDB
    fields = inference_result.get("extracted_fields", {})
    confidence_scores = inference_result.get("confidence_scores", {})
    invoice_data = {
        "id": invoice_id,
        "filename": file.filename,
        "s3_key": s3_key,
        "company": fields.get("company", "N/A"),
        "date": fields.get("date", "N/A"),
        "total": fields.get("total", "N/A"),
        "address": fields.get("address", "N/A"),
        "company_score": str(confidence_scores.get("company", 0)),
        "date_score": str(confidence_scores.get("date", 0)),
        "total_score": str(confidence_scores.get("total", 0)),
        "address_score": str(confidence_scores.get("address", 0)),
        "status": "PENDING",
        "created_at": datetime.now().isoformat()
    }

    if save_invoice_to_db(invoice_data):
        return invoice_data

    raise HTTPException(status_code=500, detail="Database save failed")

@app.get("/api/invoices")
def list_invoices():
    invoices = get_all_invoices_from_db()
    return sorted(invoices, key=lambda x: x.get('created_at', ''), reverse=True)

@app.get("/api/invoices/{invoice_id}")
def get_invoice(invoice_id: str):
    invoice = get_invoice_by_id(invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice

@app.put("/api/invoices/{invoice_id}/confirm")
async def confirm_invoice(invoice_id: str, body: ConfirmRequest):
    invoice = get_invoice_by_id(invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    invoice.update({
        "status": "CONFIRMED",
        "company": body.company or invoice.get("company"),
        "date": body.date or invoice.get("date"),
        "total": body.total or invoice.get("total"),
        "address": body.address or invoice.get("address"),
    })
    save_invoice_to_db(invoice)
    return invoice

@app.delete("/api/invoices/{invoice_id}")
async def delete_invoice(invoice_id: str):
    invoice = get_invoice_by_id(invoice_id)
    
    if invoice and invoice.get("s3_key") and S3_BUCKET:
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=invoice["s3_key"])
            logger.info(f"Deleted from S3: {invoice['s3_key']}")
        except Exception as e:
            logger.error(f"S3 Delete failed: {e}")
    
    delete_invoice_from_db(invoice_id)
    return {"deleted": invoice_id}