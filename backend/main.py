# backend/main.py
# فاطمة — Backend Service
# ─────────────────────────────────────────────────────────────
# Endpoints:
#   POST   /api/invoices/upload        ← upload + call inference API
#   GET    /api/invoices               ← list all invoices
#   GET    /api/invoices/{id}          ← get single invoice
#   GET    /api/invoices/search        ← search by company
#   PUT    /api/invoices/{id}/confirm  ← confirm + edit fields
#   DELETE /api/invoices/{id}          ← delete invoice
#   GET    /health                     ← health check
# ─────────────────────────────────────────────────────────────

import os
import httpx
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from database import engine, get_db, Base
from models import Invoice, InvoiceStatus

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice-backend")

# ── Config ────────────────────────────────────────────────────────────────────
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
# ── Create tables on startup ──────────────────────────────────────────────────
Base.metadata.create_all(bind=engine)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Invoice Intelligence — Backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ConfirmRequest(BaseModel):
    company: Optional[str] = None
    date:    Optional[str] = None
    total:   Optional[str] = None
    address: Optional[str] = None


class InvoiceOut(BaseModel):
    id:                int
    invoice_id:        str
    original_filename: Optional[str]
    s3_key:            Optional[str]
    company:           Optional[str]
    date:              Optional[str]
    total:             Optional[str]
    address:           Optional[str]
    company_score:     Optional[float]
    date_score:        Optional[float]
    total_score:       Optional[float]
    address_score:     Optional[float]
    processing_time:   Optional[float]
    status:            str
    created_at:        Optional[str]

    class Config:
        from_attributes = True


# ── Helper ────────────────────────────────────────────────────────────────────
def invoice_to_dict(inv: Invoice) -> dict:
    return {
        "id":                inv.id,
        "invoice_id":        inv.invoice_id,
        "original_filename": inv.original_filename,
        "s3_key":            inv.s3_key,
        "company":           inv.company,
        "date":              inv.date,
        "total":             inv.total,
        "address":           inv.address,
        "company_score":     inv.company_score,
        "date_score":        inv.date_score,
        "total_score":       inv.total_score,
        "address_score":     inv.address_score,
        "processing_time":   inv.processing_time,
        "status":            inv.status.value if inv.status else "PENDING",
        "created_at":        inv.created_at.isoformat() if inv.created_at else None,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"
    return {
        "status": "ok",
        "database": db_status,
        "inference_api": INFERENCE_API_URL,
    }


@app.post("/api/invoices/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    db:   Session    = Depends(get_db),
):
    """
    1. Receives invoice file from dashboard
    2. Forwards it to inference-api /extract
    3. Saves result to DB
    4. Returns saved invoice record
    """
    allowed = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    file_bytes = await file.read()

    # ── Call inference API ───────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{INFERENCE_API_URL}/extract",
                files={"file": (file.filename, file_bytes, file.content_type)},
            )
            response.raise_for_status()
            result = response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference API timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Inference API error: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cannot reach inference API: {e}")

    logger.info(f"Inference result: {result}")

    # ── Save to DB ───────────────────────────────────────────────────────────
    fields  = result.get("extracted_fields", {})
    scores  = result.get("confidence_scores", {})

    invoice = Invoice(
        invoice_id        = result.get("invoice_id"),
        original_filename = file.filename,
        s3_key            = result.get("s3_key"),
        company           = fields.get("company"),
        date              = fields.get("date"),
        total             = fields.get("total"),
        address           = fields.get("address"),
        company_score     = scores.get("company"),
        date_score        = scores.get("date"),
        total_score       = scores.get("total"),
        address_score     = scores.get("address"),
        processing_time   = result.get("processing_time_ms"),
        status            = InvoiceStatus.PENDING,
    )

    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    logger.info(f"Saved invoice {invoice.invoice_id} (db id={invoice.id})")
    return invoice_to_dict(invoice)


@app.get("/api/invoices")
def list_invoices(db: Session = Depends(get_db)):
    """Return all invoices ordered by newest first."""
    invoices = db.query(Invoice).order_by(Invoice.created_at.desc()).all()
    return [invoice_to_dict(i) for i in invoices]


@app.get("/api/invoices/search")
def search_invoices(
    company: str = Query(..., description="Company name to search for"),
    db: Session  = Depends(get_db),
):
    """Search invoices by company name (case-insensitive partial match)."""
    invoices = (
        db.query(Invoice)
        .filter(Invoice.company.ilike(f"%{company}%"))
        .order_by(Invoice.created_at.desc())
        .all()
    )
    return [invoice_to_dict(i) for i in invoices]


@app.get("/api/invoices/{invoice_id}")
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice_to_dict(invoice)


@app.put("/api/invoices/{invoice_id}/confirm")
def confirm_invoice(
    invoice_id: int,
    body: ConfirmRequest,
    db:   Session = Depends(get_db),
):
    """
    Confirm invoice — optionally update extracted fields before saving.
    """
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    # Apply any user edits from the dashboard
    if body.company is not None: invoice.company = body.company
    if body.date    is not None: invoice.date    = body.date
    if body.total   is not None: invoice.total   = body.total
    if body.address is not None: invoice.address = body.address

    invoice.status = InvoiceStatus.CONFIRMED
    db.commit()
    db.refresh(invoice)

    logger.info(f"Confirmed invoice {invoice.invoice_id}")
    return invoice_to_dict(invoice)


@app.delete("/api/invoices/{invoice_id}")
def delete_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    db.delete(invoice)
    db.commit()
    logger.info(f"Deleted invoice id={invoice_id}")
    return {"detail": "Deleted successfully"}