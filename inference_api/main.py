# inference_api/main.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Inference Service — Invoice Document Intelligence
# Aliaa (Team Lead / Cloud & Integration)
#
# Input : PDF or image invoice
# Output: structured JSON (company, date, total, address)
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import re
import sys
import time
import uuid
import logging
from contextlib import asynccontextmanager

import boto3
import cv2
import numpy as np
import pytesseract
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

# Windows only setup for Tesseract
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ import centralized config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ai.finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice-api")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "ai/model/saved_model")
S3_BUCKET  = os.getenv("S3_BUCKET", "invoice-intelligence-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

FIELDS = ["company", "date", "total", "address"]

FIELD_MAP = {
    "COMPANY": "company",
    "DATE": "date",
    "TOTAL": "total",
    "ADDRESS": "address",
}

# ── Model state ───────────────────────────────────────────────────────────────
model = None
tokenizer = None


def load_model_on_startup():
    global model, tokenizer

    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH} — running in MOCK mode")
        return

    logger.info(f"Loading model from {MODEL_PATH} ...")

    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    logger.info("Model loaded successfully")


# ── FastAPI lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_on_startup()
    yield


app = FastAPI(
    title="Invoice Intelligence API",
    description="Extract structured invoice fields using LayoutLM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── S3 upload ─────────────────────────────────────────────────────────────────
def upload_to_s3(file_bytes: bytes, filename: str) -> str | None:
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        key = f"invoices/{uuid.uuid4().hex}/{filename}"

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=file_bytes
        )

        logger.info(f"Uploaded to S3: {key}")
        return key

    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


# ── OCR extraction ────────────────────────────────────────────────────────────
def extract_tokens_from_image(image: Image.Image):
    img = np.array(image)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.5:
            h, w = gray.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    data = pytesseract.image_to_data(
        gray,
        output_type=pytesseract.Output.DICT
    )

    tokens, bboxes = [], []

    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        tokens.append(text.strip())
        bboxes.append([x, y, x + w, y + h])

    logger.debug(f"OCR extracted {len(tokens)} tokens")
    return tokens, bboxes


# ── Bounding box normalization ───────────────────────────────────────────────
def normalize_bbox(bbox, w=PAGE_WIDTH, h=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox

    return [
        max(0, min(int(1000 * x0 / w), 1000)),
        max(0, min(int(1000 * y0 / h), 1000)),
        max(0, min(int(1000 * x1 / w), 1000)),
        max(0, min(int(1000 * y1 / h), 1000)),
    ]


# ── Fallback rule-based extraction ────────────────────────────────────────────
def fallback_extraction(tokens: list) -> dict:
    text = " ".join(tokens)

    dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
    totals = re.findall(r"(?:RM\s?)?\d+\.\d{2}", text)

    return {
        "company": tokens[0] if tokens else None,
        "date": dates[0] if dates else None,
        "total": totals[-1] if totals else None,
        "address": None,
        "company_score": 0.5,
        "date_score": 0.5,
        "total_score": 0.7,
        "address_score": 0.0,
    }


# ── Model inference ───────────────────────────────────────────────────────────
def run_inference(tokens: list, bboxes: list) -> dict:
    if model is None or tokenizer is None:
        return {
            "company": "MOCK COMPANY",
            "date": "01/01/2024",
            "total": "100.00",
            "address": "123 Mock Street",
            "company_score": 0.91,
            "date_score": 0.88,
            "total_score": 0.95,
            "address_score": 0.72,
        }

    norm_bboxes = [normalize_bbox(b) for b in bboxes]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    word_ids = encoding.word_ids()

    aligned_bboxes = [
        norm_bboxes[wid] if wid is not None else [0, 0, 0, 0]
        for wid in word_ids
    ]

    encoding["bbox"] = torch.tensor([aligned_bboxes], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)

    pred_ids = torch.argmax(logits, dim=-1).tolist()
    max_probs = probs.max(dim=-1).values.tolist()

    entities = {f: [] for f in FIELDS}
    entity_probs = {f: [] for f in FIELDS}
    seen = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue

        seen.add(wid)

        label = ID2LABEL.get(pred_ids[idx], "O")
        if label == "O":
            continue

        _, etype = label.split("-", 1)
        key = FIELD_MAP.get(etype)

        if key:
            entities[key].append(tokens[wid])
            entity_probs[key].append(max_probs[idx])

    result = {}

    for f in FIELDS:
        result[f] = " ".join(entities[f]) if entities[f] else None
        result[f"{f}_score"] = (
            round(sum(entity_probs[f]) / len(entity_probs[f]), 4)
            if entity_probs[f] else 0.0
        )

    text = " ".join(tokens)
    for f in FIELDS:
        if result[f] is None or result[f"{f}_score"] < 0.60:
            fb = fallback_extraction(tokens)
            if fb.get(f):
                result[f] = fb[f]
                result[f"{f}_score"] = fb.get(f"{f}_score", 0.50)

    if all(result[f] is None for f in FIELDS):
        logger.warning("Empty model output → full fallback used")
        return fallback_extraction(tokens)

    return result


# ── Response formatter ────────────────────────────────────────────────────────
def format_response(invoice_id: str, raw: dict, s3_key: str | None, start: float):
    return {
        "invoice_id": invoice_id,
        "s3_key": s3_key,
        "extracted_fields": {f: raw.get(f) for f in FIELDS},
        "confidence_scores": {f: raw.get(f"{f}_score", 0.0) for f in FIELDS},
        "processing_time_ms": round((time.time() - start) * 1000, 2),
        "model_mode": "mock" if model is None else "finetuned",
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    start = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    allowed = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/pdf",
    }

    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    file_bytes = await file.read()
    s3_key = upload_to_s3(file_bytes, file.filename)

    try:
        if file.content_type == "application/pdf":
            from pdf2image import convert_from_bytes
            image = convert_from_bytes(file_bytes)[0]
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")

    tokens, bboxes = extract_tokens_from_image(image)

    if not tokens:
        raise HTTPException(status_code=422, detail="No text detected")

    logger.info(f"[{invoice_id}] OCR tokens: {len(tokens)}")

    raw = run_inference(tokens, bboxes)
    result = format_response(invoice_id, raw, s3_key, start)

    logger.info(f"[{invoice_id}] Done in {result['processing_time_ms']} ms")

    return result


@app.post("/extract-from-s3")
async def extract_from_s3(s3_key: str):
    start = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)

        file_bytes = obj["Body"].read()
        content_type = obj.get("ContentType", "image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {e}")

    if "pdf" in content_type:
        from pdf2image import convert_from_bytes
        image = convert_from_bytes(file_bytes)[0]
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    tokens, bboxes = extract_tokens_from_image(image)

    raw = run_inference(tokens, bboxes)
    result = format_response(invoice_id, raw, s3_key, start)

    return result