
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

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ai.finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice-api")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/saved_model")
S3_BUCKET  = os.getenv("S3_BUCKET", "invoice-intelligence-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

FIELDS = ["company", "date", "total", "address"]
FIELD_MAP = {
    "COMPANY": "company",
    "DATE":    "date",
    "TOTAL":   "total",
    "ADDRESS": "address",
}

model     = None
tokenizer = None


def load_model_on_startup():
    global model, tokenizer
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model from S3...")
        s3 = boto3.client("s3", region_name="eu-north-1")
        os.makedirs(MODEL_PATH, exist_ok=True)
        bucket = "invoice-intelligence-storage-2026"
        prefix = "ai/model/saved_model"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local_path = os.path.join(MODEL_PATH, os.path.relpath(key, prefix))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket, key, local_path)
        logger.info("Model downloaded from S3")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
    model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Model loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_on_startup()
    yield


app = FastAPI(
    title="Invoice Intelligence API",
    description="Extract structured invoice fields using LayoutLM",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def upload_to_s3(file_bytes: bytes, filename: str) -> str | None:
    try:
        s3  = boto3.client("s3", region_name=AWS_REGION)
        key = f"invoices/{uuid.uuid4().hex}/{filename}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)
        logger.info(f"Uploaded to S3: {key}")
        return key
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


def extract_tokens_from_image(image: Image.Image):
    """
    Returns (tokens, bboxes, img_width, img_height).
    We return actual image dimensions so normalize_bbox works correctly.
    """
    img_width, img_height = image.size        

    img_np = np.array(image.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    coords = np.column_stack(np.where(gray > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.5:
            h, w = gray.shape
            M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    data = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT
    )

    tokens, bboxes = [], []
    for i, text in enumerate(data["text"]):
        if not text.strip():
            continue
        conf = int(data["conf"][i])
        if conf < 30:         
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append(text.strip())
        bboxes.append([x, y, x + w, y + h])

    logger.debug(f"OCR extracted {len(tokens)} tokens")
    return tokens, bboxes, img_width, img_height



def normalize_bbox(bbox, w, h):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / w), 1000)),
        max(0, min(int(1000 * y0 / h), 1000)),
        max(0, min(int(1000 * x1 / w), 1000)),
        max(0, min(int(1000 * y1 / h), 1000)),
    ]


def fallback_extraction(tokens: list) -> dict:
    text = " ".join(tokens)

    date_patterns = [
        r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
        r'\b(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4})\b',
    ]
    date = None
    for pat in date_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            date = m.group(1)
            break

    total = None
    total_patterns = [
        r'(?:TOTAL|AMOUNT|GRAND\s*TOTAL)[^\d]{0,15}(RM\s*)?(\d[\d,]*\.\d{2})',
        r'(?:RM\s*)(\d[\d,]*\.\d{2})\s*$',
        r'\b(\d[\d,]*\.\d{2})\b',
    ]
    for pat in total_patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            total = m.group(m.lastindex)
            break

    company = None
    top_tokens = tokens[:max(1, len(tokens) // 3)]
    top_text   = " ".join(top_tokens)
    caps_m     = re.search(r'\b([A-Z][A-Z\s&\.]{3,40})\b', top_text)
    if caps_m:
        company = caps_m.group(1).strip()

    address = None
    addr_m  = re.search(
        r'(\d+[,\s]+[\w\s]+(?:STREET|ST|ROAD|RD|AVE|AVENUE|JALAN|JLN|LANE|LN)[^\n]{0,60})',
        text, re.IGNORECASE
    )
    if addr_m:
        address = addr_m.group(1).strip()

    return {
        "company":       company,
        "date":          date,
        "total":         total,
        "address":       address,
        "company_score": 0.45 if company  else 0.0,
        "date_score":    0.50 if date     else 0.0,
        "total_score":   0.55 if total    else 0.0,
        "address_score": 0.45 if address  else 0.0,
    }


def run_inference(tokens: list, bboxes: list, img_width: int, img_height: int) -> dict:
    if model is None or tokenizer is None:
        return {
            "company": "MOCK COMPANY", "date": "01/01/2024",
            "total": "100.00",         "address": "123 Mock Street",
            "company_score": 0.91,     "date_score": 0.88,
            "total_score": 0.95,       "address_score": 0.72,
        }

    norm_bboxes = [normalize_bbox(b, img_width, img_height) for b in bboxes]

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

    logits    = outputs.logits[0]
    probs     = torch.softmax(logits, dim=-1)
    pred_ids  = torch.argmax(logits, dim=-1).tolist()
    max_probs = probs.max(dim=-1).values.tolist()

    entities      = {f: [] for f in FIELDS}
    entity_probs  = {f: [] for f in FIELDS}
    seen          = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)

        label = ID2LABEL.get(pred_ids[idx], "O")
        if label == "O":
            continue

        _, etype = label.split("-", 1)
        key      = FIELD_MAP.get(etype)
        if key:
            entities[key].append(tokens[wid])
            entity_probs[key].append(max_probs[idx])

    result = {}
    for f in FIELDS:
        if entities[f]:
            raw_value = " ".join(entities[f])
            avg_conf  = sum(entity_probs[f]) / len(entity_probs[f])

            if avg_conf >= 0.50:
                result[f]            = raw_value
                result[f"{f}_score"] = round(avg_conf, 4)
            else:
                result[f]            = None
                result[f"{f}_score"] = round(avg_conf, 4)
        else:
            result[f]            = None
            result[f"{f}_score"] = 0.0

    if result.get("date"):
        date_val = result["date"]
        parts    = date_val.split()
        seen_p   = []
        for p in parts:
            if p not in seen_p:
                seen_p.append(p)
        result["date"] = " ".join(seen_p)

    fb = fallback_extraction(tokens)
    for f in FIELDS:
        if not result[f]:
            if fb.get(f):
                result[f]            = fb[f]
                result[f"{f}_score"] = fb.get(f"{f}_score", 0.0)

    if all(result[f] is None for f in FIELDS):
        logger.warning("Empty model output → full fallback used")
        return fallback_extraction(tokens)

    return result


def format_response(invoice_id: str, raw: dict, s3_key: str | None, start: float):
    return {
        "invoice_id":       invoice_id,
        "s3_key":           s3_key,
        "extracted_fields": {f: raw.get(f) for f in FIELDS},
        "confidence_scores":{f: raw.get(f"{f}_score", 0.0) for f in FIELDS},
        "processing_time_ms": round((time.time() - start) * 1000, 2),
        "model_mode": "mock" if model is None else "finetuned",
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "model_path":   MODEL_PATH,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    start      = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    allowed = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}",
        )

    file_bytes = await file.read()
    s3_key     = upload_to_s3(file_bytes, file.filename)

    try:
        if file.content_type == "application/pdf":
            from pdf2image import convert_from_bytes
            image = convert_from_bytes(file_bytes)[0]
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid file: {e}")

    tokens, bboxes, img_width, img_height = extract_tokens_from_image(image)

    if not tokens:
        raise HTTPException(status_code=422, detail="No text detected")

    logger.info(f"[{invoice_id}] OCR tokens: {len(tokens)}, size: {img_width}x{img_height}")

    raw    = run_inference(tokens, bboxes, img_width, img_height)
    result = format_response(invoice_id, raw, s3_key, start)

    logger.info(f"[{invoice_id}] Done in {result['processing_time_ms']} ms")
    return result


@app.post("/extract-from-s3")
async def extract_from_s3(s3_key: str):
    start      = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    try:
        s3           = boto3.client("s3", region_name=AWS_REGION)
        obj          = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        file_bytes   = obj["Body"].read()
        content_type = obj.get("ContentType", "image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3 object not found: {e}")

    if "pdf" in content_type:
        from pdf2image import convert_from_bytes
        image = convert_from_bytes(file_bytes)[0]
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    tokens, bboxes, img_width, img_height = extract_tokens_from_image(image)
    raw    = run_inference(tokens, bboxes, img_width, img_height)
    result = format_response(invoice_id, raw, s3_key, start)
    return result