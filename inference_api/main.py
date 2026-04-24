# main.py
# FastAPI Inference Service — Invoice Document Intelligence
# علياء (Team Leader / Cloud & Integration)
# ─────────────────────────────────────────────────────────
# يستقبل: ملف PDF أو صورة invoice
# يرجع:  JSON منظم (company, date, total, address)
# ─────────────────────────────────────────────────────────

import os
import io
import time
import uuid
import boto3
import torch
import re
import cv2
import numpy as np
import logging
import pytesseract
if os.name == "nt":  # Windows only
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invoice-api")

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH", "ai/model/saved_model")
S3_BUCKET    = os.getenv("S3_BUCKET",  "invoice-intelligence-bucket")
AWS_REGION   = os.getenv("AWS_REGION", "us-east-1")

LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
FIELDS     = ["company", "date", "total", "address"]
FIELD_MAP  = {"COMPANY": "company", "DATE": "date",
              "TOTAL": "total", "ADDRESS": "address"}

PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Invoice Intelligence API",
    description="Extracts structured fields from invoice PDFs/images using LayoutLM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────
model     = None
tokenizer = None

@app.on_event("startup")
def load_model():
    global model, tokenizer
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH} — running in MOCK mode")
        return
    logger.info(f"Loading model from {MODEL_PATH} ...")
    tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
    model     = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Model loaded successfully ✓")


# ── S3 Upload helper ──────────────────────────────────────────────────────────
def upload_to_s3(file_bytes: bytes, filename: str) -> str:
    """Upload file to S3 and return the S3 key."""
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        key = f"invoices/{uuid.uuid4().hex}/{filename}"
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)
        logger.info(f"Uploaded to S3: {key}")
        return key
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return None


# ── OCR + token extraction ────────────────────────────────────────────────────
def extract_tokens_from_image(image):
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    tokens, bboxes = [], []
    for i, text in enumerate(data["text"]):
        if text.strip() == "":
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        tokens.append(text.strip())
        bboxes.append([x, y, x + w, y + h])

    print("TOKENS:", tokens[:20])  # debug

    return tokens, bboxes


def normalize_bbox(bbox, w=PAGE_WIDTH, h=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / w), 1000)),
        max(0, min(int(1000 * y0 / h), 1000)),
        max(0, min(int(1000 * x1 / w), 1000)),
        max(0, min(int(1000 * y1 / h), 1000)),
    ]


# ── Model inference ───────────────────────────────────────────────────────────
def fallback_extraction(tokens):
    text = " ".join(tokens)

    date = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    totals = re.findall(r'(?:RM\s?)?\d+\.\d{2}', text)

    return {
        "company": tokens[0] if tokens else None,
        "date": date[0] if date else None,
        "total": totals[-1] if totals else None,
        "address": None,
        "company_score": 0.5,
        "date_score": 0.5,
        "total_score": 0.7,
        "address_score": 0.0,
    }


def run_inference(tokens: list, bboxes: list) -> dict:
    """Run LayoutLM model and return extracted fields."""
    if model is None or tokenizer is None:
        return {
            "company": "MOCK COMPANY",
            "date":    "01/01/2024",
            "total":   "100.00",
            "address": "123 Mock Street",
            "company_score": 0.91,
            "date_score":    0.88,
            "total_score":   0.95,
            "address_score": 0.72,
        }

    img_w = PAGE_WIDTH
    img_h = PAGE_HEIGHT
    norm_bboxes  = [normalize_bbox(b, img_w, img_h) for b in bboxes]
    bbox_tensor  = torch.tensor([norm_bboxes], dtype=torch.long)

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # align bboxes with subword tokens
    word_ids        = encoding.word_ids()
    aligned_bboxes  = [norm_bboxes[wid] if wid is not None else [0,0,0,0]
                        for wid in word_ids]
    encoding["bbox"] = torch.tensor([aligned_bboxes], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    logits   = outputs.logits[0]
    probs    = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1).tolist()
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
        key = FIELD_MAP.get(etype)
        if key:
            entities[key].append(tokens[wid])
            entity_probs[key].append(max_probs[idx])

    result = {}
    for f in FIELDS:
        result[f] = " ".join(entities[f]) if entities[f] else None
        result[f"{f}_score"] = (
            round(float(sum(entity_probs[f]) / len(entity_probs[f])), 4)
            if entity_probs[f] else 0.0
        )
    if all(result[f] is None for f in FIELDS):
        print("⚠️ Using fallback")
        return fallback_extraction(tokens)

    return result

# ── Format final JSON ─────────────────────────────────────────────────────────
def format_response(invoice_id: str, raw: dict, s3_key: str, start: float) -> dict:
    return {
        "invoice_id": invoice_id,
        "s3_key": s3_key,
        "extracted_fields": {f: raw.get(f) for f in FIELDS},
        "confidence_scores": {
            f: raw.get(f"{f}_score", 0.0) for f in FIELDS
        },
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
    """
    Main endpoint.
    Input : PDF or image file
    Output: structured JSON with extracted invoice fields
    """
    start      = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    # ── Validate file type ───────────────────────────────────────────────────
    allowed = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()

    # ── Upload to S3 ─────────────────────────────────────────────────────────
    s3_key = upload_to_s3(file_bytes, file.filename)

    # ── Convert to image ──────────────────────────────────────────────────────
    try:
        if file.content_type == "application/pdf":
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(file_bytes)
            image = pages[0]          # first page only
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not open file: {e}")

    # ── OCR ───────────────────────────────────────────────────────────────────
    tokens, bboxes = extract_tokens_from_image(image)
    if not tokens:
        raise HTTPException(status_code=422, detail="No text found in document")

    logger.info(f"[{invoice_id}] OCR tokens: {len(tokens)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    raw    = run_inference(tokens, bboxes)
    result = format_response(invoice_id, raw, s3_key, start)
    # override extracted_fields directly from raw
    result["extracted_fields"] = {
        "company": raw.get("company"),
        "date":    raw.get("date"),
        "total":   raw.get("total"),
        "address": raw.get("address"),
    }
    result["confidence_scores"] = {
        "company": raw.get("company_score", 0.0),
        "date":    raw.get("date_score",    0.0),
        "total":   raw.get("total_score",   0.0),
        "address": raw.get("address_score", 0.0),
    }

    logger.info(f"[{invoice_id}] Done in {result['processing_time_ms']}ms")
    return result


@app.post("/extract-from-s3")
async def extract_from_s3(s3_key: str):
    """
    Alternative endpoint — فاطمة تستخدمه لو الـ backend بعت الملف لـ S3 أول.
    Input : S3 key
    Output: extracted JSON
    """
    start      = time.time()
    invoice_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

    try:
        s3         = boto3.client("s3", region_name=AWS_REGION)
        obj        = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
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
    raw    = run_inference(tokens, bboxes)
    result = format_response(invoice_id, raw, s3_key, start)
    return result