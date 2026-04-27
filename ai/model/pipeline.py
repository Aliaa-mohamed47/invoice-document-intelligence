# ai/model/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline — used directly by inference_api
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification

# ✅ import from centralized config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")

FIELDS = ["company", "date", "total", "address"]

FIELD_MAP = {
    "COMPANY": "company",
    "DATE": "date",
    "TOTAL": "total",
    "ADDRESS": "address",
}


# ── Bounding box normalization ────────────────────────────────────────────────
def normalize_bbox(bbox, width=PAGE_WIDTH, height=PAGE_HEIGHT):
    """
    Uses actual image dimensions instead of assuming fixed size.
    """
    x0, y0, x1, y1 = bbox

    return [
        max(0, min(int(1000 * x0 / width), 1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width), 1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LayoutLMForTokenClassification.from_pretrained(model_path)

    model.eval()
    return model, tokenizer


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(tokens, bboxes, model, tokenizer,
            img_width=PAGE_WIDTH, img_height=PAGE_HEIGHT):

    norm = [
        normalize_bbox(b, img_width, img_height)
        for b in bboxes
    ]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    word_ids = encoding.word_ids()

    bbox_tensor = [
        norm[wid] if wid is not None else [0, 0, 0, 0]
        for wid in word_ids
    ]

    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    results = []
    seen = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue

        seen.add(wid)

        results.append({
            "token": tokens[wid],
            "label": ID2LABEL[preds[idx]],
            "bbox": bboxes[wid],
        })

    return results


# ── Entity extraction ─────────────────────────────────────────────────────────
def regex_fallback(field: str, tokens: list[str]) -> str | None:
    text = " ".join(tokens)
    patterns = {
        "date":    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b',
        "total":   r'(?:total|amount|grand)[^\d]{0,10}(\d+[\.,]\d{2})',
        "company": r'^([A-Z][A-Za-z0-9\s&\.\,]+?)(?:\n|$)',
        "address": r'(\d+[,\s]+[A-Za-z\s]+(?:street|st|road|rd|ave|jalan|jln)[^\n]*)',
    }
    pat = patterns.get(field)
    if not pat:
        return None
    m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_entities(tokens, bboxes, model, tokenizer,
                     img_width=PAGE_WIDTH, img_height=PAGE_HEIGHT,
                     confidence_threshold=0.70):
    """
    Returns structured fields with confidence scores.
    Falls back to regex if model confidence is below threshold.
    """
    buckets      = {f: [] for f in FIELDS}
    conf_buckets = {f: [] for f in FIELDS}

    raw_preds = predict(tokens, bboxes, model, tokenizer, img_width, img_height)

    # collect logits per token for confidence
    norm = [normalize_bbox(b, img_width, img_height) for b in bboxes]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    word_ids = encoding.word_ids()
    bbox_tensor = [norm[wid] if wid is not None else [0,0,0,0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1)[0]

    seen = set()
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)

        label     = ID2LABEL[torch.argmax(probs[idx]).item()]
        conf      = probs[idx].max().item()

        if label == "O":
            continue

        parts = label.split("-")
        if len(parts) < 2:
            continue

        key = FIELD_MAP.get(parts[1])
        if key:
            buckets[key].append(tokens[wid])
            conf_buckets[key].append(conf)

    result = {}
    for field in FIELDS:
        if buckets[field]:
            avg_conf = sum(conf_buckets[field]) / len(conf_buckets[field])
            value    = " ".join(buckets[field])

            if avg_conf < confidence_threshold:
                fallback = regex_fallback(field, tokens)
                if fallback:
                    value    = fallback
                    avg_conf = 0.60  # regex confidence ثابتة

            result[field] = {"value": value, "confidence": round(avg_conf, 2)}
        else:
            # model didn't find anything — try regex directly
            fallback = regex_fallback(field, tokens)
            result[field] = {
                "value":      fallback,
                "confidence": 0.55 if fallback else 0.0,
            }

    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokens = ["KEDAI", "GUNTING", "Date:", "01/01/2023", "Total:", "25.00"]

    bboxes = [
        [100, 100, 200, 120],
        [210, 100, 300, 120],
        [100, 200, 150, 220],
        [160, 200, 250, 220],
        [100, 300, 150, 320],
        [160, 300, 220, 320],
    ]

    img_width, img_height = 1200, 1600

    if os.path.exists(MODEL_PATH):
        model, tokenizer = load_model()

        result = extract_entities(
            tokens, bboxes, model, tokenizer,
            img_width, img_height
        )

        print("\n--- Inference Result ---")
        print(json.dumps(result, indent=2))

    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Run finetune.py first to generate saved_model")