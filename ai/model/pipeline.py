# ai/model/pipeline.py  ── FIXED VERSION
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")
FIELDS     = ["company", "date", "total", "address"]
FIELD_MAP  = {
    "COMPANY": "company",
    "DATE":    "date",
    "TOTAL":   "total",
    "ADDRESS": "address",
}


def normalize_bbox(bbox, width, height):
    """Always use actual image dimensions."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]


def load_model(model_path=MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = LayoutLMForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict(tokens, bboxes, model, tokenizer, img_width, img_height):
    norm = [normalize_bbox(b, img_width, img_height) for b in bboxes]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    word_ids    = encoding.word_ids()
    bbox_tensor = [norm[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    preds   = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    results = []
    seen    = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        results.append({
            "token": tokens[wid],
            "label": ID2LABEL[preds[idx]],
            "bbox":  bboxes[wid],
        })

    return results


def regex_fallback(field: str, tokens: list[str]) -> str | None:
    text = " ".join(tokens)
    patterns = {
        "date":    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
        "total":   r'(?:TOTAL|AMOUNT|GRAND)[^\d]{0,15}([\d,]+\.\d{2})',
        "company": r'^([A-Z][A-Z\s&\.]{3,40})',
        "address": r'(\d+[,\s]+[\w\s]+(?:STREET|ST|ROAD|RD|AVE|JALAN|JLN|LANE)[^\n]{0,60})',
    }
    pat = patterns.get(field)
    if not pat:
        return None
    m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_entities(tokens, bboxes, model, tokenizer,
                     img_width, img_height,
                     confidence_threshold=0.50):   # FIX: lowered from 0.70
    """
    Returns structured fields with confidence scores.
    """
    buckets      = {f: [] for f in FIELDS}
    conf_buckets = {f: [] for f in FIELDS}

    norm = [normalize_bbox(b, img_width, img_height) for b in bboxes]
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    word_ids    = encoding.word_ids()
    bbox_tensor = [norm[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    seen  = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)

        label = ID2LABEL[torch.argmax(probs[idx]).item()]
        conf  = probs[idx].max().item()

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

            # FIX: deduplicate repeated tokens (date bug)
            parts_dedup = []
            for p in value.split():
                if p not in parts_dedup:
                    parts_dedup.append(p)
            value = " ".join(parts_dedup)

            if avg_conf < confidence_threshold:
                fallback = regex_fallback(field, tokens)
                if fallback:
                    value    = fallback
                    avg_conf = 0.50

            result[field] = {"value": value, "confidence": round(avg_conf, 2)}
        else:
            fallback = regex_fallback(field, tokens)
            result[field] = {
                "value":      fallback,
                "confidence": 0.45 if fallback else 0.0,
            }

    return result


if __name__ == "__main__":
    tokens = ["KEDAI", "GUNTING", "Date:", "01/01/2023", "Total:", "25.00"]
    bboxes = [
        [100, 100, 200, 120], [210, 100, 300, 120],
        [100, 200, 150, 220], [160, 200, 250, 220],
        [100, 300, 150, 320], [160, 300, 220, 320],
    ]
    img_width, img_height = 1200, 1600

    if os.path.exists(MODEL_PATH):
        model, tokenizer = load_model()
        result = extract_entities(tokens, bboxes, model, tokenizer, img_width, img_height)
        print("\n--- Inference Result ---")
        print(json.dumps(result, indent=2))
    else:
        print(f"Model not found at {MODEL_PATH}")