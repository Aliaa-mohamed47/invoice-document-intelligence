# ai/model/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline — used directly by inference_api
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
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
def extract_entities(tokens, bboxes, model, tokenizer,
                     img_width=PAGE_WIDTH, img_height=PAGE_HEIGHT):
    """
    Returns structured fields:
    company, date, total, address
    """
    buckets = {f: [] for f in FIELDS}

    for item in predict(tokens, bboxes, model, tokenizer,
                        img_width, img_height):

        if item["label"] == "O":
            continue

        parts = item["label"].split("-")

        if len(parts) < 2:
            continue

        key = FIELD_MAP.get(parts[1])

        if key:
            buckets[key].append(item["token"])

    return {
        k: " ".join(v) if v else None
        for k, v in buckets.items()
    }


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