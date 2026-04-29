# ai/model/pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import sys
import torch
from transformers import AutoTokenizer, LayoutLMForTokenClassification

# Ensure imports work regardless of execution directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    MAX_SEQ_LENGTH,
)

# AWS Ready: Allow overriding model path via environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "saved_model"))
FIELDS     = ["company", "date", "total", "address"]
FIELD_MAP  = {
    "COMPANY": "company",
    "DATE":    "date",
    "TOTAL":   "total",
    "ADDRESS": "address",
}


def normalize_bbox(bbox, width, height):
    """Normalize bounding box coordinates to 0-1000 scale based on actual image dimensions."""
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]


def load_model(model_path=MODEL_PATH):
    """Load tokenizer and model from the specified directory."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = LayoutLMForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def regex_fallback(field: str, tokens: list[str]) -> str | None:
    """Apply rule-based extraction if model confidence is low."""
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
                     confidence_threshold=0.50):
    """
    Extracts entities by grouping contiguous B- and I- tags, scoring them,
    and returning the highest confidence match per field.
    """
    norm_bboxes = [normalize_bbox(b, img_width, img_height) for b in bboxes]

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    word_ids    = encoding.word_ids()
    bbox_tensor = [norm_bboxes[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    preds = torch.argmax(probs, dim=-1).tolist()

    # 1. Group contiguous spans
    extracted_spans = {f: [] for f in FIELDS}
    current_span = None
    seen_word_ids = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen_word_ids:
            continue

        seen_word_ids.add(wid)
        label = ID2LABEL[preds[idx]]
        conf  = probs[idx].max().item()

        if label.startswith("B-"):
            if current_span:
                extracted_spans[current_span["field"]].append(current_span)

            field_key = FIELD_MAP.get(label.split("-")[1])
            if field_key:
                current_span = {"field": field_key, "tokens": [tokens[wid]], "confidences": [conf]}
            else:
                current_span = None

        elif label.startswith("I-") and current_span:
            field_key = FIELD_MAP.get(label.split("-")[1])
            if field_key == current_span["field"]:
                current_span["tokens"].append(tokens[wid])
                current_span["confidences"].append(conf)
            else:
                extracted_spans[current_span["field"]].append(current_span)
                current_span = None
        else:
            if current_span:
                extracted_spans[current_span["field"]].append(current_span)
                current_span = None

    if current_span:
        extracted_spans[current_span["field"]].append(current_span)

    # 2. Select best span per field or fallback to regex
    final_result = {}
    for field in FIELDS:
        candidates = extracted_spans[field]
        best_value = None
        best_conf = 0.0

        if candidates:
            # Score spans by average confidence
            for span in candidates:
                avg_conf = sum(span["confidences"]) / len(span["confidences"])
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_value = " ".join(span["tokens"])

        if best_value and best_conf >= confidence_threshold:
            final_result[field] = {"value": best_value, "confidence": round(best_conf, 2)}
        else:
            fallback = regex_fallback(field, tokens)
            final_result[field] = {
                "value": fallback,
                "confidence": 0.45 if fallback else 0.0,
            }

    return final_result


if __name__ == "__main__":
    # Local dry-run testing
    test_tokens = ["KEDAI", "GUNTING", "Date:", "01/01/2023", "Total:", "25.00"]
    test_bboxes = [
        [100, 100, 200, 120], [210, 100, 300, 120],
        [100, 200, 150, 220], [160, 200, 250, 220],
        [100, 300, 150, 320], [160, 300, 220, 320],
    ]

    try:
        loaded_model, loaded_tokenizer = load_model()
        res = extract_entities(test_tokens, test_bboxes, loaded_model, loaded_tokenizer, 1200, 1600)
        print(json.dumps(res, indent=2))
    except FileNotFoundError as e:
        print(e)