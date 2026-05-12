import json
import os
import re
import sys
import torch
import io
from PIL import Image
import pytesseract
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from finetuning.config import LABEL_LIST, LABEL2ID, ID2LABEL, MAX_SEQ_LENGTH

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "saved_model"))
FIELDS     = ["company", "date", "total", "address"]
FIELD_MAP  = {"COMPANY": "company", "DATE": "date", "TOTAL": "total", "ADDRESS": "address"}

def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")

    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
    model     = LayoutLMForTokenClassification.from_pretrained(model_path, use_safetensors=True)
    model.eval()
    return model, tokenizer

def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]

def predict_invoice(file_bytes, model, tokenizer):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = image.size

    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    tokens = []
    bboxes = []
    
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        if word != "": 
            tokens.append(word)
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            bboxes.append([x, y, x + w, y + h])

    if not tokens:
        return {"extracted_fields": {}}

    raw_results = extract_entities(tokens, bboxes, model, tokenizer, width, height)
    
    clean_result = {field: res["value"] for field, res in raw_results.items()}
    return {"extracted_fields": clean_result}

def regex_fallback(field: str, tokens: list[str]) -> str | None:
    text = " ".join(tokens)
    patterns = {
        "date":    r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\b',
        "total":   r'(?:TOTAL|AMOUNT|GRAND)[^\d]{0,15}([\d,]+\.\d{2})',
        "company": r'^([A-Z][A-Z\s&\.]{3,40})',
        "address": r'(\d+[,\s]+[\w\s]+(?:STREET|ST|ROAD|RD|AVE|JALAN|JLN|LANE)[^\n]{0,60})',
    }
    pat = patterns.get(field)
    if not pat: return None
    m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None

def extract_entities(tokens, bboxes, model, tokenizer, img_width, img_height, confidence_threshold=0.50):
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
    bbox_tensor = [norm_bboxes[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    preds = torch.argmax(probs, dim=-1).tolist()

    extracted_spans = {f: [] for f in FIELDS}
    current_span = None
    seen_word_ids = set()

    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen_word_ids: continue
        seen_word_ids.add(wid)
        label = ID2LABEL[preds[idx]]
        conf  = probs[idx].max().item()

        if label.startswith("B-"):
            if current_span: extracted_spans[current_span["field"]].append(current_span)
            field_key = FIELD_MAP.get(label.split("-")[1])
            if field_key: current_span = {"field": field_key, "tokens": [tokens[wid]], "confidences": [conf]}
            else: current_span = None
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

    if current_span: extracted_spans[current_span["field"]].append(current_span)

    final_result = {}
    for field in FIELDS:
        candidates = extracted_spans[field]
        best_value, best_conf = None, 0.0
        if candidates:
            for span in candidates:
                avg_conf = sum(span["confidences"]) / len(span["confidences"])
                if avg_conf > best_conf:
                    best_conf = avg_conf
                    best_value = " ".join(span["tokens"])

        if best_value and best_conf >= confidence_threshold:
            final_result[field] = {"value": best_value, "confidence": round(best_conf, 2)}
        else:
            fallback = regex_fallback(field, tokens)
            final_result[field] = {"value": fallback, "confidence": 0.45 if fallback else 0.0}
    return final_result