import json
import torch
import os

from model import LABEL_LIST, LABEL2ID, ID2LABEL, TOKENIZER

PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")


def normalize_bbox(bbox, width=PAGE_WIDTH, height=PAGE_HEIGHT):
    """
    ✅ التعديل: بنحسب الـ width و height من الصورة الفعلية
    مش بنافترض إنها دايماً 762×1000.
    لو الصورة خارجية بأبعاد مختلفة، هيتحسب صح.
    """
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]


def load_model(model_path=MODEL_PATH):
    tokenizer = TOKENIZER.__class__.from_pretrained(model_path)
    from transformers import LayoutLMForTokenClassification
    model = LayoutLMForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def predict(tokens, bboxes, model, tokenizer, img_width=PAGE_WIDTH, img_height=PAGE_HEIGHT):
    """
    ✅ التعديل الرئيسي: بناخد img_width و img_height كـ parameters
    عشان نعمل normalize صح لأي صورة بأي أبعاد.
    """
    norm     = [normalize_bbox(b, img_width, img_height) for b in bboxes]
    encoding = tokenizer(
        tokens, is_split_into_words=True,
        return_tensors="pt", truncation=True, max_length=512,
        padding="max_length",
    )
    word_ids    = encoding.word_ids()
    bbox_tensor = [norm[wid] if wid is not None else [0, 0, 0, 0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    preds   = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    results, seen = [], set()

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


def extract_entities(tokens, bboxes, model, tokenizer,
                     img_width=PAGE_WIDTH, img_height=PAGE_HEIGHT):
    """
    ✅ التعديل: بنمرر أبعاد الصورة الحقيقية للـ predict
    """
    label_map = {
        "COMPANY": "company",
        "DATE":    "date",
        "TOTAL":   "total",
        "ADDRESS": "address",
    }
    buckets = {"company": [], "date": [], "total": [], "address": []}

    for item in predict(tokens, bboxes, model, tokenizer, img_width, img_height):
        if item["label"] == "O":
            continue
        parts = item["label"].split("-")
        if len(parts) < 2:
            continue
        etype = parts[1]
        key   = label_map.get(etype)
        if key:
            buckets[key].append(item["token"])

    return {k: " ".join(v) if v else None for k, v in buckets.items()}


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
    # ✅ مثال: صورة عندها أبعاد مختلفة عن الـ default
    img_width, img_height = 1200, 1600

    if os.path.exists(MODEL_PATH):
        model, tokenizer = load_model()
        result = extract_entities(tokens, bboxes, model, tokenizer,
                                  img_width, img_height)
        print("\n--- Inference Result ---")
        print(json.dumps(result, indent=2))
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}. Run training first.")