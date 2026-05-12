
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL,
    PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)

from results_to_json import format_output

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
TEST_JSON     = os.path.join(BASE_DIR, "data", "test.json")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "evaluation_results")
FINETUNED_OUT = os.path.join(RESULTS_DIR, "finetuned_results.json")
BASELINE_OUT  = os.path.join(RESULTS_DIR, "baseline_results.json")
FIELDS        = ["company", "date", "total", "address"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_test_data():
    if not os.path.exists(TEST_JSON):
        raise FileNotFoundError(f"[!] {TEST_JSON} not found.")
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[✓] Loaded {len(data)} test records")
    return data


def extract_entity_from_bio(tokens, labels, entity_type):
    words = []
    for token, label in zip(tokens, labels):
        if label == f"B-{entity_type}":
            words.append(token)
        elif label == f"I-{entity_type}" and words:
            words.append(token)
    return " ".join(words) if words else None


def get_ground_truth(record):
    return {
        "company": extract_entity_from_bio(record["tokens"], record["labels"], "COMPANY"),
        "date":    extract_entity_from_bio(record["tokens"], record["labels"], "DATE"),
        "total":   extract_entity_from_bio(record["tokens"], record["labels"], "TOTAL"),
        "address": extract_entity_from_bio(record["tokens"], record["labels"], "ADDRESS"),
    }


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"[→] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model     = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    print("[✓] Model loaded")
    return model, tokenizer


# ── BBox normalization ────────────────────────────────────────────────────────
def normalize_bbox(bbox, width=PAGE_WIDTH, height=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / width),  1000)),
        max(0, min(int(1000 * y0 / height), 1000)),
        max(0, min(int(1000 * x1 / width),  1000)),
        max(0, min(int(1000 * y1 / height), 1000)),
    ]


# ── Inference ─────────────────────────────────────────────────────────────────
def real_predict(tokens, bboxes, model, tokenizer):

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )

    if bboxes is not None:
        word_ids    = encoding.word_ids()
        norm_bboxes = [normalize_bbox(b) for b in bboxes]
        bbox_tensor = [
            norm_bboxes[wid] if wid is not None else [0, 0, 0, 0]
            for wid in word_ids
        ]
        encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)

    with torch.no_grad():
        outputs = model(**encoding)

    logits    = outputs.logits[0]
    probs     = torch.softmax(logits, dim=-1)
    pred_ids  = torch.argmax(logits, dim=-1).tolist()
    max_probs = probs.max(dim=-1).values.tolist()

    field_map    = {"COMPANY": "company", "DATE": "date",
                    "TOTAL": "total", "ADDRESS": "address"}
    entities     = {f: [] for f in FIELDS}
    entity_probs = {f: [] for f in FIELDS}

    seen     = set()
    word_ids = encoding.word_ids()
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        label = ID2LABEL[pred_ids[idx]]
        if label == "O":
            continue
        _, etype = label.split("-", 1)
        key = field_map.get(etype)
        if key:
            entities[key].append(tokens[wid])
            entity_probs[key].append(max_probs[idx])

    output = {f: " ".join(entities[f]) if entities[f] else None for f in FIELDS}
    for f in FIELDS:
        output[f"{f}_score"] = (
            round(float(np.mean(entity_probs[f])), 4)
            if entity_probs[f] else 0.0
        )
    return output


# ── Metrics ───────────────────────────────────────────────────────────────────
def normalize_text(text):
    if not isinstance(text, str) or not text:
        return ""
    return (text.lower().strip()
            .replace(",", "")
            .replace("$", "")
            .replace("rm", "")
            .strip())


def compute_metrics(ground_truths, predictions):
    TP = FP = FN = 0
    for gt, pred in zip(ground_truths, predictions):
        gt_n, pred_n = normalize_text(gt), normalize_text(pred)
        if gt_n and pred_n:
            if gt_n == pred_n:
                TP += 1
            else:
                FP += 1
                FN += 1
        elif pred_n and not gt_n:
            FP += 1
        elif gt_n and not pred_n:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "TP": TP, "FP": FP, "FN": FN,
    }


# ── Evaluation runner ─────────────────────────────────────────────────────────
def run_evaluation(records, output_path, model, tokenizer):
    all_predictions = []

    print(f"\n[→] Evaluating {len(records)} records ...")
    for record in tqdm(records):
        ground_truth = get_ground_truth(record)
        start        = time.time()
        bboxes       = record.get("bboxes")
        raw_pred     = real_predict(record["tokens"], bboxes, model, tokenizer)
        formatted    = format_output(record["id"], raw_pred, start)
        formatted["ground_truth"] = ground_truth
        all_predictions.append(formatted)

    results = {"per_field": {}, "macro_avg": {}, "predictions": all_predictions}

    print(f"\n{'Field':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 44)

    for field in FIELDS:
        gts   = [p["ground_truth"].get(field) for p in all_predictions]
        preds = [p["extracted_fields"].get(field) for p in all_predictions]
        m     = compute_metrics(gts, preds)
        results["per_field"][field] = m
        print(f"{field:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    avg_p  = sum(results["per_field"][f]["precision"] for f in FIELDS) / len(FIELDS)
    avg_r  = sum(results["per_field"][f]["recall"]    for f in FIELDS) / len(FIELDS)
    avg_f1 = sum(results["per_field"][f]["f1"]        for f in FIELDS) / len(FIELDS)
    results["macro_avg"] = {
        "precision": round(avg_p, 4),
        "recall":    round(avg_r, 4),
        "f1":        round(avg_f1, 4),
    }
    print("─" * 44)
    print(f"{'MACRO AVG':<12} {avg_p:>10.4f} {avg_r:>8.4f} {avg_f1:>8.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved → {output_path}")
    return results


# ── Comparison ────────────────────────────────────────────────────────────────
def compare():
    if not os.path.exists(BASELINE_OUT):
        print("[!] baseline_results.json not found — skipping comparison")
        return

    with open(BASELINE_OUT,  encoding="utf-8") as f:
        b = json.load(f)
    with open(FINETUNED_OUT, encoding="utf-8") as f:
        ft = json.load(f)

    print(f"\n{'Field':<12} {'Base F1':>9} {'FT F1':>9} {'Δ F1':>9} {'Improved?':>10}")
    print("─" * 54)
    for field in FIELDS:
        bf = b["per_field"][field]["f1"]
        ff = ft["per_field"][field]["f1"]
        print(f"{field:<12} {bf:>9.4f} {ff:>9.4f} {ff-bf:>+9.4f} "
              f"{'✓' if ff > bf else '✗':>10}")
    print("─" * 54)
    bm = b["macro_avg"]["f1"]
    fm = ft["macro_avg"]["f1"]
    print(f"{'MACRO AVG':<12} {bm:>9.4f} {fm:>9.4f} {fm-bm:>+9.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    records = load_test_data()

    MODEL_PATH = (
        "/content/invoice-document-intelligence/ai/model/saved_model"
        if os.path.exists("/content/invoice-document-intelligence/ai/model/saved_model")
        else os.path.join(BASE_DIR, "model", "saved_model")
    )

    model, tokenizer = load_model(MODEL_PATH)

    print("\n=== Fine-tuned Model Evaluation ===")
    run_evaluation(records, FINETUNED_OUT, model, tokenizer)

    print("\n=== Baseline vs Fine-tuned Comparison ===")
    compare()