# evaluate.py

import json
import time
import os
import numpy as np
from tqdm import tqdm
from results_to_json import format_output
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ── CONFIG ───────────────────────────────────────────────────────────────────
USE_MOCK      = True

TEST_JSON     = "ai/data/test.json"
RESULTS_DIR   = "evaluation_results"
FINETUNED_OUT = f"{RESULTS_DIR}/finetuned_results.json"
BASELINE_OUT  = f"{RESULTS_DIR}/baseline_results.json"
FIELDS        = ["company", "date", "total", "address"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── LABEL DEFINITIONS (must match finetune.py exactly) ───────────────────────
LABEL_LIST = [
    "O",
    "B-ADDRESS", "I-ADDRESS",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ── LOAD TEST DATA ───────────────────────────────────────────────────────────
def load_test_data():
    if not os.path.exists(TEST_JSON):
        raise FileNotFoundError(
            f"[!] {TEST_JSON} not found. "
            f"Make sure وعد's data pipeline has been run first."
        )
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[✓] Loaded {len(data)} test records from {TEST_JSON}")
    return data


# ── EXTRACT GROUND TRUTH FROM BIO LABELS ────────────────────────────────────
def extract_entity_from_bio(tokens, labels, entity_type):
    """Reconstruct entity text from BIO-tagged tokens."""
    words = []
    for token, label in zip(tokens, labels):
        if label == f"B-{entity_type}":
            words.append(token)
        elif label == f"I-{entity_type}" and words:
            words.append(token)
    return " ".join(words) if words else None


def get_ground_truth(record):
    """Extract all 4 fields from a record's BIO labels."""
    tokens = record["tokens"]
    labels = record["labels"]
    return {
        "company": extract_entity_from_bio(tokens, labels, "COMPANY"),
        "date":    extract_entity_from_bio(tokens, labels, "DATE"),
        "total":   extract_entity_from_bio(tokens, labels, "TOTAL"),
        "address": extract_entity_from_bio(tokens, labels, "ADDRESS"),
    }


# ── LOAD REAL MODEL ──────────────────────────────────────────────────────────
def load_model(model_path):
    print(f"[→] Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    print("[✓] Model loaded")
    return model, tokenizer


# ── REAL PREDICTION ──────────────────────────────────────────────────────────
def real_predict(tokens, model, tokenizer):
    import torch

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits    = outputs.logits[0]
    probs     = torch.softmax(logits, dim=-1)
    pred_ids  = torch.argmax(logits, dim=-1).tolist()
    max_probs = probs.max(dim=-1).values.tolist()

    field_map = {"COMPANY": "company", "DATE": "date",
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


# ── MOCK PREDICTION (used while USE_MOCK = True) ─────────────────────────────
def mock_predict(tokens, ground_truth):
    """Simulates ~85% accuracy. Replace with real model when ready."""
    import random
    output = {}
    for f in FIELDS:
        gt_val = ground_truth.get(f)
        output[f] = gt_val if (gt_val and random.random() > 0.15) else None
        output[f"{f}_score"] = round(random.uniform(0.65, 0.98), 4)
    return output


# ── METRICS ──────────────────────────────────────────────────────────────────
def normalize(text):
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
        gt_n, pred_n = normalize(gt), normalize(pred)
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
    f1 = ((2 * precision * recall) / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "TP": TP, "FP": FP, "FN": FN
    }


# ── RUN EVALUATION ────────────────────────────────────────────────────────────
def run_evaluation(records, output_path, model=None, tokenizer=None):
    all_predictions = []

    print(f"\n[→] Evaluating {len(records)} records ...")
    for record in tqdm(records):
        ground_truth = get_ground_truth(record)
        start = time.time()

        if USE_MOCK or model is None:
            raw_pred = mock_predict(record["tokens"], ground_truth)
        else:
            raw_pred = real_predict(record["tokens"], model, tokenizer)

        formatted = format_output(record["id"], raw_pred, start)
        formatted["ground_truth"] = ground_truth
        all_predictions.append(formatted)

    results = {"per_field": {}, "macro_avg": {}, "predictions": all_predictions}

    print(f"\n{'Field':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("─" * 44)

    for field in FIELDS:
        gts   = [p["ground_truth"].get(field) for p in all_predictions]
        preds = [p["extracted_fields"].get(field) for p in all_predictions]
        m = compute_metrics(gts, preds)
        results["per_field"][field] = m
        print(f"{field:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

    avg_p  = sum(results["per_field"][f]["precision"] for f in FIELDS) / len(FIELDS)
    avg_r  = sum(results["per_field"][f]["recall"]    for f in FIELDS) / len(FIELDS)
    avg_f1 = sum(results["per_field"][f]["f1"]        for f in FIELDS) / len(FIELDS)
    results["macro_avg"] = {
        "precision": round(avg_p, 4),
        "recall":    round(avg_r, 4),
        "f1":        round(avg_f1, 4)
    }
    print("─" * 44)
    print(f"{'MACRO AVG':<12} {avg_p:>10.4f} {avg_r:>8.4f} {avg_f1:>8.4f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[✓] Saved → {output_path}")
    return results


# ── COMPARE BASELINE VS FINE-TUNED ───────────────────────────────────────────
def compare():
    if not os.path.exists(BASELINE_OUT):
        print("[!] baseline_results.json not found — skipping comparison")
        return

    b = json.load(open(BASELINE_OUT,  encoding="utf-8"))
    f = json.load(open(FINETUNED_OUT, encoding="utf-8"))

    print(f"\n{'Field':<12} {'Base F1':>9} {'FT F1':>9} {'Δ F1':>9} {'Improved?':>10}")
    print("─" * 54)
    for field in FIELDS:
        bf = b["per_field"][field]["f1"]
        ff = f["per_field"][field]["f1"]
        print(f"{field:<12} {bf:>9.4f} {ff:>9.4f} {ff-bf:>+9.4f} "
              f"{'✓' if ff > bf else '✗':>10}")
    print("─" * 54)
    bm = b["macro_avg"]["f1"]
    fm = f["macro_avg"]["f1"]
    print(f"{'MACRO AVG':<12} {bm:>9.4f} {fm:>9.4f} {fm-bm:>+9.4f}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)

    records = load_test_data()

    if not USE_MOCK:
        MODEL_PATH = (
            "/content/invoice-document-intelligence/ai/model/saved_model"
            if os.path.exists("/content/invoice-document-intelligence/ai/model/saved_model")
            else "ai/model/saved_model"
        )
        model, tokenizer = load_model(MODEL_PATH)
    else:
        print("[!] USE_MOCK=True — using simulated predictions for testing")
        model, tokenizer = None, None

    print("\n=== Fine-tuned Model Evaluation ===")
    run_evaluation(records, FINETUNED_OUT, model, tokenizer)

    print("\n=== Baseline vs Fine-tuned Comparison ===")
    compare()
