# ai/data/clean_data.py
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_split

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def tokenize_entity(entity_value: str) -> list[str]:
    return entity_value.split()

def find_span(tokens: list[str], entity_tokens: list[str]) -> tuple[int | None, int]:
    n, m = len(tokens), len(entity_tokens)
    norm_tokens = [normalize(t) for t in tokens]
    norm_entity = [normalize(t) for t in entity_tokens]

    for i in range(n - m + 1):
        if norm_tokens[i: i + m] == norm_entity:
            return i, m

    best_start, best_len = None, 0
    for i in range(n):
        match_len = 0
        for j, et in enumerate(norm_entity):
            if i + j < n and norm_tokens[i + j] == et:
                match_len += 1
            else:
                break
        if match_len > len(norm_entity) // 2 and match_len > best_len:
            best_start, best_len = i, match_len

    if best_start is not None:
        return best_start, best_len

    return None, 0

ENTITY_PRIORITY = {"company": 4, "address": 3, "date": 2, "total": 1}
LABEL_MAP = {
    "company": "COMPANY",
    "date":    "DATE",
    "total":   "TOTAL",
    "address": "ADDRESS",
}

def bio_tag(tokens: list[str], entities: dict) -> list[str]:
    n = len(tokens)
    label_priority = [None] * n

    for entity_key, label_str in LABEL_MAP.items():
        raw_value = entities.get(entity_key, "").strip()
        if not raw_value:
            continue

        entity_tokens = tokenize_entity(raw_value)
        start, span_len = find_span(tokens, entity_tokens)
        if start is None:
            continue

        prio = ENTITY_PRIORITY.get(entity_key, 0)
        for offset in range(span_len):
            pos = start + offset
            if pos >= n:
                break
            existing = label_priority[pos]
            if existing is None or prio > existing[0]:
                bio_prefix = "B" if offset == 0 else "I"
                label_priority[pos] = (prio, f"{bio_prefix}-{label_str}")

    return [lp[1] if lp else "O" for lp in label_priority]

def clean_split(split_dir: str) -> list[dict]:
    preprocessed = preprocess_split(split_dir)
    cleaned = []

    for r in preprocessed:
        labels = bio_tag(r["tokens"], r["entities"])
        cleaned.append({
            "id":     r["id"],
            "tokens": r["tokens"],
            "bboxes": r["bboxes"],
            "labels": labels,
        })
    return cleaned

if __name__ == "__main__":
    dataset_base = os.environ.get("DATASET_DIR", os.path.join(os.path.dirname(__file__), "SROIE2019"))
    out_dir = os.environ.get("OUTPUT_DIR", os.path.dirname(__file__))

    train = clean_split(os.path.join(dataset_base, "train"))
    test  = clean_split(os.path.join(dataset_base, "test"))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(os.path.join(out_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)