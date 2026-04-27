"""
clean_data.py
-------------
Aligns OCR tokens with entity values and converts to BIO tag sequences.

Entity keys supported (lower-cased from JSON):
    company, date, total, address

Output per record:
    { "id": str, "tokens": [str], "labels": [str] }

BIO scheme:
    B-COMPANY / I-COMPANY
    B-DATE    / I-DATE
    B-TOTAL   / I-TOTAL
    B-ADDRESS / I-ADDRESS
    O  (outside)
"""
import json
import re
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_split


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize_entity(entity_value: str) -> list:
    return entity_value.split()


def find_span(tokens: list, entity_tokens: list):

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


def bio_tag(tokens: list, entities: dict) -> list:
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


def clean_split(split_dir: str) -> list:
    preprocessed = preprocess_split(split_dir)
    cleaned = []

    for r in preprocessed:
        labels = bio_tag(r["tokens"], r["entities"])
        assert len(labels) == len(r["tokens"]), \
            f"label/token mismatch in {r['id']}"

        non_o = [l for l in labels if l != "O"]
        if not non_o:
            print(f"[WARN] No entities found in {r['id']} — check entity alignment")

        cleaned.append({
            "id":     r["id"],
            "tokens": r["tokens"],
            "bboxes": r["bboxes"],
            "labels": labels,
        })

    return cleaned


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset")
    out  = os.path.dirname(__file__)

    train = clean_split(os.path.join(base, "train"))
    test  = clean_split(os.path.join(base, "test"))

    train_path = os.path.join(out, "train.json")
    test_path  = os.path.join(out, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(train)} train records → {train_path}")
    print(f"✅ Saved {len(test)} test records  → {test_path}")

    from collections import Counter
    all_labels = [l for r in train for l in r["labels"]]
    print("\nLabel distribution (train):")
    for lbl, cnt in Counter(all_labels).most_common():
        print(f"  {lbl:<15} {cnt}")

    total = len(all_labels)
    o_count = all_labels.count("O")
    o_ratio = o_count / total * 100
    print(f"\n  O ratio: {o_ratio:.1f}%")
    if o_ratio > 95:
        print("  ⚠️  WARNING: O ratio > 95% — entity alignment might be failing!")