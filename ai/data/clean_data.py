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
from preprocess import preprocess_split


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lower-case and collapse whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize_entity(entity_value: str) -> list[str]:
    """Split an entity string into individual tokens (same logic as OCR tokens)."""
    return entity_value.split()


def find_span(tokens: list[str], entity_tokens: list[str]) -> int | None:
    """
    Return the start index of the first occurrence of `entity_tokens`
    inside `tokens` (case-insensitive).  Returns None if not found.
    """
    n, m = len(tokens), len(entity_tokens)
    norm_tokens  = [normalize(t) for t in tokens]
    norm_entity  = [normalize(t) for t in entity_tokens]

    for i in range(n - m + 1):
        if norm_tokens[i : i + m] == norm_entity:
            return i
    return None


# ── BIO tagger ────────────────────────────────────────────────────────────────

# Priority order when spans overlap (higher = wins)
ENTITY_PRIORITY = {"company": 4, "address": 3, "date": 2, "total": 1}
LABEL_MAP = {
    "company": "COMPANY",
    "date":    "DATE",
    "total":   "TOTAL",
    "address": "ADDRESS",
}


def bio_tag(tokens: list[str], entities: dict) -> list[str]:
    """
    Produce a BIO label for every token.

    Strategy:
        1. Find the token span for each entity value.
        2. In case of overlap, higher-priority entity wins.
        3. Assign B- / I- / O labels.
    """
    n = len(tokens)
    # label_priority[i] = (priority, label_prefix) for position i
    label_priority: list[tuple[int, str] | None] = [None] * n

    for entity_key, label_str in LABEL_MAP.items():
        raw_value = entities.get(entity_key, "").strip()
        if not raw_value:
            continue

        entity_tokens = tokenize_entity(raw_value)
        start = find_span(tokens, entity_tokens)
        if start is None:
            # Try harder: sometimes only a substring matches
            # Fall back: skip silently (keeps O labels for those tokens)
            continue

        prio = ENTITY_PRIORITY.get(entity_key, 0)
        for offset, pos in enumerate(range(start, start + len(entity_tokens))):
            if pos >= n:
                break
            existing = label_priority[pos]
            if existing is None or prio > existing[0]:
                bio_prefix = "B" if offset == 0 else "I"
                label_priority[pos] = (prio, f"{bio_prefix}-{label_str}")

    return [lp[1] if lp else "O" for lp in label_priority]


# ── Public API ────────────────────────────────────────────────────────────────

def clean_split(split_dir: str) -> list[dict]:
    """Preprocess + BIO-tag an entire dataset split."""
    preprocessed = preprocess_split(split_dir)
    cleaned = []
    for r in preprocessed:
        labels = bio_tag(r["tokens"], r["entities"])
        assert len(labels) == len(r["tokens"]), "label/token length mismatch"
        cleaned.append({
            "id":     r["id"],
            "tokens": r["tokens"],
            "labels": labels,
        })
    return cleaned


if __name__ == "__main__":
    train = clean_split(r"D:\invoice-document-intelligence\Dataset\train")
    test  = clean_split(r"D:\invoice-document-intelligence\Dataset\test")

    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    print("✅ Data saved as train.json and test.json")
    
    if train:
        r = train[0]
        print(f"\nid      : {r['id']}")
        for tok, lbl in zip(r["tokens"][:20], r["labels"][:20]):
            print(f"  {tok:<20} {lbl}")

    # Label distribution
    from collections import Counter
    all_labels = [l for r in train for l in r["labels"]]
    print("\nLabel distribution (train):")
    for lbl, cnt in Counter(all_labels).most_common():
        print(f"  {lbl:<15} {cnt}")
