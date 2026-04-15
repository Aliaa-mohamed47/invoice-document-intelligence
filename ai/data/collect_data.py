"""
collect_data.py
---------------
Loads the raw SROIE2019 dataset.
Returns a list of dicts:  { "id": str, "box_lines": [str], "entities": dict }
"""

import os
import json
from pathlib import Path


def load_split(split_dir: str) -> list[dict]:
    """
    Load one split (train or test).

    split_dir  – path to  SROIE2019/train/  or  SROIE2019/test/
    """
    split_dir = Path(split_dir)
    box_dir    = split_dir / "box"
    entity_dir = split_dir / "entities"

    records = []

    for box_file in sorted(box_dir.glob("*.txt")):
        sample_id = box_file.stem           # e.g. "X51005365187"

        # ── OCR lines ──────────────────────────────────────────────────────────
        with open(box_file, "r", encoding="utf-8", errors="replace") as f:
            box_lines = [line.rstrip("\n") for line in f if line.strip()]

        # ── Entities JSON ───────────────────────────────────────────────────────
        entity_file = entity_dir / f"{sample_id}.txt"
        if not entity_file.exists():
            entity_file = entity_dir / f"{sample_id}.json"

        if not entity_file.exists():
            print(f"[WARN] no entity file for {sample_id}, skipping")
            continue

        with open(entity_file, "r", encoding="utf-8", errors="replace") as f:
            try:
                entities = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON error in {entity_file}: {e}, skipping")
                continue

        records.append({
            "id":        sample_id,
            "box_lines": box_lines,
            "entities":  entities,
        })

    print(f"Loaded {len(records)} records from {split_dir}")
    return records


if __name__ == "__main__":
    train_records = load_split("SROIE2019/train")
    test_records  = load_split("SROIE2019/test")

    # Quick sanity check
    if train_records:
        r = train_records[0]
        print(f"\nSample id : {r['id']}")
        print(f"Box lines : {r['box_lines'][:3]}")
        print(f"Entities  : {r['entities']}")