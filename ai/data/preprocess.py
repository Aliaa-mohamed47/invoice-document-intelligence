# ai/data/preprocess.py
import os
from collect_data import load_split

def parse_box_line(line: str) -> tuple[str | None, list[int] | None]:
    parts = line.split(",", maxsplit=8)
    if len(parts) < 9:
        return None, None

    try:
        coords = [int(p) for p in parts[:8]]
    except ValueError:
        return None, None

    text = parts[8].strip()
    if not text:
        return None, None

    xs = [coords[0], coords[2], coords[4], coords[6]]
    ys = [coords[1], coords[3], coords[5], coords[7]]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    return text, bbox

def extract_tokens_and_bboxes(box_lines: list[str]) -> tuple[list[str], list[list[int]]]:
    tokens: list[str] = []
    bboxes: list[list[int]] = []

    for line in box_lines:
        text, bbox = parse_box_line(line)
        if text is None:
            continue
        words = text.split()
        tokens.extend(words)
        bboxes.extend([bbox] * len(words))

    return tokens, bboxes

def preprocess_split(split_dir: str) -> list[dict]:
    raw_records = load_split(split_dir)
    processed = []

    for r in raw_records:
        tokens, bboxes = extract_tokens_and_bboxes(r["box_lines"])
        if not tokens:
            continue
        processed.append({
            "id":       r["id"],
            "tokens":   tokens,
            "bboxes":   bboxes,
            "entities": r["entities"],
        })

    return processed