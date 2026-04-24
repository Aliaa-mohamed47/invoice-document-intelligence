"""
preprocess.py
-------------
Extracts OCR tokens AND bounding boxes from SROIE box-file lines.

Each raw line format:
    x1,y1,x2,y2,x3,y3,x4,y4,text

We keep the text token AND derive a rectangular bbox [x_min, y_min, x_max, y_max]
from the four corner coordinates.

Output per record:
    { "id": str, "tokens": [str], "bboxes": [[x0,y0,x1,y1], ...], "entities": dict }
"""

from collect_data import load_split


def parse_box_line(line: str) -> tuple[str, list[int]] | tuple[None, None]:
    """
    Extract (text, bbox) from one box-file line.
    bbox is [x_min, y_min, x_max, y_max] derived from the 8 coordinate values.
    Returns (None, None) if the line is malformed.
    """
    parts = line.split(",", maxsplit=8)   # split on first 8 commas only
    if len(parts) < 9:
        return None, None

    try:
        coords = [int(parts[i]) for i in range(8)]
    except ValueError:
        return None, None

    text = parts[8].strip()
    if not text:
        return None, None

    # Derive axis-aligned bbox from four corner points
    xs = [coords[0], coords[2], coords[4], coords[6]]
    ys = [coords[1], coords[3], coords[5], coords[7]]
    bbox = [min(xs), min(ys), max(xs), max(ys)]   # [x_min, y_min, x_max, y_max]

    return text, bbox


def extract_tokens_and_bboxes(box_lines: list[str]) -> tuple[list[str], list[list[int]]]:
    """
    Convert a list of raw box lines into:
        - a flat list of word tokens
        - a parallel list of bboxes (one per token)

    Multi-word text segments on the same line share the same bbox.
    """
    tokens: list[str] = []
    bboxes: list[list[int]] = []

    for line in box_lines:
        text, bbox = parse_box_line(line)
        if text is None:
            continue
        words = text.split()
        tokens.extend(words)
        bboxes.extend([bbox] * len(words))   # same bbox for every word on the line

    return tokens, bboxes


def preprocess_split(split_dir: str) -> list[dict]:
    """Load + preprocess one dataset split."""
    raw_records = load_split(split_dir)
    processed = []
    for r in raw_records:
        tokens, bboxes = extract_tokens_and_bboxes(r["box_lines"])
        if not tokens:
            print(f"[WARN] empty token list for {r['id']}, skipping")
            continue
        processed.append({
            "id":       r["id"],
            "tokens":   tokens,
            "bboxes":   bboxes,    # ← NEW: required by model.py / pipeline.py
            "entities": r["entities"],
        })
    return processed


if __name__ == "__main__":
    train = preprocess_split("SROIE2019/train")
    test  = preprocess_split("SROIE2019/test")

    if train:
        r = train[0]
        print(f"\nid      : {r['id']}")
        print(f"tokens  : {r['tokens'][:10]}")
        print(f"bboxes  : {r['bboxes'][:3]}")
        print(f"entities: {r['entities']}")