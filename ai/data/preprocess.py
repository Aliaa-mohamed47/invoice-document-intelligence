"""
preprocess.py
-------------
Extracts plain OCR tokens from the SROIE box-file lines.

Each raw line format:
    x1,y1,x2,y2,x3,y3,x4,y4,text

We discard the eight coordinate values and keep only `text`.
Output per record:
    { "id": str, "tokens": [str], "entities": dict }
"""

from collect_data import load_split


def parse_box_line(line: str) -> str | None:
    """
    Extract the text portion from one box-file line.
    Returns None if the line is malformed.
    """
    parts = line.split(",", maxsplit=8)   # split on first 8 commas only
    if len(parts) < 9:
        return None
    text = parts[8].strip()
    return text if text else None


def extract_tokens(box_lines: list[str]) -> list[str]:
    """
    Convert a list of raw box lines into a flat list of word tokens.
    Multiple words on the same line are split by whitespace.
    """
    tokens = []
    for line in box_lines:
        text = parse_box_line(line)
        if text:
            tokens.extend(text.split())
    return tokens


def preprocess_split(split_dir: str) -> list[dict]:
    """Load + preprocess one dataset split."""
    raw_records = load_split(split_dir)
    processed = []
    for r in raw_records:
        tokens = extract_tokens(r["box_lines"])
        if not tokens:
            print(f"[WARN] empty token list for {r['id']}, skipping")
            continue
        processed.append({
            "id":       r["id"],
            "tokens":   tokens,
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
        print(f"entities: {r['entities']}")