from collect_data import load_split

def parse_box_line(line):
    parts = line.split(",", maxsplit=8)
    if len(parts) < 9: return None
    text = parts[8].strip()
    try:
        coords = [int(p) for p in parts[:8]]
        xs, ys = coords[0::2], coords[1::2]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        return text, bbox
    except: return None

def extract_tokens_and_bboxes(box_lines):
    tokens, bboxes = [], []
    for line in box_lines:
        res = parse_box_line(line)
        if res:
            text, bbox = res
            words = text.split()
            tokens.extend(words)
            bboxes.extend([bbox] * len(words))
    return tokens, bboxes

def preprocess_split(split_dir):
    raw_records = load_split(split_dir)
    processed = []
    for r in raw_records:
        tokens, bboxes = extract_tokens_and_bboxes(r["box_lines"])
        if tokens:
            processed.append({
                "id": r["id"],
                "tokens": tokens,
                "bboxes": bboxes,
                "entities": r["entities"]
            })
    return processed