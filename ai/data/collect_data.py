import os, json
from pathlib import Path

def load_split(split_dir):
    split_dir = Path(split_dir)
    box_dir = split_dir / "box"
    entity_dir = split_dir / "entities"
    records = []
    
    if not box_dir.exists():
        print(f"[ERROR] Directory not found: {box_dir}")
        return []

    for box_file in sorted(box_dir.glob("*.txt")):
        sample_id = box_file.stem
        with open(box_file, "r", encoding="utf-8", errors="replace") as f:
            box_lines = [line.rstrip("\n") for line in f if line.strip()]
        
        entity_file = entity_dir / f"{sample_id}.txt"
        if not entity_file.exists():
            entity_file = entity_dir / f"{sample_id}.json"
            
        if not entity_file.exists():
            continue
            
        with open(entity_file, "r", encoding="utf-8", errors="replace") as f:
            try:
                # SROIE entities are sometimes flat text, sometimes JSON
                content = f.read().strip()
                try:
                    entities = json.loads(content)
                except:
                    # If it's the old SROIE format (text), we handle it or skip
                    continue 
            except Exception as e:
                continue
        
        records.append({"id": sample_id, "box_lines": box_lines, "entities": entities})
    
    print(f"Loaded {len(records)} records from {split_dir}")
    return records