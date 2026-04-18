import json
import re
import os
from datasets import load_dataset
from preprocess import preprocess_split

# --- 1. Helpers for SROIE ---
def normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())

def find_span(tokens, entity_tokens):
    n, m = len(tokens), len(entity_tokens)
    norm_tokens = [normalize(t) for t in tokens]
    norm_entity = [normalize(t) for t in entity_tokens]
    for i in range(n - m + 1):
        if norm_tokens[i : i + m] == norm_entity:
            return i
    return None

LABEL_MAP = {"company": "COMPANY", "date": "DATE", "total": "TOTAL", "address": "ADDRESS"}

def bio_tag(tokens, entities):
    labels = ["O"] * len(tokens)
    for key, label_str in LABEL_MAP.items():
        val = str(entities.get(key, "")).strip()
        if not val: continue
        e_tokens = val.split()
        start = find_span(tokens, e_tokens)
        if start is not None:
            labels[start] = f"B-{label_str}"
            for i in range(start + 1, start + len(e_tokens)):
                labels[i] = f"I-{label_str}"
    return labels

# --- 2. CORD Processing (Fixing the Error) ---
def convert_cord(example):
    """تحويل بيانات CORD لتناسب تنسيق SROIE مع إضافة bboxes"""
    try:
        data = json.loads(example["ground_truth"])
        valid_lines = data.get("valid_line", [])
        tokens, bboxes, labels = [], [], []
        
        for line in valid_lines:
            cat = line.get("category", "O").upper()
            # mapping labels
            if "COMPANY" in cat: lab = "COMPANY"
            elif "DATE" in cat: lab = "DATE"
            elif "ADDRESS" in cat: lab = "ADDRESS"
            elif "TOTAL" in cat: lab = "TOTAL"
            else: lab = None
            
            for word in line.get("words", []):
                text = word["text"].strip()
                if text:
                    tokens.append(text)
                    # تحويل quad لـ [x_min, y_min, x_max, y_max]
                    q = word["quad"]
                    xs = [q["x1"], q["x2"], q["x3"], q["x4"]]
                    ys = [q["y1"], q["y2"], q["y3"], q["y4"]]
                    bboxes.append([min(xs), min(ys), max(xs), max(ys)])
                    labels.append(f"B-{lab}" if lab else "O")
        return {"tokens": tokens, "bboxes": bboxes, "labels": labels} if tokens else None
    except:
        return None

# --- 3. Main Logic ---
if __name__ == "__main__":
    # A. تجهيز بيانات SROIE من جهازك
    print("Processing SROIE from Disk...")
    # المسارات بناءً على الـ Terminal بتاعك
    sroie_path = r"D:\invoice-document-intelligence\Dataset\train"
    sroie_raw = preprocess_split(sroie_path)
    
    final_data = []
    for r in sroie_raw:
        final_data.append({
            "id": r["id"],
            "tokens": r["tokens"],
            "bboxes": r["bboxes"], # دي اللي كانت ناقصة!
            "labels": bio_tag(r["tokens"], r["entities"])
        })
    print(f"✅ SROIE Done: {len(final_data)} records")

# B. تحميل بيانات CORD
    print("Loading CORD-v2 from Hugging Face...")
    try:
        # أضفنا .remove_columns(["image"]) عشان نوفر وقت ورامات ونحل مشكلة Pillow
        ds = load_dataset("naver-clova-ix/cord-v2", split="train") 
        ds = ds.remove_columns(["image"]) 
        
        cord_count = 0
        for x in ds:
            res = convert_cord(x)
            if res:
                res["id"] = f"cord_{cord_count}"
                final_data.append(res)
                cord_count += 1
        print(f"✅ CORD Done: {cord_count} records added")

    except Exception as e:
        print(f"❌ Could not load CORD: {e}")

    # C. حفظ الملف النهائي المدموج
    output_file = "train_merged.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 SUCCESS! Final dataset saved to: {output_file}")
    print(f"Total records in merged file: {len(final_data)}")

    import json
import random

# 1. تحميل الملف المدموج الكبير
with open("train_merged.json", "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 2. ترتيب البيانات بشكل عشوائي عشان نضمن تنوع (SROIE + CORD)
random.seed(42)
random.shuffle(all_data)

# 3. تقسيم البيانات (80% تدريب و 20% اختبار)
split_point = int(len(all_data) * 0.8)
train_split = all_data[:split_point]
test_split = all_data[split_point:]

# 4. حفظ الملفات الجديدة في مجلد ai/data
import os
os.makedirs("ai/data", exist_ok=True)

with open("ai/data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_split, f, indent=2, ensure_ascii=False)

with open("ai/data/test.json", "w", encoding="utf-8") as f:
    json.dump(test_split, f, indent=2, ensure_ascii=False)

print(f"✅ تم إنشاء الملفات بنجاح في مجلد ai/data:")
print(f"   - Train records: {len(train_split)}")
print(f"   - Test records: {len(test_split)}")