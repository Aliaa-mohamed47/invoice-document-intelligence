import json, torch, os
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast

from model import LABEL_LIST, LABEL2ID, ID2LABEL, TOKENIZER

PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000
MODEL_PATH  = "/content/invoice-document-intelligence/ai/model/saved_model"

def normalize_bbox(bbox, width=PAGE_WIDTH, height=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox
    return [int(1000*x0/width), int(1000*y0/height),
            int(1000*x1/width), int(1000*y1/height)]

def load_model(model_path=MODEL_PATH):
    tokenizer = LayoutLMTokenizerFast.from_pretrained(model_path)
    model     = LayoutLMForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def predict(tokens, bboxes, model, tokenizer):
    norm     = [normalize_bbox(b) for b in bboxes]
    encoding = tokenizer(tokens, is_split_into_words=True,
                         return_tensors="pt", truncation=True, max_length=512)
    word_ids    = encoding.word_ids()
    bbox_tensor = [norm[wid] if wid is not None else [0,0,0,0] for wid in word_ids]
    encoding["bbox"] = torch.tensor([bbox_tensor], dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    preds   = torch.argmax(outputs.logits, dim=-1)[0].tolist()
    results, seen = [], set()
    for idx, wid in enumerate(word_ids):
        if wid is None or wid in seen: continue
        seen.add(wid)
        results.append({"token": tokens[wid], "label": ID2LABEL[preds[idx]], "bbox": bboxes[wid]})
    return results

def extract_entities(tokens, bboxes, model, tokenizer):
    label_map = {"COMPANY":"company","DATE":"date","TOTAL":"total","ADDRESS":"address"}
    buckets   = {"company":[],"date":[],"total":[],"address":[]}
    for item in predict(tokens, bboxes, model, tokenizer):
        if item["label"] == "O": continue
        parts = item["label"].split("-")
        if len(parts) < 2: continue
        etype = parts[1]
        key = label_map.get(etype)
        if key: buckets[key].append(item["token"])
    return {k: " ".join(v) for k, v in buckets.items()}

if __name__ == "__main__":
    tokens = ["KEDAI", "GUNTING", "Date:", "01/01/2023", "Total:", "25.00"]
    bboxes = [[100,100,200,120], [210,100,300,120], [100,200,150,220], 
              [160,200,250,220], [100,300,150,320], [160,300,220,320]]
    
    if os.path.exists(MODEL_PATH):
        model, tokenizer = load_model()
        result = extract_entities(tokens, bboxes, model, tokenizer)
        print("\n--- Inference Result ---")
        print(json.dumps(result, indent=2))
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}. Run training first.")
