import json, torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]
LABEL2ID    = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL    = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS  = len(LABEL_LIST)
PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000
TOKENIZER   = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

def normalize_bbox(bbox, width=PAGE_WIDTH, height=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox

    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))

    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]

class InvoiceLayoutLMDataset(Dataset):
    def __init__(self, json_path, tokenizer=TOKENIZER, max_length=512):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample      = self.data[idx]
        tokens      = sample["tokens"]
        bboxes      = sample["bboxes"]
        labels      = sample["labels"]
        norm_bboxes = [normalize_bbox(b) for b in bboxes]
        encoding    = self.tokenizer(
            tokens, is_split_into_words=True, truncation=True,
            padding="max_length", max_length=self.max_length, return_tensors="pt",
        )
        word_ids       = encoding.word_ids(batch_index=0)
        aligned_labels = []
        aligned_bboxes = []
        prev_word_id   = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
                aligned_bboxes.append([0, 0, 0, 0])
            elif word_id != prev_word_id:
                aligned_labels.append(LABEL2ID[labels[word_id]])
                aligned_bboxes.append(norm_bboxes[word_id])
            else:
                aligned_labels.append(-100)
                aligned_bboxes.append(norm_bboxes[word_id])
            prev_word_id = word_id
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "bbox":           torch.tensor(aligned_bboxes, dtype=torch.long),
            "labels":         torch.tensor(aligned_labels, dtype=torch.long),
        }

def build_model():
    return LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
    )

if __name__ == "__main__":
    import os
    base     = "/content/invoice-document-intelligence"
    train_ds = InvoiceLayoutLMDataset(os.path.join(base, "ai/data/train.json"))
    test_ds  = InvoiceLayoutLMDataset(os.path.join(base, "ai/data/test.json"))
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")
    s    = train_ds[0]
    real = s["labels"][s["labels"] != -100]
    print(f"Real labels sample: {[ID2LABEL[i.item()] for i in real[:8]]}")
    print(f"Params: {sum(p.numel() for p in build_model().parameters()):,}")
    print("Sanity check passed.")
