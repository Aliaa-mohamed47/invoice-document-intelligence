# ai/model/model.py
# ─────────────────────────────────────────────────────────────────────────────
# Dataset class + LayoutLM model builder
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMTokenizerFast, LayoutLMForTokenClassification

# ✅ import from centralized config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from finetuning.config import (
    LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS,
    BASE_MODEL_NAME, PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
)


# ── Tokenizer — lazy loading ──────────────────────────────────────────────────
# Loads only when needed (avoids heavy import overhead)
_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_MODEL_NAME)
    return _tokenizer


# ── Bounding box normalization ────────────────────────────────────────────────
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class InvoiceLayoutLMDataset(Dataset):
    def __init__(self, json_path, max_length=MAX_SEQ_LENGTH):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer  = get_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        tokens = sample["tokens"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        norm_bboxes = [normalize_bbox(b) for b in bboxes]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        aligned_labels = []
        aligned_bboxes = []

        prev_word_id = None

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
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "bbox": torch.tensor(aligned_bboxes, dtype=torch.long),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model():
    return LayoutLMForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.environ.get(
        "BASE_DIR",
        os.path.join(os.path.dirname(__file__), "../..")
    )

    train_path = os.path.join(base, "ai/data/train.json")
    test_path  = os.path.join(base, "ai/data/test.json")

    if not os.path.exists(train_path):
        print(f"⚠️ Train data not found at {train_path}")
        print("   Run ai/data/clean_data.py first to generate the dataset")
    else:
        train_ds = InvoiceLayoutLMDataset(train_path)
        test_ds  = InvoiceLayoutLMDataset(test_path)

        print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")

        sample = train_ds[0]
        real_labels = sample["labels"][sample["labels"] != -100]

        print(
            "Sample labels:",
            [ID2LABEL[i.item()] for i in real_labels[:8]]
        )

    model = build_model()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Sanity check completed successfully.")