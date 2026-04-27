"""
finetuning/finetune.py
----------------------
Fine-tuning كامل لـ LayoutLM على بيانات SROIE

شغّل ده على Google Colab بـ GPU (T4 أو A100)
وقت التدريب: ~30-60 دقيقة على T4

بعد التدريب: انسخ مجلد saved_model/ للـ repo
"""

import gc
import json
import os
import sys
import shutil

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, classification_report
from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMTokenizerFast,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

gc.collect()
torch.cuda.empty_cache()

# ── Paths ──────────────────────────────────────────────────────────────────────
# ✅ يشتغل على Colab و VS Code بدون تعديل
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "train.json")
TEST_JSON  = os.path.join(BASE_DIR, "data", "test.json")
OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "saved_model")
)
BEST_DIR   = os.path.join(BASE_DIR, "model", "best_model")

# تنظيف المجلدات القديمة
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
if os.path.exists(BEST_DIR):
    shutil.rmtree(BEST_DIR)

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_LIST  = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE",
               "B-TOTAL", "I-TOTAL", "B-ADDRESS", "I-ADDRESS"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

tokenizer   = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000


# ── Preprocessing ──────────────────────────────────────────────────────────────
def normalize_bbox(bbox, w=PAGE_WIDTH, h=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(int(1000 * x0 / w), 1000)),
        max(0, min(int(1000 * y0 / h), 1000)),
        max(0, min(int(1000 * x1 / w), 1000)),
        max(0, min(int(1000 * y1 / h), 1000)),
    ]


def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    all_labels = []
    all_bboxes = []

    for i in range(len(examples["tokens"])):
        labels   = examples["labels"][i]
        bboxes   = examples["bboxes"][i]
        norm     = [normalize_bbox(b) for b in bboxes]
        word_ids = tokenized.word_ids(batch_index=i)

        label_ids    = []
        bbox_ids     = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                bbox_ids.append([0, 0, 0, 0])
            elif word_id != prev_word_id:
                label_ids.append(LABEL_TO_ID[labels[word_id]])
                bbox_ids.append(norm[word_id])
            else:
                label_ids.append(-100)
                bbox_ids.append(norm[word_id])
            prev_word_id = word_id

        all_labels.append(label_ids)
        all_bboxes.append(bbox_ids)

    tokenized["labels"] = all_labels
    tokenized["bbox"]   = all_bboxes
    return tokenized


def build_dataset(json_path: str) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    ds = Dataset.from_dict({
        "tokens": [r["tokens"] for r in records],
        "bboxes": [r["bboxes"] for r in records],
        "labels": [r["labels"] for r in records],
    })
    return ds.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "bboxes", "labels"],
    )


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds          = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []

    for pr, lr in zip(preds, labels):
        tl, tp = [], []
        for p, l in zip(pr, lr):
            if l == -100:
                continue
            tl.append(ID_TO_LABEL[l])
            tp.append(ID_TO_LABEL[p])
        true_labels.append(tl)
        true_preds.append(tp)

    f1 = f1_score(true_labels, true_preds)
    print("\n" + classification_report(true_labels, true_preds))
    return {"f1": f1}


# ── Main ───────────────────────────────────────────────────────────────────────
print(f"Train JSON: {os.path.abspath(TRAIN_JSON)}")
print(f"Test JSON:  {os.path.abspath(TEST_JSON)}")

if not os.path.exists(TRAIN_JSON):
    print(f"❌ Train data not found at {TRAIN_JSON}")
    print("   شغّل ai/data/clean_data.py أولاً لتوليد train.json")
    sys.exit(1)

print("🔄 Loading and tokenizing data...")
train_ds = build_dataset(TRAIN_JSON)
test_ds  = build_dataset(TEST_JSON)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels  = len(LABEL_LIST),
    id2label    = ID_TO_LABEL,
    label2id    = LABEL_TO_ID,
)

args = TrainingArguments(
    output_dir                  = BEST_DIR,
    num_train_epochs            = 15,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    learning_rate               = 5e-5,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    greater_is_better           = True,
    fp16                        = torch.cuda.is_available(),
    logging_steps               = 20,
    report_to                   = "none",
)

trainer = Trainer(
    model            = model,
    args             = args,
    train_dataset    = train_ds,
    eval_dataset     = test_ds,
    processing_class = tokenizer,
    data_collator    = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=5)],
)

print("🚀 Starting fine-tuning...")
trainer.train()

os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to: {OUTPUT_DIR}")
