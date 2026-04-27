"""
ai/finetuning/finetune.py ── FINAL STABLE VERSION
─────────────────────────────────────────────────────────────────────────────
Fixes:
✔ Removed class weights (major instability source)
✔ Cleaner TrainingArguments
✔ Better stability for LayoutLM fine-tuning
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

from config import (
    LABEL2ID, ID2LABEL, NUM_LABELS,
    BASE_MODEL_NAME, PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH,
    PER_DEVICE_EVAL_BATCH,
    LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

gc.collect()
torch.cuda.empty_cache()

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
TRAIN_JSON = os.path.join(BASE_DIR, "data", "train.json")
TEST_JSON  = os.path.join(BASE_DIR, "data", "test.json")

OUTPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "saved_model")
)
BEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "best_model")
)

for d in [OUTPUT_DIR, BEST_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)

tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_MODEL_NAME)


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
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
        max_length=MAX_SEQ_LENGTH,
    )

    all_labels = []
    all_bboxes = []

    for i in range(len(examples["tokens"])):
        labels = examples["labels"][i]
        bboxes = examples["bboxes"][i]
        norm = [normalize_bbox(b) for b in bboxes]

        word_ids = tokenized.word_ids(batch_index=i)

        label_ids = []
        bbox_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
                bbox_ids.append([0, 0, 0, 0])

            elif word_id != prev_word_id:
                label_ids.append(LABEL2ID[labels[word_id]])
                bbox_ids.append(norm[word_id])

            else:
                # sub-token handling
                label_ids.append(-100)
                bbox_ids.append(norm[word_id])

            prev_word_id = word_id

        all_labels.append(label_ids)
        all_bboxes.append(bbox_ids)

    tokenized["labels"] = all_labels
    tokenized["bbox"] = all_bboxes
    return tokenized


# ─────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────
def build_dataset(json_path: str) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    valid = []
    for r in records:
        if any(l != "O" for l in r["labels"]):
            valid.append(r)

    print(f"  Loaded {len(records)} records, {len(valid)} have entities")

    ds = Dataset.from_dict({
        "tokens": [r["tokens"] for r in valid],
        "bboxes": [r["bboxes"] for r in valid],
        "labels": [r["labels"] for r in valid],
    })

    return ds.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "bboxes", "labels"],
    )


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []

    for pr, lr in zip(preds, labels):
        tl, tp = [], []

        for p, l in zip(pr, lr):
            if l == -100:
                continue

            tl.append(ID2LABEL[l])
            tp.append(ID2LABEL[p])

        true_labels.append(tl)
        true_preds.append(tp)

    f1 = f1_score(true_labels, true_preds)

    print("\n" + classification_report(true_labels, true_preds))

    return {"f1": f1}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Train JSON : {TRAIN_JSON}")
    print(f"Test  JSON : {TEST_JSON}")
    print(f"Output Dir : {OUTPUT_DIR}")

    train_ds = build_dataset(TRAIN_JSON)
    test_ds  = build_dataset(TEST_JSON)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    model = LayoutLMForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=BEST_DIR,

        num_train_epochs=5,  # 🔥 مهم
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        learning_rate=2e-5,  # 🔥 optimal for LayoutLM
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),

        gradient_accumulation_steps=2,  # 🔥 stability boost

        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    print("\n🚀 Starting fine-tuning...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Saved to: {OUTPUT_DIR}")