"""
ai/finetuning/finetune.py  ── FIXED VERSION
─────────────────────────────────────────────────────────────────────────────
Key fixes:
1. Class weights applied via custom Trainer
2. eval_strategy fixed (was causing issues in newer transformers)
3. Better metric tracking
"""

import gc
import json
import os
import sys
import shutil

import numpy as np
import torch
import torch.nn as nn
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
    LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS,
    BASE_MODEL_NAME, PAGE_WIDTH, PAGE_HEIGHT, MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS, PER_DEVICE_TRAIN_BATCH, PER_DEVICE_EVAL_BATCH,
    LEARNING_RATE, WARMUP_STEPS, WEIGHT_DECAY, LR_SCHEDULER,
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
BEST_DIR   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "model", "best_model")
)

for d in [OUTPUT_DIR, BEST_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)

tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_MODEL_NAME)


# ── Class weights ─────────────────────────────────────────────────────────────
# FIX: balanced weights — ADDRESS and TOTAL are rare so upweight them
# O, B-COMPANY, I-COMPANY, B-DATE, I-DATE, B-TOTAL, I-TOTAL, B-ADDRESS, I-ADDRESS
CLASS_WEIGHTS = torch.tensor(
    [0.1, 1.5, 1.5, 1.2, 1.2, 2.5, 2.5, 2.0, 2.0],
    dtype=torch.float
)


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to CrossEntropyLoss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        device  = logits.device
        weights = CLASS_WEIGHTS.to(device)

        loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss    = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ── Preprocessing ─────────────────────────────────────────────────────────────
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
                label_ids.append(LABEL2ID[labels[word_id]])
                bbox_ids.append(norm[word_id])
            else:
                # FIX: use -100 for sub-tokens (don't propagate label)
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

    # FIX: filter out records with no entity labels
    valid_records = []
    for r in records:
        if any(l != "O" for l in r["labels"]):
            valid_records.append(r)
        else:
            print(f"[SKIP] {r['id']} — all O labels")

    print(f"  Loaded {len(records)} records, {len(valid_records)} have entities")

    ds = Dataset.from_dict({
        "tokens": [r["tokens"] for r in valid_records],
        "bboxes": [r["bboxes"] for r in valid_records],
        "labels": [r["labels"] for r in valid_records],
    })

    return ds.map(
        tokenize_and_align,
        batched=True,
        remove_columns=["tokens", "bboxes", "labels"],
    )


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds          = np.argmax(logits, axis=-1)

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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Train JSON : {os.path.abspath(TRAIN_JSON)}")
    print(f"Test  JSON : {os.path.abspath(TEST_JSON)}")
    print(f"Output Dir : {OUTPUT_DIR}")

    if not os.path.exists(TRAIN_JSON):
        print(f"\n❌ Train data not found at {TRAIN_JSON}")
        sys.exit(1)

    print("\n🔄 Loading and tokenizing data...")
    train_ds = build_dataset(TRAIN_JSON)
    test_ds  = build_dataset(TEST_JSON)
    print(f"Train: {len(train_ds)} samples  |  Test: {len(test_ds)} samples")

    model = LayoutLMForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=BEST_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
        save_total_limit=2,           # keep only 2 checkpoints
        dataloader_num_workers=0,     # safe on Windows
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=8
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    print("\n🚀 Starting fine-tuning...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ Model saved to: {OUTPUT_DIR}")
