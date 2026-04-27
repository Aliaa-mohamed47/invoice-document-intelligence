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

# ── Paths ─────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")

TRAIN_JSON = os.path.join(BASE_DIR, "data", "train.json")
TEST_JSON  = os.path.join(BASE_DIR, "data", "test.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "model", "saved_model")
BEST_DIR   = os.path.join(BASE_DIR, "model", "best_model")

# تنظيف قديم
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
if os.path.exists(BEST_DIR):
    shutil.rmtree(BEST_DIR)

# ── Labels ─────────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS"
]

LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

tokenizer = LayoutLMTokenizerFast.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)

PAGE_WIDTH = 762
PAGE_HEIGHT = 1000

# ── bbox normalization ─────────────────────────────
def normalize_bbox(bbox):
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / PAGE_WIDTH),
        int(1000 * y0 / PAGE_HEIGHT),
        int(1000 * x1 / PAGE_WIDTH),
        int(1000 * y1 / PAGE_HEIGHT),
    ]

# ── tokenization ───────────────────────────────────
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
        labels = examples["labels"][i]
        bboxes = examples["bboxes"][i]
        bboxes = [normalize_bbox(b) for b in bboxes]

        word_ids = tokenized.word_ids(batch_index=i)

        label_ids = []
        bbox_ids = []
        prev = None

        for w in word_ids:
            if w is None:
                label_ids.append(-100)
                bbox_ids.append([0, 0, 0, 0])

            elif w != prev:
                label_ids.append(LABEL_TO_ID[labels[w]])
                bbox_ids.append(bboxes[w])

            else:
                label_ids.append(-100)
                bbox_ids.append(bboxes[w])

            prev = w

        all_labels.append(label_ids)
        all_bboxes.append(bbox_ids)

    tokenized["labels"] = all_labels
    tokenized["bbox"] = all_bboxes
    return tokenized


def build_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = Dataset.from_dict({
        "tokens": [x["tokens"] for x in data],
        "bboxes": [x["bboxes"] for x in data],
        "labels": [x["labels"] for x in data],
    })

    return ds.map(tokenize_and_align, batched=True,
                  remove_columns=["tokens", "bboxes", "labels"])


# ── metrics ────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []

    for p, l in zip(preds, labels):
        tl, tp = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            tl.append(ID_TO_LABEL[li])
            tp.append(ID_TO_LABEL[pi])
        true_labels.append(tl)
        true_preds.append(tp)

    f1 = f1_score(true_labels, true_preds)
    print(classification_report(true_labels, true_preds))
    return {"f1": f1}


# ── load data ──────────────────────────────────────
train_ds = build_dataset(TRAIN_JSON)
test_ds  = build_dataset(TEST_JSON)

print("Train:", len(train_ds), "Test:", len(test_ds))

# ── model ──────────────────────────────────────────
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)

# ── training args (IMPORTANT PART) ─────────────────
args = TrainingArguments(
    output_dir=BEST_DIR,

    num_train_epochs=20,   # 👈 زودنا epochs

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,  # 👈 يحسن الاستقرار

    learning_rate=3e-5,   # 👈 أقل من 5e-5 = أفضل غالبًا

    weight_decay=0.01,

    lr_scheduler_type="cosine",

    warmup_steps=200,  # 👈 بدل warmup_ratio

    evaluation_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",

    fp16=torch.cuda.is_available(),

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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("🚀 Training...")
trainer.train()

# ── save ───────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved to:", OUTPUT_DIR)