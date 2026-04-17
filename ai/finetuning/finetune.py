#الاء كمال
"""
finetune_lora.py  —  feature/finetuning branch
------------------------------------------------
Fine-tunes bert-base-multilingual-cased with LoRA (PEFT)
for invoice NER: COMPANY, DATE, TOTAL, ADDRESS.

Fixes applied vs previous version:
  1. learning_rate: 3e-5  →  3e-4   (most critical — LoRA needs a high LR)
  2. LoRA rank: 16 → 32, targets add "key", bias="lora_only"
  3. warmup_ratio=0.1 replaces fixed warmup_steps=100
  4. early_stopping_patience: 3 → 5
  5. lora_dropout: 0.1 → 0.05  (small dataset — less dropout helps)
"""

import json
import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score, classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_JSON = "/content/invoice-document-intelligence/ai/data/train.json"
TEST_JSON  = "/content/invoice-document-intelligence/ai/data/test.json"
OUTPUT_DIR = "/content/invoice-document-intelligence/ai/model/saved_model"
BEST_DIR   = "/content/invoice-document-intelligence/ai/model/best_model"

# ── Labels ─────────────────────────────────────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-ADDRESS", "I-ADDRESS",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased", use_fast=True
)


# ── Data helpers ───────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512,
    )
    all_labels = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(LABEL_TO_ID[labels[wid]])
            else:
                lbl = labels[wid]
                if lbl.startswith("B-"):
                    lbl = "I-" + lbl[2:]
                label_ids.append(LABEL_TO_ID[lbl])
            prev = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def build_dataset(records):
    ds = Dataset.from_dict({
        "tokens": [r["tokens"] for r in records],
        "labels": [r["labels"] for r in records],
    })
    return ds.map(
        tokenize_and_align, batched=True,
        remove_columns=["tokens", "labels"]
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pred_row, label_row in zip(predictions, labels):
        tl, tp = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            tl.append(ID_TO_LABEL[l])
            tp.append(ID_TO_LABEL[p])
        true_labels.append(tl)
        true_preds.append(tp)
    return {"f1": f1_score(true_labels, true_preds)}


# ── Load data ──────────────────────────────────────────────────────────────────
train_records = load_json(TRAIN_JSON)
test_records  = load_json(TEST_JSON)
train_ds = build_dataset(train_records)
test_ds  = build_dataset(test_records)

# ── Base model ─────────────────────────────────────────────────────────────────
base_model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased",
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
    ignore_mismatched_sizes=True,
)

# ── LoRA config  (FIX #2) ──────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=32,                                        # was 16 → more capacity
    lora_alpha=64,                               # keep 2× of r
    lora_dropout=0.05,                           # was 0.1 → less dropout for small dataset
    bias="lora_only",                            # was "none" → also train bias terms
    target_modules=["query", "key", "value"],    # was query+value → add key
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params ~1.2M (0.67%) vs 177M total

# ── Training args  (FIX #1 + #3 + #4) ─────────────────────────────────────────
args = TrainingArguments(
    output_dir=BEST_DIR,
    num_train_epochs=15,                         # more epochs — early stopping handles it
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,              # effective batch = 32
    learning_rate=3e-4,                          # FIX #1: was 3e-5 — most important change
    warmup_ratio=0.1,                            # FIX #3: was warmup_steps=100 (too short)
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8
    ),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # FIX #4: was 3
)

trainer.train()

# ── Evaluate ───────────────────────────────────────────────────────────────────
preds_output = trainer.predict(test_ds)
logits, labels = preds_output.predictions, preds_output.label_ids
predictions = np.argmax(logits, axis=-1)
true_labels, true_preds = [], []
for pred_row, label_row in zip(predictions, labels):
    tl, tp = [], []
    for p, l in zip(pred_row, label_row):
        if l == -100:
            continue
        tl.append(ID_TO_LABEL[l])
        tp.append(ID_TO_LABEL[p])
    true_labels.append(tl)
    true_preds.append(tp)

print(classification_report(true_labels, true_preds, digits=4))

# ── Save — merge LoRA weights back into base model ─────────────────────────────
merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Fine-tuned model with LoRA saved to", OUTPUT_DIR)
