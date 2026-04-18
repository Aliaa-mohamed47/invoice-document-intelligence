import gc, json, os, sys, shutil
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

# تنظيف الذاكرة قبل البدء
gc.collect()
torch.cuda.empty_cache()

# ── 1. المسارات (Paths) ────────────────────────────────────────────────────────
BASE_DIR   = "/content/invoice-document-intelligence"
TRAIN_JSON = f"{BASE_DIR}/ai/data/train.json"
TEST_JSON  = f"{BASE_DIR}/ai/data/test.json"
OUTPUT_DIR = f"{BASE_DIR}/ai/model/saved_model"
BEST_DIR   = f"{BASE_DIR}/ai/model/best_model"

# تنظيف المجلدات القديمة لضمان Fresh Start
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
if os.path.exists(BEST_DIR):   shutil.rmtree(BEST_DIR)

# ── 2. تعريف الـ Labels ────────────────────────────────────────────────────────
LABEL_LIST  = ["O","B-COMPANY","I-COMPANY","B-DATE","I-DATE",
               "B-TOTAL","I-TOTAL","B-ADDRESS","I-ADDRESS"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: l for l, i in LABEL_TO_ID.items()}

tokenizer   = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000

# ── 3. دوال المساعدة (Helpers) ─────────────────────────────────────────────────
def normalize_bbox(bbox, w=PAGE_WIDTH, h=PAGE_HEIGHT):
    x0, y0, x1, y1 = bbox
    # تحويل الإحداثيات لمدى [0, 1000] وهو المطلوب لـ LayoutLM
    nx0 = max(0, min(int(1000 * x0 / w), 1000))
    ny0 = max(0, min(int(1000 * y0 / h), 1000))
    nx1 = max(0, min(int(1000 * x1 / w), 1000))
    ny1 = max(0, min(int(1000 * x1 / h), 1000))
    return [nx0, ny0, nx1, ny1]

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=512
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
                label_ids.append(LABEL_TO_ID[labels[word_id]])
                bbox_ids.append(norm[word_id])
            else:
                # الـ Sub-tokens بنحطلهم -100 عشان الـ Loss ميتأثرش بيهم
                label_ids.append(-100)
                bbox_ids.append(norm[word_id])
            prev_word_id = word_id

        all_labels.append(label_ids)
        all_bboxes.append(bbox_ids)

    tokenized["labels"] = all_labels
    tokenized["bbox"] = all_bboxes
    return tokenized

# ── 4. بناء الـ Datasets ───────────────────────────────────────────────────────
def build_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    ds = Dataset.from_dict({
        "tokens": [r["tokens"] for r in records],
        "bboxes": [r["bboxes"]  for r in records],
        "labels": [r["labels"]  for r in records],
    })
    return ds.map(tokenize_and_align, batched=True, remove_columns=["tokens","bboxes","labels"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    true_labels, true_preds = [], []
    for pr, lr in zip(preds, labels):
        tl, tp = [], []
        for p, l in zip(pr, lr):
            if l == -100: continue
            tl.append(ID_TO_LABEL[l]); tp.append(ID_TO_LABEL[p])
        true_labels.append(tl); true_preds.append(tp)
    return {"f1": f1_score(true_labels, true_preds)}

# تجهيز البيانات
print("🔄 Loading and Tokenizing data...")
train_ds = build_dataset(TRAIN_JSON)
test_ds  = build_dataset(TEST_JSON)

# ── 5. إعداد الموديل والتدريب ──────────────────────────────────────────────────
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(LABEL_LIST),
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)

args = TrainingArguments(
    output_dir=BEST_DIR,
    num_train_epochs=15,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=20,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("🚀 Starting fine-tuning...")
trainer.train()

# ── 6. حفظ الموديل النهائي ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Model saved successfully to {OUTPUT_DIR}")
