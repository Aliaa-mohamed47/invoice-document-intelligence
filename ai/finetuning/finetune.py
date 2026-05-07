import os, torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from transformers import (
    LayoutLMForTokenClassification, LayoutLMTokenizerFast, 
    DataCollatorForTokenClassification, Trainer, TrainingArguments
)
from config import * # استيراد الإعدادات

# 1. التجهيز السريع للبيانات
tokenizer = LayoutLMTokenizerFast.from_pretrained(BASE_MODEL_NAME)

def process_data(examples):
    tokenized = tokenizer(examples["tokens"], is_split_into_words=True, 
                          truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)
    
    labels, bboxes = [], []
    for i, word_ids in enumerate(tokenized.word_ids()):
        # تسوية الصناديق (Bboxes) وتجهيز الليبلز في سطر واحد
        norm_boxes = [[max(0, min(int(1000 * b / [PAGE_WIDTH, PAGE_HEIGHT, PAGE_WIDTH, PAGE_HEIGHT][j]), 1000)) 
                      for j, b in enumerate(box)] for box in examples["bboxes"][i]]
        
        labels.append([LABEL2ID[examples["labels"][i][w]] if w is not None else -100 for w in word_ids])
        bboxes.append([norm_boxes[w] if w is not None else [0,0,0,0] for w in word_ids])
        
    tokenized.update({"labels": labels, "bbox": bboxes})
    return tokenized

# تحميل البيانات مباشرة (بدون كود يدوي كتير)
dataset = load_dataset("json", data_files={"train": "data/train.json", "test": "data/test.json"})
train_ds = dataset["train"].filter(lambda x: any(l != "O" for l in x["labels"])).map(process_data, batched=True)
test_ds = dataset["test"].map(process_data, batched=True)

# 2. الموديل (الوزن بيتحط هنا مباشرة في Loss لو عايز تبسط، أو خليه تلقائي)
model = LayoutLMForTokenClassification.from_pretrained(BASE_MODEL_NAME, num_labels=NUM_LABELS)

# 3. إعدادات التدريب (مختصرة)
args = TrainingArguments(
    output_dir="./model/best_model", evaluation_strategy="epoch", save_strategy="epoch",
    learning_rate=LEARNING_RATE, num_train_epochs=NUM_TRAIN_EPOCHS, 
    load_best_model_at_end=True, metric_for_best_model="f1", fp16=torch.cuda.is_available()
)


trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=lambda p: {"f1": f1_score([[ID2LABEL[l] for l in label if l != -100] for label in p.label_ids], 
                                              [[ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(p.predictions.argmax(-1), p.label_ids)])}
)

trainer.train()
model.save_pretrained("./model/saved_model")
