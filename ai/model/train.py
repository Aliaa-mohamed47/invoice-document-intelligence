import os
from transformers import LayoutLMForTokenClassification, Trainer, TrainingArguments
from model import InvoiceLayoutLMDataset, build_model, TOKENIZER
# paths
base = r"D:\invoice-intelligence"

train_ds = InvoiceLayoutLMDataset(os.path.join(base, "ai/data/train.json"))
test_ds  = InvoiceLayoutLMDataset(os.path.join(base, "ai/data/test.json"))

model = build_model()

training_args = TrainingArguments(
    output_dir=r"D:\invoice-intelligence\ai\model\results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

trainer.save_model(r"D:\invoice-intelligence\ai\model\saved_model")
TOKENIZER.save_pretrained(r"D:\invoice-intelligence\ai\model\saved_model")

print("✅ Model saved!")