# ai/finetuning/train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model import InvoiceLayoutLMDataset, build_model, get_tokenizer

# Dynamically resolve project base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "ai", "data")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "ai", "model", "saved_model")
RESULTS_DIR = os.path.join(BASE_DIR, "ai", "model", "results")

def main():
    train_path = os.path.join(DATA_DIR, "train.json")
    test_path  = os.path.join(DATA_DIR, "test.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Training data missing in {DATA_DIR}. Run clean_data.py first.")

    print("Loading datasets...")
    train_ds = InvoiceLayoutLMDataset(train_path)
    test_ds  = InvoiceLayoutLMDataset(test_path)

    print("Initializing model...")
    model = build_model()
    tokenizer = get_tokenizer()

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        per_device_train_batch_size=4, # Increased for better stability, depends on GPU memory
        per_device_eval_batch_size=4,
        num_train_epochs=5,            # Increased epochs since we have Early Stopping
        logging_steps=10,
        save_steps=50,
        eval_strategy="epoch",         # Updated from deprecated evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving finalized model to {MODEL_OUT_DIR}...")
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)

    print("✅ Training complete. Model is AWS-ready.")

if __name__ == "__main__":
    main()