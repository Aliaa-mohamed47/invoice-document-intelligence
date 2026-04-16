# سما
"""
pipeline.py
-----------
Inference pipeline for the BERT invoice NER model.

Takes a list of tokens (words from one invoice) and returns
the predicted BIO label for each token.

Usage:
    from pipeline import InvoiceNERPipeline

    pipe = InvoiceNERPipeline("path/to/saved_model")
    tokens = ["TAN", "WOON", "YANN", "BOOK", "TA", "DATE:", "25/12/2018", "TOTAL:", "9.00"]
    results = pipe.predict(tokens)
    # [{"token": "TAN",      "label": "B-COMPANY"},
    #  {"token": "WOON",     "label": "I-COMPANY"},
    #  {"token": "25/12/2018","label": "B-DATE"},
    #  {"token": "9.00",     "label": "B-TOTAL"}, ...]
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification

from model import LABEL2ID, ID2LABEL, NUM_LABELS, TOKENIZER, build_model


# ─────────────────────────────────────────────
# Pipeline class
# ─────────────────────────────────────────────

class InvoiceNERPipeline:
    """
    End-to-end inference pipeline:

        tokens (list[str])
            ↓  BertTokenizerFast  (is_split_into_words=True)
        input_ids + attention_mask
            ↓  BertForTokenClassification
        logits  shape: (1, seq_len, num_labels)
            ↓  argmax(-1)
        predicted label id per subword token
            ↓  keep only first subword per word, map id → label string
        predicted label per original word token
    """

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path : path to a saved model directory (from Trainer or torch.save).
                         Pass None to load the base pretrained weights (for testing only).
            device     : "cpu" | "cuda" | None (auto-detect)
        """
        # ── Device ────────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = TOKENIZER

        # ── Model ─────────────────────────────────────────────────────────
        if model_path:
            # Load a fine-tuned model saved with trainer.save_model() or torch.save()
            self.model = BertForTokenClassification.from_pretrained(
                model_path,
                num_labels=NUM_LABELS,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        else:
            # Base weights only — useful for testing the pipeline structure
            self.model = build_model()

        self.model.to(self.device)
        self.model.eval()  # disable dropout

    # ─────────────────────────────────────────────
    # Core predict method
    # ─────────────────────────────────────────────

    def predict(self, tokens, max_length=512):
        """
        Predict BIO labels for a single invoice (list of word tokens).

        Args:
            tokens     : list[str] — pre-split words from one invoice
            max_length : int       — max BERT sequence length (default 512)

        Returns:
            list[dict] with keys "token" and "label" for every input token.
            Example:
                [
                    {"token": "TAN",       "label": "B-COMPANY"},
                    {"token": "WOON",      "label": "I-COMPANY"},
                    {"token": "25/12/2018","label": "B-DATE"},
                    {"token": "9.00",      "label": "B-TOTAL"},
                    {"token": "THANK",     "label": "O"},
                ]
        """
        # ── 1. Tokenise ───────────────────────────────────────────────────
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding=False,          # no padding needed for single inference
            max_length=max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)   # None | int per subword

        # Move tensors to the right device
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # ── 2. Forward pass ───────────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # logits: (1, seq_len, num_labels)  →  predicted id per subword: (seq_len,)
        predicted_ids = outputs.logits.argmax(dim=-1).squeeze(0).tolist()

        # ── 3. Map predictions back to original words ─────────────────────
        # We only keep the prediction for the FIRST subword of each word.
        # Subword continuations (##...) and special tokens are skipped.
        results = []
        seen_words = set()

        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                # [CLS] or [SEP] — skip
                continue
            if word_idx in seen_words:
                # Continuation subword — skip
                continue

            seen_words.add(word_idx)
            predicted_label = ID2LABEL[predicted_ids[subword_idx]]

            results.append({
                "token": tokens[word_idx],
                "label": predicted_label,
            })

        return results

    # ─────────────────────────────────────────────
    # Helper: extract structured entities
    # ─────────────────────────────────────────────

    def extract_entities(self, tokens, max_length=512):
        """
        Run predict() and group consecutive B-/I- tokens into entities.

        Returns:
            dict with keys: "company", "date", "total", "address"
            Each value is a string (joined tokens) or None if not found.

        Example:
            {
                "company": "TAN WOON YANN BOOK TA",
                "date":    "25/12/2018",
                "total":   "9.00",
                "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA",
            }
        """
        predictions = self.predict(tokens, max_length=max_length)

        entities = {"company": [], "date": [], "total": [], "address": []}
        entity_map = {
            "COMPANY": "company",
            "DATE":    "date",
            "TOTAL":   "total",
            "ADDRESS": "address",
        }

        for pred in predictions:
            label = pred["label"]
            token = pred["token"]

            if label == "O":
                continue

            # label is like "B-COMPANY" or "I-DATE"
            _, entity_type = label.split("-", 1)
            key = entity_map.get(entity_type)
            if key:
                entities[key].append(token)

        # Join tokens into strings; return None for missing entities
        return {
            k: " ".join(v) if v else None
            for k, v in entities.items()
        }


# ─────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json, os

    # Load one real invoice from the test set
    base      = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(base, "../data/test.json")

    with open(test_path) as f:
        test_data = json.load(f)

    sample = test_data[0]
    tokens = sample["tokens"]
    true_labels = sample["labels"]

    print(f"Invoice ID : {sample['id']}")
    print(f"Tokens     : {len(tokens)}")
    print(f"First 10 tokens: {tokens[:10]}")

    # ── Build pipeline (base weights — no fine-tuning yet) ─────────────────
    # After fine-tuning, replace None with the path to your saved model:
    #   pipe = InvoiceNERPipeline("ai/model/saved_model")
    pipe = InvoiceNERPipeline(model_path=None)
    print(f"\nDevice: {pipe.device}")

    # ── Token-level predictions ────────────────────────────────────────────
    results = pipe.predict(tokens)
    print("\nToken-level predictions (first 15):")
    print(f"{'Token':<20} {'Predicted':<15} {'True':<15}")
    print("-" * 50)
    for i, r in enumerate(results[:15]):
        print(f"{r['token']:<20} {r['label']:<15} {true_labels[i]:<15}")

    # ── Entity extraction ──────────────────────────────────────────────────
    entities = pipe.extract_entities(tokens)
    print("\nExtracted entities:")
    for key, val in entities.items():
        print(f"  {key:<10}: {val}")

    print("\nPipeline demo complete.")