# سما
"""
model.py
--------
BERT-based Token Classification model for invoice NER.
Handles: label definitions, dataset class, tokenization, and label alignment.

Labels (BIO scheme):
    O, B-COMPANY, I-COMPANY, B-DATE, I-DATE,
    B-TOTAL, I-TOTAL, B-ADDRESS, I-ADDRESS
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification


# ─────────────────────────────────────────────
# 1. Label definitions
# ─────────────────────────────────────────────

LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

NUM_LABELS = len(LABEL_LIST)  # 9


# ─────────────────────────────────────────────
# 2. Tokenizer
# ─────────────────────────────────────────────
# We use bert-base-cased (NOT uncased) because invoices contain
# company names, addresses, and amounts where capitalisation matters.

TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-cased")


# ─────────────────────────────────────────────
# 3. Label alignment
# ─────────────────────────────────────────────

def align_labels_with_tokens(labels, word_ids):
    """
    BERT's WordPiece tokeniser splits words into subword tokens.
    Example:  "JOHOR"  →  ["JO", "##HOR"]

    Rules:
      - [CLS] / [SEP] tokens  →  -100  (ignored by loss)
      - First subword of a word →  real label
      - Continuation subwords  →  -100  (ignored by loss)

    Args:
        labels   : list[str]  — BIO label per original token, e.g. ["B-COMPANY", "I-COMPANY", "O"]
        word_ids : list[int|None] — output of encoding.word_ids()

    Returns:
        list[int] — aligned label ids, same length as word_ids
    """
    aligned = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            # Special token [CLS] or [SEP]
            aligned.append(-100)
        elif word_idx != previous_word_idx:
            # First subword of a new word → assign the real label
            aligned.append(LABEL2ID[labels[word_idx]])
        else:
            # Continuation subword → ignore in loss
            aligned.append(-100)

        previous_word_idx = word_idx

    return aligned


# ─────────────────────────────────────────────
# 4. Dataset class
# ─────────────────────────────────────────────

class InvoiceNERDataset(Dataset):
    """
    PyTorch Dataset for the SROIE invoice NER task.

    Each sample in the JSON file has the shape:
        {
            "id":     str,
            "tokens": ["TAN", "WOON", ...],
            "labels": ["O", "O", ..., "B-COMPANY", ...]
        }

    The dataset tokenises each sample with BERT's WordPiece tokeniser
    and aligns the BIO labels to the resulting subword sequence.
    """

    def __init__(self, json_path, tokenizer=TOKENIZER, max_length=512):
        """
        Args:
            json_path  : path to train.json or test.json
            tokenizer  : BertTokenizerFast instance
            max_length : maximum sequence length for BERT (default 512)
        """
        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample["tokens"]   # already split words, e.g. ["TAN", "WOON", ...]
        labels = sample["labels"]   # BIO strings,          e.g. ["O",   "O",   ...]

        # ── Tokenise ──────────────────────────────────────────────────────
        # is_split_into_words=True tells the tokeniser that `tokens` is
        # already a list of words, not a raw string to be split.
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ── Align labels ──────────────────────────────────────────────────
        word_ids   = encoding.word_ids(batch_index=0)
        label_ids  = align_labels_with_tokens(labels, word_ids)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "labels":         torch.tensor(label_ids, dtype=torch.long),  # (max_length,)
        }


# ─────────────────────────────────────────────
# 5. Model
# ─────────────────────────────────────────────

def build_model():
    """
    Load bert-base-cased with a Token Classification head on top.

    The head is a linear layer:
        hidden_size (768)  →  num_labels (9)

    It is randomly initialised — fine-tuning trains it from scratch
    while the BERT encoder weights start from the pretrained checkpoint.

    Returns:
        BertForTokenClassification
    """
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base, "../data/train.json")
    test_path  = os.path.join(base, "../data/test.json")

    # Dataset
    train_ds = InvoiceNERDataset(train_path)
    test_ds  = InvoiceNERDataset(test_path)
    print(f"Train samples : {len(train_ds)}")
    print(f"Test  samples : {len(test_ds)}")

    # Single sample
    sample = train_ds[0]
    print(f"\ninput_ids shape      : {sample['input_ids'].shape}")
    print(f"attention_mask shape : {sample['attention_mask'].shape}")
    print(f"labels shape         : {sample['labels'].shape}")

    # Non-ignored labels
    real = sample["labels"][sample["labels"] != -100]
    print(f"Real label count     : {len(real)}")
    print(f"Label ids sample     : {real[:10].tolist()}")
    print(f"Label names sample   : {[ID2LABEL[i.item()] for i in real[:10]]}")

    # Model
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters     : {total_params:,}")
    print(f"Classification head  : hidden(768) → labels({NUM_LABELS})")
    print("\nSanity check passed.")