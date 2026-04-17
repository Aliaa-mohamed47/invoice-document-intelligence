# pipeline.py
"""
pipeline.py
-----------
Base model inference pipeline for invoice NER.
Loads bert-base-multilingual-cased and runs token classification.
"""

import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from model import LABEL_LIST, LABEL_TO_ID, ID_TO_LABEL, TOKENIZER

BASE_MODEL_NAME = "bert-base-multilingual-cased"


def load_base_model():
    """Load the pretrained base model (no fine-tuning)."""
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model


def load_finetuned_model(model_path: str):
    """Load a fine-tuned model from disk."""
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return model


def predict(tokens: list[str], model, tokenizer=TOKENIZER) -> list[dict]:
    """
    Run NER inference on a list of tokens.
    Returns list of {token, label} dicts.
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = encoding.word_ids()

    results = []
    seen = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        results.append({
            "token": tokens[word_id],
            "label": ID_TO_LABEL[predictions[idx]],
        })
    return results


def extract_entities(tokens: list[str], model) -> dict:
    """
    Convert NER predictions into structured entity dict.
    Returns: {company, date, total, address}
    """
    predictions = predict(tokens, model)

    entities = {"company": [], "date": [], "total": [], "address": []}
    label_map = {
        "COMPANY": "company",
        "DATE": "date",
        "TOTAL": "total",
        "ADDRESS": "address",
    }

    for item in predictions:
        label = item["label"]
        if label == "O":
            continue
        prefix, entity_type = label.split("-", 1)
        key = label_map.get(entity_type)
        if key:
            entities[key].append(item["token"])

    # Join tokens into strings
    return {k: " ".join(v) for k, v in entities.items()}


if __name__ == "__main__":
    # Quick test with dummy tokens
    sample_tokens = ["KEDAI", "GUNTING", "RAMBUT", "Date:", "01/01/2023",
                     "Total:", "RM", "25.00", "No", "12", "Jalan", "Maju"]

    model = load_base_model()
    result = extract_entities(sample_tokens, model)
    print("Extracted entities:", json.dumps(result, indent=2))
