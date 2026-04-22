# config.py

import os

# ── Model paths ──────────────────────────────────────────────────────────────
COLAB_MODEL_PATH = "/content/invoice-document-intelligence/ai/model/saved_model"
LOCAL_MODEL_PATH = "ai/model/saved_model"

# Auto-detects whether running on Colab or locally
MODEL_PATH = COLAB_MODEL_PATH if os.path.exists(COLAB_MODEL_PATH) else LOCAL_MODEL_PATH

BASE_MODEL_NAME = "bert-base-multilingual-cased"

# ── Data paths ───────────────────────────────────────────────────────────────
TRAIN_JSON = "ai/data/train.json"
TEST_JSON  = "ai/data/test.json"

# ── Evaluation output ────────────────────────────────────────────────────────
RESULTS_DIR      = "evaluation_results"
FINETUNED_OUTPUT = f"{RESULTS_DIR}/finetuned_results.json"
BASELINE_OUTPUT  = f"{RESULTS_DIR}/baseline_results.json"

# ── Labels (must match finetune.py exactly) ──────────────────────────────────
LABEL_LIST = [
    "O",
    "B-ADDRESS", "I-ADDRESS",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── Fields (lowercase — used across evaluate.py and results_to_json.py) ──────
FIELDS = ["company", "date", "total", "address"]
