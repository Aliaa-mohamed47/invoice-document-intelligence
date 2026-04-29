# ai/finetuning/config.py
import os

BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "microsoft/layoutlm-base-uncased")
SAVED_MODEL_DIR = os.environ.get("SAVED_MODEL_DIR", "ai/model/saved_model")
BEST_MODEL_DIR  = os.environ.get("BEST_MODEL_DIR", "ai/model/best_model")

TRAIN_JSON = os.environ.get("TRAIN_JSON", "ai/data/train.json")
TEST_JSON  = os.environ.get("TEST_JSON", "ai/data/test.json")

LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]

LABEL2ID   = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL   = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

NUM_TRAIN_EPOCHS        = int(os.environ.get("NUM_TRAIN_EPOCHS", 20))
PER_DEVICE_TRAIN_BATCH  = int(os.environ.get("PER_DEVICE_TRAIN_BATCH", 8))
PER_DEVICE_EVAL_BATCH   = int(os.environ.get("PER_DEVICE_EVAL_BATCH", 8))
LEARNING_RATE           = float(os.environ.get("LEARNING_RATE", 3e-5))
WARMUP_STEPS            = int(os.environ.get("WARMUP_STEPS", 200))
WEIGHT_DECAY            = 0.01
LR_SCHEDULER            = "cosine"
EARLY_STOPPING_PATIENCE = 5
MAX_SEQ_LENGTH          = 512

LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
LORA_TARGET  = ["query", "value"]

PAGE_WIDTH  = 1000 
PAGE_HEIGHT = 1000