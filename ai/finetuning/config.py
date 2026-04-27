# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "microsoft/layoutlm-base-uncased"
SAVED_MODEL_DIR = "ai/model/saved_model"
BEST_MODEL_DIR  = "ai/model/best_model"

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_JSON = "ai/data/train.json"
TEST_JSON  = "ai/data/test.json"


LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

# ── Training ──────────────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS        = 15
PER_DEVICE_TRAIN_BATCH  = 16
PER_DEVICE_EVAL_BATCH   = 16
LEARNING_RATE           = 3e-5
WARMUP_STEPS            = 100
WEIGHT_DECAY            = 0.01
LR_SCHEDULER            = "cosine"
EARLY_STOPPING_PATIENCE = 5
MAX_SEQ_LENGTH          = 512

# ── LoRA (stretch experiment) ─────────────────────────────────────────────────
LORA_R       = 8        
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
LORA_TARGET  = ["query", "value"]

# ── Layout ────────────────────────────────────────────────────────────────────
PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000