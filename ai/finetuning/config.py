# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "microsoft/layoutlm-base-uncased"
SAVED_MODEL_DIR = "ai/model/saved_model"
BEST_MODEL_DIR  = "ai/model/best_model"

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_JSON = "ai/data/train.json"
TEST_JSON  = "ai/data/test.json"

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_LIST = [
    "O",
    "B-COMPANY", "I-COMPANY",
    "B-DATE",    "I-DATE",
    "B-TOTAL",   "I-TOTAL",
    "B-ADDRESS", "I-ADDRESS",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# ── Training ──────────────────────────────────────────────────────────────────
NUM_TRAIN_EPOCHS          = 15
PER_DEVICE_TRAIN_BATCH    = 16
PER_DEVICE_EVAL_BATCH     = 16
LEARNING_RATE             = 5e-5
WARMUP_RATIO              = 0.1
WEIGHT_DECAY              = 0.01
LR_SCHEDULER              = "cosine"
EARLY_STOPPING_PATIENCE   = 5
MAX_SEQ_LENGTH            = 512

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R          = 8       # rank — جربي 16 كـ stretch experiment
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1
LORA_TARGET     = ["query", "value"]

# ── Layout ────────────────────────────────────────────────────────────────────
PAGE_WIDTH  = 762
PAGE_HEIGHT = 1000