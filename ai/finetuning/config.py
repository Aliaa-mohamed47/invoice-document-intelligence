# ai/finetuning/config.py  ── FIXED VERSION
# ─────────────────────────────────────────────────────────────────────────────

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
LABEL2ID   = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL   = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)

# ── Training ──────────────────────────────────────────────────────────────────
# FIX: better hyperparams for reaching 85-90% accuracy
NUM_TRAIN_EPOCHS        = 20          # was 15 — more epochs needed
PER_DEVICE_TRAIN_BATCH  = 8           # was 16 — smaller batch = better generalization
PER_DEVICE_EVAL_BATCH   = 8
LEARNING_RATE           = 3e-5        # was 2e-5 — slightly higher for LayoutLM
WARMUP_STEPS            = 200         # was 100 — more warmup for stability
WEIGHT_DECAY            = 0.01
LR_SCHEDULER            = "cosine"
EARLY_STOPPING_PATIENCE = 5           # was 3 — give more chances before stopping
MAX_SEQ_LENGTH          = 512

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
LORA_TARGET  = ["query", "value"]

# ── Layout ────────────────────────────────────────────────────────────────────
# FIX: SROIE actual image dimensions (not 762!)
# SROIE images are scanned receipts, typically ~1200x1600 or similar
# We set these as DEFAULT fallback only — actual dims should come from the image
PAGE_WIDTH  = 1000     # normalized coordinate space (LayoutLM standard)
PAGE_HEIGHT = 1000     # normalized coordinate space (LayoutLM standard)
# NOTE: In inference, always pass actual img_width/img_height to normalize_bbox