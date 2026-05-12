
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "evaluation_results")
BASELINE_OUT  = os.path.join(RESULTS_DIR, "baseline_results.json")
FINETUNED_OUT = os.path.join(RESULTS_DIR, "finetuned_results.json")
CHARTS_DIR    = os.path.join(RESULTS_DIR, "charts")

FIELDS = ["company", "date", "total", "address"]

os.makedirs(CHARTS_DIR, exist_ok=True)


# ── Loader ─────────────────────────────────────────────────────────────────
def load(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[!] {path} not found.\n"
            f"    Run evaluate.py and generate_baseline.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


baseline  = load(BASELINE_OUT)
finetuned = load(FINETUNED_OUT)


# ── Chart 1: F1 Comparison — Baseline vs Fine-tuned ───────────────────────
base_f1 = [baseline["per_field"][f]["f1"]  for f in FIELDS]
ft_f1   = [finetuned["per_field"][f]["f1"] for f in FIELDS]

x   = np.arange(len(FIELDS))
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - 0.2, base_f1, 0.38, label="Baseline",   color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + 0.2, ft_f1,   0.38, label="Fine-tuned", color="#2ecc71", alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels([f.capitalize() for f in FIELDS], fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_title("F1 Score: Baseline vs Fine-tuned", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path1 = os.path.join(CHARTS_DIR, "f1_comparison.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"[✓] Saved → {path1}")


# ── Chart 2: Precision / Recall / F1 per field (fine-tuned only) ──────────
metrics_keys = ["precision", "recall", "f1"]
colors       = ["#3498db", "#e67e22", "#2ecc71"]

fig, ax = plt.subplots(figsize=(11, 6))
width   = 0.25

for i, (metric, color) in enumerate(zip(metrics_keys, colors)):
    vals = [finetuned["per_field"][f][metric] for f in FIELDS]
    bars = ax.bar(x + (i - 1) * width, vals, width,
                  label=metric.capitalize(), color=color, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f.capitalize() for f in FIELDS], fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Fine-tuned Model: Precision / Recall / F1 per Field",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path2 = os.path.join(CHARTS_DIR, "metrics_per_field.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"[✓] Saved → {path2}")

print("\n[✓] All charts generated in evaluation_results/charts/")