"""
Visualize few-shot NER results:
  1. Learning curve: Micro F1 vs K
  2. Per-entity F1 heatmap
  3. Per-entity bar chart (grouped by K)
  4. Gap chart: Full vs 20-shot per entity
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

BASE    = Path(__file__).parent
OUT_DIR = BASE / "output/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES   = [("k1","1-shot",10), ("k5","5-shot",50),
           ("k10","10-shot",100), ("k20","20-shot",200),
           ("full","Full",5126)]
ENTITIES = ["AGE","DATE","GENDER","JOB","LOCATION","NAME",
            "ORGANIZATION","PATIENT_ID","SYMPTOM_AND_DISEASE","TRANSPORTATION"]
ENTITY_SHORT = ["AGE","DATE","GENDER","JOB","LOC","NAME","ORG","PAT_ID","SYM_DIS","TRANS"]

COLORS = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db"]
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data ────────────────────────────────────────────────────────────────
results = {}
for mode, label, train_size in MODES:
    path = BASE / f"output/baseline_{mode}/results.json"
    with open(path, encoding="utf-8") as f:
        results[label] = json.load(f)["report"]

k_labels   = [l for _, l, _ in MODES]
train_sizes = [s for _, _, s in MODES]
micro_f1   = [results[l]["micro avg"]["f1-score"] * 100 for l in k_labels]

# Per-entity F1 matrix  [entity x setting]
f1_matrix = np.array([
    [results[l].get(e, {}).get("f1-score", 0.0) * 100 for l in k_labels]
    for e in ENTITIES
])

# ── Figure 1: Learning Curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(k_labels, micro_f1, marker="o", linewidth=2.5, markersize=8,
        color="#2980b9", zorder=3)
for i, (x, y) in enumerate(zip(k_labels, micro_f1)):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

ax.fill_between(k_labels, micro_f1, alpha=0.1, color="#2980b9")
ax.axhline(y=micro_f1[-1], color="#e74c3c", linestyle="--", linewidth=1.5,
           label=f"Full training ({micro_f1[-1]:.1f}%)")
ax.set_xlabel("Training Setting", fontsize=12)
ax.set_ylabel("Micro F1 (%)", fontsize=12)
ax.set_title("Few-shot NER: Learning Curve (PhoBERT on PhoNER_COVID19)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "1_learning_curve.png", dpi=150)
plt.close()
print("Saved: 1_learning_curve.png")

# ── Figure 2: Heatmap ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
mask = f1_matrix == 0

sns.heatmap(f1_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=k_labels, yticklabels=ENTITY_SHORT,
            linewidths=0.5, ax=ax, vmin=0, vmax=100,
            cbar_kws={"label": "F1 Score (%)"})
# Overlay hatching for zero cells
for i in range(f1_matrix.shape[0]):
    for j in range(f1_matrix.shape[1]):
        if f1_matrix[i, j] < 1.0:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                         hatch="///", edgecolor="gray", lw=0))

ax.set_title("Per-entity F1 Score (%) across Few-shot Settings", fontsize=13, fontweight="bold")
ax.set_xlabel("Training Setting", fontsize=12)
ax.set_ylabel("Entity Type", fontsize=12)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig(OUT_DIR / "2_heatmap.png", dpi=150)
plt.close()
print("Saved: 2_heatmap.png")

# ── Figure 3: Grouped Bar Chart ──────────────────────────────────────────────
n_entities = len(ENTITIES)
n_settings = len(MODES)
x = np.arange(n_entities)
width = 0.15

fig, ax = plt.subplots(figsize=(14, 6))
for i, (label, color) in enumerate(zip(k_labels, COLORS)):
    bars = ax.bar(x + i * width, f1_matrix[:, i], width,
                  label=label, color=color, alpha=0.85, edgecolor="white")

ax.set_xlabel("Entity Type", fontsize=12)
ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_title("Per-entity F1 Score by Training Setting", fontsize=13, fontweight="bold")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(ENTITY_SHORT, fontsize=10)
ax.set_ylim(0, 115)
ax.legend(title="Setting", fontsize=10)
ax.axhline(y=90, color="gray", linestyle=":", linewidth=1, alpha=0.7)
plt.tight_layout()
plt.savefig(OUT_DIR / "3_grouped_bar.png", dpi=150)
plt.close()
print("Saved: 3_grouped_bar.png")

# ── Figure 4: Gap Chart (Full vs 20-shot) ────────────────────────────────────
full_f1    = f1_matrix[:, -1]   # Full
shot20_f1  = f1_matrix[:, -2]   # 20-shot
gap        = full_f1 - shot20_f1
sort_idx   = np.argsort(gap)[::-1]

fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#e74c3c" if g > 10 else "#f39c12" if g > 5 else "#2ecc71"
              for g in gap[sort_idx]]
bars = ax.barh([ENTITY_SHORT[i] for i in sort_idx], gap[sort_idx],
               color=bar_colors, edgecolor="white", height=0.6)

for bar, val in zip(bars, gap[sort_idx]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"+{val:.1f}pp", va="center", fontsize=10)

ax.set_xlabel("F1 Gap (Full - 20-shot) in percentage points", fontsize=12)
ax.set_title("How much does full data help beyond 20-shot?", fontsize=13, fontweight="bold")
ax.axvline(x=5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="5pp threshold")
red_patch   = mpatches.Patch(color="#e74c3c", label="Gap > 10pp (needs more data)")
orange_patch= mpatches.Patch(color="#f39c12", label="Gap 5-10pp")
green_patch = mpatches.Patch(color="#2ecc71", label="Gap < 5pp (20-shot sufficient)")
ax.legend(handles=[red_patch, orange_patch, green_patch], fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "4_gap_full_vs_20shot.png", dpi=150)
plt.close()
print("Saved: 4_gap_full_vs_20shot.png")

print(f"\nAll figures saved to: {OUT_DIR}")
