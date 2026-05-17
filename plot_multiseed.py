"""
Ve bieu do ket qua multi-seed few-shot NER
Can chay sau khi run_multiseed.py hoan thanh
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
IN_PATH = BASE / "output/multiseed_results.json"
OUT_DIR = BASE / "output/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES      = ["AGE","DATE","GENDER","JOB","LOCATION","NAME",
                 "ORGANIZATION","PATIENT_ID","SYMPTOM_AND_DISEASE","TRANSPORTATION"]
ENTITY_SHORT  = ["AGE","DATE","GENDER","JOB","LOC","NAME","ORG","PAT_ID","SYM_DIS","TRANS"]
MODES_LABEL   = ["5-shot", "10-shot", "20-shot", "Full"]
FULL_F1       = 94.23
FULL_PER_ENTITY = {          # seed=42 baseline full
    "AGE": 96.4, "DATE": 98.6, "GENDER": 94.0, "JOB": 70.1,
    "LOCATION": 94.1, "NAME": 92.6, "ORGANIZATION": 88.6,
    "PATIENT_ID": 98.1, "SYMPTOM_AND_DISEASE": 88.0, "TRANSPORTATION": 97.9,
}
COLORS = ["#e67e22", "#f1c40f", "#2ecc71", "#3498db"]

sns.set_theme(style="whitegrid", font_scale=1.1)


def load():
    with open(IN_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Figure 5: Learning Curve with Error Bars ────────────────────────────────
def plot_learning_curve(data):
    modes   = ["k5", "k10", "k20"]
    means   = [data[m]["micro_f1_mean"] for m in modes] + [FULL_F1]
    stds    = [data[m]["micro_f1_std"]  for m in modes] + [0.0]
    x_labels = MODES_LABEL

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x_labels, means, yerr=stds,
                fmt="o-", linewidth=2.5, markersize=8,
                capsize=6, capthick=2, elinewidth=2,
                color="#2980b9", ecolor="#e74c3c", zorder=3)

    for x, m, s in zip(x_labels, means, stds):
        label = f"{m:.1f}%" if s == 0 else f"{m:.1f}\n±{s:.1f}"
        ax.annotate(label, (x, m), textcoords="offset points",
                    xytext=(0, 14), ha="center", fontsize=9, fontweight="bold")

    ax.fill_between(x_labels, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)], alpha=0.15, color="#2980b9")
    ax.axhline(y=FULL_F1, color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Full training ({FULL_F1}%)")

    ax.set_xlabel("Training Setting", fontsize=12)
    ax.set_ylabel("Micro F1 (%)", fontsize=12)
    ax.set_title("Few-shot NER Learning Curve with Std Dev\n(PhoBERT, seeds=[1,42,100])",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_learning_curve_multiseed.png", dpi=150)
    plt.close()
    print("Saved: 5_learning_curve_multiseed.png")


# ── Figure 6: Per-entity F1 with Error Bars (grouped) ───────────────────────
def plot_per_entity_errorbars(data):
    modes       = ["k5", "k10", "k20"]
    n_entities  = len(ENTITIES)
    n_modes     = len(modes) + 1   # +1 for Full
    x           = np.arange(n_entities)
    width       = 0.18

    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (mode, label, color) in enumerate(zip(modes, MODES_LABEL[:3], COLORS[:3])):
        means = [data[mode]["per_entity"][e]["mean"] for e in ENTITIES]
        stds  = [data[mode]["per_entity"][e]["std"]  for e in ENTITIES]
        ax.bar(x + i * width, means, width, label=label,
               color=color, alpha=0.85, edgecolor="white")
        ax.errorbar(x + i * width, means, yerr=stds,
                    fmt="none", ecolor="black", capsize=3, capthick=1.2, elinewidth=1.2)

    # Full bar (no error bar)
    full_vals = [FULL_PER_ENTITY[e] for e in ENTITIES]
    ax.bar(x + 3 * width, full_vals, width, label="Full",
           color=COLORS[3], alpha=0.85, edgecolor="white")

    ax.set_xlabel("Entity Type", fontsize=12)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-entity F1 with Std Dev (seeds=[1,42,100])", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(ENTITY_SHORT, fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(y=90, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.legend(title="Setting", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "6_per_entity_multiseed.png", dpi=150)
    plt.close()
    print("Saved: 6_per_entity_multiseed.png")


# ── Figure 7: Variance heatmap — std per entity per K ───────────────────────
def plot_variance_heatmap(data):
    modes  = ["k5", "k10", "k20"]
    labels = ["5-shot", "10-shot", "20-shot"]

    std_matrix = np.array([
        [data[m]["per_entity"][e]["std"] for m in modes]
        for e in ENTITIES
    ])

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(std_matrix, annot=True, fmt=".1f", cmap="Oranges",
                xticklabels=labels, yticklabels=ENTITY_SHORT,
                linewidths=0.5, ax=ax, vmin=0,
                cbar_kws={"label": "Std Dev (%)"})
    ax.set_title("Variance (Std Dev) across Seeds\n— Higher = less stable",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Setting", fontsize=11)
    ax.set_ylabel("Entity Type", fontsize=11)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "7_variance_heatmap.png", dpi=150)
    plt.close()
    print("Saved: 7_variance_heatmap.png")


# ── Figure 8: Summary table as figure ───────────────────────────────────────
def plot_summary_table(data):
    modes  = ["k5", "k10", "k20"]
    labels = ["5-shot", "10-shot", "20-shot", "Full (seed=42)"]

    col_labels = ["Setting", "Train size", "Micro F1", "Std Dev", "vs Full"]
    train_sizes = [50, 100, 200, 5126]
    rows = []
    for mode, label, size in zip(modes, labels[:3], train_sizes[:3]):
        m = data[mode]["micro_f1_mean"]
        s = data[mode]["micro_f1_std"]
        diff = m - FULL_F1
        rows.append([label, str(size), f"{m:.2f}%", f"±{s:.2f}%", f"{diff:+.2f}pp"])
    rows.append(["Full (seed=42)", "5,126", f"{FULL_F1:.2f}%", "—", "—"])

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2980b9")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        color = "#f0f8ff" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(color)

    ax.set_title("Few-shot NER Results Summary (PhoBERT-base, PhoNER_COVID19)",
                 fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "8_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 8_summary_table.png")


def main():
    if not IN_PATH.exists():
        print(f"Chua co ket qua: {IN_PATH}")
        print("Hay chay run_multiseed.py truoc.")
        return

    data = load()
    print(f"Loaded results for modes: {list(data.keys())}")

    plot_learning_curve(data)
    plot_per_entity_errorbars(data)
    plot_variance_heatmap(data)
    plot_summary_table(data)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
