"""Generate Figure 3.1 - Overall Few-shot NER pipeline diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis("off")

# ── Color palette ──────────────────────────────────────────────────────────────
C_BOX  = "#E8F4FD"   # light blue fill
C_EDGE = "#2980B9"   # blue border
C_HEAD = "#2980B9"   # header fill
C_HTXT = "white"     # header text
C_TXT  = "#1A252F"   # body text
C_ARR  = "#2C3E50"   # arrow

def box(ax, x, y, w, h, title, lines, color=C_BOX, edgecolor=C_EDGE,
        title_color=C_HEAD, title_text_color=C_HTXT):
    """Draw a labelled box with header + body lines."""
    title_h = 0.55
    # Header
    header = FancyBboxPatch((x, y + h - title_h), w, title_h,
                             boxstyle="round,pad=0.02",
                             facecolor=title_color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(header)
    ax.text(x + w / 2, y + h - title_h / 2, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=title_text_color)
    # Body
    body = FancyBboxPatch((x, y), w, h - title_h,
                           boxstyle="round,pad=0.02",
                           facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(body)
    step = (h - title_h) / (len(lines) + 1)
    for i, line in enumerate(lines):
        ax.text(x + 0.25, y + (h - title_h) - step * (i + 1), line,
                ha="left", va="center", fontsize=9.5, color=C_TXT,
                fontfamily="Segoe UI")

def arrow(ax, x, y1, y2):
    """Vertical downward arrow."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARR,
                                lw=2, mutation_scale=18))
    # Label on arrow
    mid = (y1 + y2) / 2
    return mid

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(7, 17.5, "Hình 3.1 – Mô hình tổng quát hệ thống Few-shot NER tiếng Việt COVID-19",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_TXT)

# ── Stage 1 ────────────────────────────────────────────────────────────────────
box(ax, 1, 14.5, 12, 2.5,
    "GIAI ĐOẠN 1: XÂY DỰNG DỮ LIỆU",
    ["PhoNER_COVID19 (gốc: 5,027 câu)  +  bổ sung 99 câu  →  5,126 câu train",
     "Entity hiếm: JOB +71 (+34.6%),  TRANSPORTATION +18 (+8.0%)",
     "Chuẩn hoá schema  |  Deduplication (loại 3 câu trùng)  |  Kiểm tra nhất quán"])

arrow(ax, 7, 14.5, 13.7)
ax.text(7.3, 14.1, "D_train = 5,126 câu", fontsize=9, color=C_ARR, style="italic")

# ── Stage 2 ────────────────────────────────────────────────────────────────────
box(ax, 1, 10.7, 12, 2.7,
    "GIAI ĐOẠN 2: TẠO FEW-SHOT SPLITS  (seed = 42)",
    ["SUPPORT SET S:  K câu × 10 entity types,  K ∈ {1, 5, 10, 20}",
     "  K=1→~10 câu  |  K=5→~50 câu  |  K=10→~100 câu  |  K=20→~200 câu",
     "Dev set (2,000 câu) → chọn best checkpoint (few-shot: cuối epoch)",
     "Test set (3,000 câu) → cố định, chỉ dùng để đánh giá cuối"])

arrow(ax, 7, 10.7, 9.9)
ax.text(7.3, 10.3, "support.json", fontsize=9, color=C_ARR, style="italic")

# ── Stage 3 ────────────────────────────────────────────────────────────────────
box(ax, 1, 7.5, 12, 2.1,
    "GIAI ĐOẠN 3: TOKENIZE & ALIGN NHÃN BIO",
    ['word:    ["phi_công",   "người",  "Anh" ]    BIO: ["B-JOB",  "O",  "O"]',
     'subword: [<s>, "phi", "_công", "người", "Anh", </s>]',
     'label:   [-100,  B-JOB,  I-JOB,    O,     O,   -100]   (MAX_LEN=128)'])

arrow(ax, 7, 7.5, 6.7)
ax.text(7.3, 7.1, "input_ids, attention_mask, labels", fontsize=9, color=C_ARR, style="italic")

# ── Stage 4 ────────────────────────────────────────────────────────────────────
box(ax, 1, 4.0, 12, 2.4,
    "GIAI ĐOẠN 4: FINE-TUNE PhoBERT-BASE",
    ["PhoBERT Encoder (12 Transformer layers, hidden=768, 135M params)",
     "Linear(768 → 21 labels)  ← Token Classification Head",
     "Cross-entropy Loss → AdamW (lr=5e-5)  |  3 seeds: {1, 42, 100}"])

arrow(ax, 7, 4.0, 3.2)
ax.text(7.3, 3.6, "trained model (per seed × per K)", fontsize=9, color=C_ARR, style="italic")

# ── Stage 5 ────────────────────────────────────────────────────────────────────
box(ax, 1, 1.0, 12, 2.0,
    "GIAI ĐOẠN 5: ĐÁNH GIÁ & PHÂN TÍCH LỖI",
    ["Inference trên Test set (3,000 câu, cố định)  →  Micro F1 (span-level exact match)",
     "Báo cáo: mean ± std qua 3 seeds  |  Phân tích 877 lỗi → 5 pattern"])

# ── Result callout ─────────────────────────────────────────────────────────────
result_box = FancyBboxPatch((1, 0.1), 12, 0.7,
                             boxstyle="round,pad=0.05",
                             facecolor="#D5F5E3", edgecolor="#27AE60", linewidth=2)
ax.add_patch(result_box)
ax.text(7, 0.45,
        "Kết quả: Full→94.23%  |  20-shot→88.17%±0.35%  |  10-shot→83.35%  |  5-shot→68.02%",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#1E8449")

plt.tight_layout(pad=0.5)
out_path = r"d:\data_for_NLP_COVID19\PhoNER_COVID19\output\figures\hinh_3_1_pipeline.jpg"
plt.savefig(out_path, dpi=200, bbox_inches="tight", format="jpeg")
print(f"Saved: {out_path}")
