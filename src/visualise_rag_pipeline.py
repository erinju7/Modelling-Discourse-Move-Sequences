import re
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE     = "/Users/macbook/Desktop/AIED cw2"
TXT_FILE = f"{BASE}/rag_demonstration_v3.txt"
OUT_FILE = f"{BASE}/rag_demonstration_figure.png"

# ── Parse rag_demonstration_v3.txt ───────────────────────────────────────────
with open(TXT_FILE, encoding="utf-8") as f:
    raw = f.read()

# A1 essay text
a1_text = re.search(r"Essay text:\n(.+)", raw).group(1).strip()

# A1 move sequence (list inside square brackets)
a1_seq_raw = re.search(r"Move sequence: \[(.+?)\]", raw).group(1)
a1_moves   = [m.strip().strip("'") for m in a1_seq_raw.split(",")]

# Rank 1 C2 essay
rank1_block  = re.search(r"── Rank 1.*?edit_distance=(\d+)\).*?Move sequence: (.+?)\nEssay text:\n(.+?)(?=\n── Rank|\n=====)", raw, re.S)
edit_dist    = rank1_block.group(1)
c2_seq_str   = rank1_block.group(2).strip()
c2_moves     = c2_seq_str.split("|")
c2_text      = rank1_block.group(3).strip()

# Generated feedback (everything after "Edit distance" line up to end)
feedback_raw = re.search(r"Edit distance\s+:.*?\n\n(.+)", raw, re.S).group(1).strip()

# ── Truncate helpers ──────────────────────────────────────────────────────────
def trunc(text, n):
    return text[:n] + "..." if len(text) > n else text

def wrap(text, width=38):
    return textwrap.fill(text, width=width)

def fmt_moves(moves, per_line=2):
    """Format move list as wrapped lines of per_line items each."""
    lines = []
    for i in range(0, len(moves), per_line):
        lines.append(" → ".join(m.replace("_", " ") for m in moves[i:i+per_line]))
    return "\n".join(lines)

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6), facecolor="white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

BOX_STYLE  = "round,pad=0.15"
TITLE_Y    = 5.55
CONTENT_Y  = 5.15

BOXES = [
    dict(x=0.3,  w=4.5,  colour="#dbeafe", title="Input: A1 Essay",
         essay=trunc(a1_text, 120), moves=a1_moves),
    dict(x=5.75, w=4.5,  colour="#d1fae5", title=f"Retrieved: C2 Essay  (edit distance = {edit_dist})",
         essay=trunc(c2_text, 120), moves=c2_moves),
    dict(x=11.2, w=4.5,  colour="#ede9fe", title="Generated Feedback  (Qwen2.5-3B)",
         essay=None,                moves=None,
         feedback=trunc(feedback_raw, 220)),
]

for b in BOXES:
    # Draw box
    rect = FancyBboxPatch((b["x"], 0.3), b["w"], 5.4,
                          boxstyle=BOX_STYLE, linewidth=1.2,
                          edgecolor="#64748b", facecolor=b["colour"])
    ax.add_patch(rect)

    # Bold title
    ax.text(b["x"] + b["w"]/2, TITLE_Y, b["title"],
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="#1e293b",
            wrap=True)

    # Divider line below title
    ax.plot([b["x"]+0.15, b["x"]+b["w"]-0.15], [5.35, 5.35],
            color="#94a3b8", linewidth=0.8)

    if b.get("feedback"):
        # Feedback box: just wrapped text
        wrapped = wrap(b["feedback"], width=42)
        ax.text(b["x"] + 0.2, 5.15, wrapped,
                ha="left", va="top", fontsize=8, color="#1e293b",
                fontfamily="monospace", linespacing=1.4)
    else:
        # Essay text
        wrapped_essay = wrap(b["essay"], width=42)
        ax.text(b["x"] + 0.2, CONTENT_Y, wrapped_essay,
                ha="left", va="top", fontsize=8, color="#334155",
                linespacing=1.4)

        # Move sequence section
        ax.text(b["x"] + 0.2, 2.85, "Discourse move sequence:",
                ha="left", va="top", fontsize=7.5, fontstyle="italic",
                color="#475569")
        move_str = fmt_moves(b["moves"])
        ax.text(b["x"] + 0.2, 2.55, move_str,
                ha="left", va="top", fontsize=7.5, color="#1e3a5f",
                fontweight="bold", linespacing=1.5)

# ── Arrows between boxes ─────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="-|>", color="#475569", lw=1.5,
                connectionstyle="arc3,rad=0.0",
                mutation_scale=16)

# Arrow 1: Box1 → Box2
ax.annotate("", xy=(5.75, 3.0), xytext=(4.8, 3.0),
            arrowprops=arrow_kw)
ax.text(5.27, 3.22, "edit distance\nretrieval",
        ha="center", va="bottom", fontsize=7.5, color="#475569", style="italic")

# Arrow 2: Box2 → Box3
ax.annotate("", xy=(11.2, 3.0), xytext=(10.25, 3.0),
            arrowprops=arrow_kw)
ax.text(10.72, 3.22, "LLM\ngeneration",
        ha="center", va="bottom", fontsize=7.5, color="#475569", style="italic")

plt.tight_layout(pad=0.3)
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {OUT_FILE}")
