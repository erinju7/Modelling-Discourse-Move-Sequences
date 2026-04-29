import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/macbook/Desktop/AIED cw2"

emb_2d = np.load(f"{BASE}/umap_2d.npy")
meta   = pd.read_csv(f"{BASE}/clustering_labelled.csv")

# ── Colour palettes ───────────────────────────────────────────────────────────
MOVE_COLOURS = {
    "Social_Opening":       "#e6194b",
    "Self_Introduction":    "#3cb44b",
    "Physical_Description": "#4363d8",
    "Daily_Routine":        "#f58231",
    "Narrative_Experience": "#911eb4",
    "Opinion_Evaluation":   "#42d4f4",
    "Information_Reporting":"#f032e6",
    "Social_Closing":       "#bfef45",
    "Other":                "#aaaaaa",
}

CEFR_COLOURS = {
    "A1": "#1f77b4",
    "A2": "#ff7f0e",
    "B1": "#2ca02c",
    "B2": "#d62728",
    "C1": "#9467bd",
    "C2": "#8c564b",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

# ── Plot 1: coloured by discourse move ───────────────────────────────────────
ax = axes[0]
ax.set_facecolor("white")

for move, colour in MOVE_COLOURS.items():
    mask = meta["discourse_move"] == move
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=colour, s=8, alpha=0.6, label=move.replace("_", " "), linewidths=0)

ax.set_title("UMAP coloured by Discourse Move", fontsize=12, pad=10)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper right", fontsize=7, markerscale=1.5,
          framealpha=0.9, edgecolor="lightgrey")

# ── Plot 2: coloured by CEFR level ────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor("white")

for level, colour in CEFR_COLOURS.items():
    mask = meta["cefr_level"] == level
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=colour, s=8, alpha=0.6, label=level, linewidths=0)

ax.set_title("UMAP coloured by CEFR Level", fontsize=12, pad=10)
ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="upper right", fontsize=9, markerscale=1.5,
          framealpha=0.9, edgecolor="lightgrey")

plt.tight_layout()
plt.savefig(f"{BASE}/umap_visualization.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved umap_visualization.png")
