"""
04_rst_analysis.py
Compute RST relation frequency distributions per CEFR level,
run Welch t-tests (each level vs C2), and draw a heatmap.

Input:  rst_results.csv  (from 03_rst_parse.py)
Output: outputs/rst_relation_heatmap.png
        rst_ttest_results.csv
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from collections import Counter
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
INPUT_CSV  = BASE / "rst_results.csv"
OUT_HEAT   = BASE / "outputs" / "rst_relation_heatmap.png"
OUT_STATS  = BASE / "rst_ttest_results.csv"
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
ALPHA      = 0.05

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["relations"] = df["relations"].apply(json.loads)
df = df[df["n_edus"] > 0].copy()        # drop parse failures
print(f"Loaded {len(df)} essays")

# ── Build frequency matrix (relation rate = count / n_edus per essay) ─────────
all_relations = sorted(
    {r for rels in df["relations"] for r in rels}
)
print(f"Unique RST relations: {len(all_relations)}")
print(f"  {all_relations}\n")

freq_records = []
for _, row in df.iterrows():
    counts = Counter(row["relations"])
    n = row["n_edus"]
    record = {rel: counts.get(rel, 0) / n for rel in all_relations}
    record["cefr_level"] = row["cefr_level"]
    record["writing_id"] = row["writing_id"]
    freq_records.append(record)

freq_df = pd.DataFrame(freq_records)

# ── Welch t-tests: each level vs C2, per relation ─────────────────────────────
c2_data = freq_df[freq_df["cefr_level"] == "C2"]
ttest_rows = []

for level in CEFR_ORDER[:-1]:            # A1 … C1
    level_data = freq_df[freq_df["cefr_level"] == level]
    for rel in all_relations:
        t, p = stats.ttest_ind(
            level_data[rel].values,
            c2_data[rel].values,
            equal_var=False              # Welch's t-test
        )
        ttest_rows.append({
            "level":       level,
            "relation":    rel,
            "mean_level":  level_data[rel].mean(),
            "mean_c2":     c2_data[rel].mean(),
            "t_stat":      round(t, 4),
            "p_value":     round(p, 4),
            "significant": p < ALPHA,
        })

ttest_df = pd.DataFrame(ttest_rows)
ttest_df.to_csv(OUT_STATS, index=False)
print(f"Saved t-test results to {OUT_STATS}")

n_sig = ttest_df["significant"].sum()
print(f"Significant comparisons (p < {ALPHA}): {n_sig} / {len(ttest_df)}\n")

# ── Mean frequency heatmap ─────────────────────────────────────────────────────
mean_by_level = (
    freq_df.groupby("cefr_level")[all_relations]
           .mean()
           .reindex(CEFR_ORDER)          # rows = CEFR levels
)

# Build significance mask: True where level vs C2 is significant
sig_matrix = pd.DataFrame(False,
                           index=all_relations,
                           columns=CEFR_ORDER)
for _, row in ttest_df.iterrows():
    if row["significant"]:
        sig_matrix.loc[row["relation"], row["level"]] = True
# C2 column is always the reference (no asterisk)
sig_matrix["C2"] = False

fig, ax = plt.subplots(figsize=(max(10, len(all_relations) * 0.8), 6))

sns.heatmap(
    mean_by_level.T,           # relations on Y-axis, CEFR on X-axis
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    linewidths=0.4,
    linecolor="white",
    ax=ax,
    cbar_kws={"label": "Mean relation rate (per EDU)"},
)

# Overlay asterisks for significant cells
for yi, rel in enumerate(all_relations):
    for xi, level in enumerate(CEFR_ORDER):
        if sig_matrix.loc[rel, level]:
            ax.text(xi + 0.5, yi + 0.15, "*",
                    ha="center", va="bottom",
                    fontsize=11, color="black", fontweight="bold")

ax.set_title(
    "RST Relation Frequency by CEFR Level\n"
    "* = significantly different from C2 (Welch t-test, p < 0.05)",
    fontsize=12, pad=12
)
ax.set_xlabel("CEFR Level", fontsize=11)
ax.set_ylabel("RST Relation", fontsize=11)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
OUT_HEAT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_HEAT, dpi=150, bbox_inches="tight")
print(f"Saved heatmap to {OUT_HEAT}")
plt.show()

# ── Console summary ────────────────────────────────────────────────────────────
print("\nTop significant relations per level vs C2:")
for level in CEFR_ORDER[:-1]:
    sig = ttest_df[(ttest_df["level"] == level) & ttest_df["significant"]]
    if sig.empty:
        print(f"  {level}: none")
    else:
        rels = ", ".join(sig.sort_values("p_value")["relation"].tolist())
        print(f"  {level}: {rels}")
