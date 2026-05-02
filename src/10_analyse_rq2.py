"""
10_analyse_rq2.py
Statistical analysis of RQ2 evaluation results.

- Wilcoxon signed-rank test (pairwise) with Bonferroni correction
- Rank-biserial correlation (effect size)
- Results table (mean ± SD per condition × dimension)

Inputs:  scores_rq2_claude.csv
Outputs: results_table_rq2.csv, wilcoxon_rq2.csv, results_rq2.png
"""

import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from itertools import combinations
from scipy import stats

BASE        = Path("/Users/macbook/Desktop/AIED cw2")
CLAUDE_CSV  = BASE / os.environ.get("SCORES_CSV", "scores_n30_claude.csv")
OUT_TABLE   = BASE / os.environ.get("OUT_TABLE", "results_table_n30.csv")
OUT_PLOT    = BASE / os.environ.get("OUT_PLOT", "outputs/results_n30.png")

CONDITIONS = ["RAG", "Baseline", "One-shot"]
COND_LABELS = {"RAG": "RAG", "Baseline": "Zero-shot Baseline", "One-shot": "One-shot"}
DIMENSIONS  = ["specificity", "helpfulness", "validity"]

# Bonferroni correction: 3 pairs × 3 dimensions = 9 comparisons
N_COMPARISONS   = len(list(combinations(CONDITIONS, 2))) * len(DIMENSIONS)
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS   # ≈ 0.0056


def bonferroni_correct(p_value: float) -> float:
    """Cap Bonferroni-adjusted p-values at 1.0 for reporting."""
    return min(p_value * N_COMPARISONS, 1.0)


def rank_biserial(x, y):
    """Rank-biserial correlation as effect size for Wilcoxon test."""
    n = len(x)
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0.0
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return (pos - neg) / len(diffs)


def analyse(df: pd.DataFrame, judge_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Wilcoxon tests and build results table for one judge."""

    # Extract score arrays: {condition: {dimension: array}}
    scores = {
        c: {d: df[f"{c}_{d}"].dropna().values for d in DIMENSIONS}
        for c in CONDITIONS
    }

    # ── Results table ─────────────────────────────────────────────────────────
    table_rows = []
    stat_rows  = []

    for dim in DIMENSIONS:
        row = {"Judge": judge_name, "Dimension": dim.title()}
        for c in CONDITIONS:
            vals = scores[c][dim]
            row[COND_LABELS[c]]            = f"{vals.mean():.2f} ± {vals.std():.2f}"
            row[f"_mean_{c}"]              = vals.mean()   # for sorting / plotting
        table_rows.append(row)

        # Pairwise Wilcoxon
        for c_a, c_b in combinations(CONDITIONS, 2):
            a_vals = scores[c_a][dim]
            b_vals = scores[c_b][dim]
            # Wilcoxon requires same length; use min
            n = min(len(a_vals), len(b_vals))
            stat, p = stats.wilcoxon(a_vals[:n], b_vals[:n],
                                     zero_method="wilcox", alternative="two-sided")
            rb = rank_biserial(a_vals[:n].tolist(), b_vals[:n].tolist())
            p_corr = bonferroni_correct(p)
            stat_rows.append({
                "Judge":       judge_name,
                "Dimension":   dim.title(),
                "Comparison":  f"{c_a} vs {c_b}",
                "W":           round(stat, 1),
                "p_value":     round(p, 4),
                "p_corrected": round(p_corr, 4),
                "significant": p < BONFERRONI_ALPHA,
                "r_rb":        round(rb, 3),
                "mean_A":      round(a_vals.mean(), 3),
                "mean_B":      round(b_vals.mean(), 3),
            })

    return pd.DataFrame(table_rows), pd.DataFrame(stat_rows)


# ── Load and analyse ──────────────────────────────────────────────────────────

claude_df = pd.read_csv(CLAUDE_CSV)

results_table, results_stats = analyse(claude_df, "Claude Sonnet")

print(f"\n{'='*65}  Claude Sonnet")
print(results_table[[c for c in results_table.columns if not c.startswith("_")]].to_string(index=False))
print(f"\nWilcoxon (Bonferroni α={BONFERRONI_ALPHA:.4f}):")
print(results_stats[["Dimension","Comparison","p_value","p_corrected","significant","r_rb"]].to_string(index=False))

results_table.to_csv(OUT_TABLE, index=False)
results_stats.to_csv(OUT_TABLE.with_name("wilcoxon_rq2.csv"), index=False)
print(f"\nSaved {OUT_TABLE.name}, wilcoxon_rq2.csv")

# ── Plot: grouped bar chart ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(DIMENSIONS), figsize=(12, 5), sharey=True)
cmap = {"RAG": "#4878CF", "Baseline": "#E24A33", "One-shot": "#6ACC65"}
x = np.arange(len(CONDITIONS))

for ax, dim in zip(axes, DIMENSIONS):
    width = 0.6
    means = [claude_df[f"{c}_{dim}"].dropna().mean() for c in CONDITIONS]
    sems  = [claude_df[f"{c}_{dim}"].dropna().sem()  for c in CONDITIONS]
    bars  = ax.bar(x, means, width, yerr=sems, capsize=4, alpha=0.85,
                   color=[cmap[c] for c in CONDITIONS])

    ax.set_title(dim.title(), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], fontsize=9, rotation=15, ha="right")
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean score (1–5)" if ax == axes[0] else "")
    ax.axhline(y=3, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[c]) for c in CONDITIONS]
axes[-1].legend(handles, [COND_LABELS[c] for c in CONDITIONS],
                title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.suptitle("RQ2: Feedback Quality by Condition (Claude Sonnet judge)", fontsize=13)
plt.tight_layout()
OUT_PLOT.parent.mkdir(exist_ok=True)
plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Saved plot to {OUT_PLOT}")
plt.show()
