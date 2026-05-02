"""
sensitivity_analysis.py
Post-hoc sensitivity analysis: restrict to essays with edit_distance <= threshold.
Tests whether RAG validity disadvantage persists when retrieval quality is sufficient.

Outputs: sensitivity_results.csv
"""

import os
import pandas as pd, numpy as np
from scipy import stats
from itertools import combinations
from pathlib import Path

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
SCORES_CSV   = BASE / os.environ.get("SCORES_CSV", "scores_n30_claude.csv")
FEEDBACK_CSV = BASE / os.environ.get("FEEDBACK_CSV", "feedback_n30.csv")
OUT_CSV      = BASE / os.environ.get("OUT_CSV", "sensitivity_results.csv")

CONDITIONS = ["RAG", "Baseline", "One-shot"]
DIMENSIONS = ["specificity", "helpfulness", "validity"]
N_COMP     = 9
ALPHA      = 0.05 / N_COMP  # Bonferroni

scores_df   = pd.read_csv(SCORES_CSV)
feedback_df = pd.read_csv(FEEDBACK_CSV)
df = scores_df.merge(feedback_df[["writing_id", "edit_distance"]], on="writing_id")


def bonferroni_correct(p_value):
    return min(p_value * N_COMP, 1.0)


def rank_biserial(x, y):
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0.0
    return (sum(1 for d in diffs if d > 0) - sum(1 for d in diffs if d < 0)) / len(diffs)


def run_analysis(sub, label):
    rows = []
    print(f"\n{'='*65}  {label}  (n={len(sub)})")

    # Means table
    for dim in DIMENSIONS:
        means = {c: sub[f"{c}_{dim}"].mean() for c in CONDITIONS}
        sds   = {c: sub[f"{c}_{dim}"].std()  for c in CONDITIONS}
        print(f"  {dim:12s}: " +
              "  ".join(f"{c}={means[c]:.2f}±{sds[c]:.2f}" for c in CONDITIONS))

    print(f"\n  Wilcoxon (Bonferroni α={ALPHA:.4f}):")
    for dim in DIMENSIONS:
        for ca, cb in combinations(CONDITIONS, 2):
            a = sub[f"{ca}_{dim}"].dropna().values
            b = sub[f"{cb}_{dim}"].dropna().values
            n = min(len(a), len(b))
            stat, p = stats.wilcoxon(a[:n], b[:n],
                                     zero_method="wilcox", alternative="two-sided")
            rb  = rank_biserial(a[:n].tolist(), b[:n].tolist())
            p_corr = bonferroni_correct(p)
            sig = "**" if p < ALPHA else ("*" if p < 0.05 else "")
            print(f"  {dim:12s}  {ca} vs {cb:10s}: "
                  f"p={p:.4f}  p_corr={p_corr:.4f}  r_rb={rb:.3f}  {sig}")
            rows.append({
                "subset":      label,
                "n":           len(sub),
                "dimension":   dim,
                "comparison":  f"{ca} vs {cb}",
                "p_value":     round(p, 4),
                "p_corrected": round(p_corr, 4),
                "significant": p < ALPHA,
                "r_rb":        round(rb, 3),
                "mean_A":      round(a.mean(), 3),
                "mean_B":      round(b.mean(), 3),
            })
    return rows


# ── Run for full sample + each threshold ──────────────────────────────────────

all_rows = []
all_rows += run_analysis(df, "Full sample (n=30)")

for threshold in [3, 4]:
    sub   = df[df["edit_distance"] <= threshold]
    label = f"edit_dist ≤ {threshold} (n={len(sub)})"
    all_rows += run_analysis(sub, label)

# ── Save ──────────────────────────────────────────────────────────────────────

out_df = pd.DataFrame(all_rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV.name}")

# ── Summary: validity RAG vs One-shot across subsets ─────────────────────────

print(f"\n{'='*65}")
print("SUMMARY: Validity — RAG vs One-shot")
print(f"{'='*65}")
validity_rows = out_df[
    (out_df["dimension"] == "validity") &
    (out_df["comparison"] == "RAG vs One-shot")
][["subset", "n", "mean_A", "mean_B", "p_value", "p_corrected", "significant", "r_rb"]]
validity_rows.columns = ["Subset", "n", "RAG mean", "One-shot mean",
                         "p", "p_corr", "sig", "r_rb"]
print(validity_rows.to_string(index=False))
