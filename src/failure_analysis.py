"""
failure_analysis.py
Targeted failure analysis: why does RAG score lowest on validity?

For each essay, compares RAG vs One-shot validity scores and
flags cases where RAG is substantially worse.
Outputs: failure_analysis.csv + printed summary
"""

import ast
import pandas as pd
import numpy as np
from pathlib import Path

BASE        = Path("/Users/macbook/Desktop/AIED cw2")
SCORES_CSV  = BASE / "scores_n30_claude.csv"
FEEDBACK_CSV = BASE / "feedback_n30.csv"
OUT_CSV     = BASE / "failure_analysis.csv"

scores_df   = pd.read_csv(SCORES_CSV)
feedback_df = pd.read_csv(FEEDBACK_CSV)

df = scores_df.merge(feedback_df, on="writing_id")

# ── Per-essay validity gap ─────────────────────────────────────────────────────

df["rag_vs_oneshot_validity"] = df["RAG_validity"] - df["One-shot_validity"]
df["rag_vs_baseline_validity"] = df["RAG_validity"] - df["Baseline_validity"]

# Cases where RAG validity is lower than One-shot by ≥1 point
failures = df[df["rag_vs_oneshot_validity"] <= -1].copy()
neutral  = df[(df["rag_vs_oneshot_validity"] > -1) & (df["rag_vs_oneshot_validity"] < 1)].copy()
wins     = df[df["rag_vs_oneshot_validity"] >= 1].copy()

print(f"Total essays: {len(df)}")
print(f"RAG validity < One-shot by ≥1 (failures): {len(failures)}")
print(f"RAG validity ≈ One-shot (neutral):         {len(neutral)}")
print(f"RAG validity > One-shot by ≥1 (wins):      {len(wins)}")

# ── Inspect failures ───────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("FAILURE CASES (RAG validity ≤ One-shot − 1)")
print(f"{'='*70}")

col_task = "task_type" if "task_type" in df.columns else "genre"

for _, row in failures.sort_values("rag_vs_oneshot_validity").iterrows():
    print(f"\n--- writing_id={row['writing_id']}  task_type={row[col_task]}  edit_dist={row['edit_distance']} ---")
    print(f"Scores  RAG validity={row['RAG_validity']:.2f}  "
          f"Baseline={row['Baseline_validity']:.2f}  "
          f"One-shot={row['One-shot_validity']:.2f}")
    print(f"\nESSAY:\n{row['essay_text']}")
    print(f"\nMOVE SEQUENCE: {row['move_sequence']}")
    print(f"\nRAG FEEDBACK:\n{row['feedback_rag']}")
    print(f"\nONE-SHOT FEEDBACK:\n{row['feedback_oneshot']}")
    print()

# ── Word count comparison ──────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("WORD COUNT COMPARISON (failures vs rest)")
print(f"{'='*70}")

for group, label in [(failures, "Failures"), (neutral, "Neutral"), (wins, "Wins")]:
    if len(group) == 0:
        continue
    rag_wc = group["feedback_rag"].apply(lambda x: len(str(x).split())).mean()
    one_wc = group["feedback_oneshot"].apply(lambda x: len(str(x).split())).mean()
    print(f"{label:10s} ({len(group):2d} essays)  RAG wc={rag_wc:.0f}  One-shot wc={one_wc:.0f}  "
          f"mean RAG validity={group['RAG_validity'].mean():.2f}  "
          f"mean One-shot validity={group['One-shot_validity'].mean():.2f}")

# ── Move sequence features ─────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("MOVE SEQUENCE FEATURES")
print(f"{'='*70}")

def seq_features(seq_str):
    moves = [m.strip() for m in seq_str.split("->")]
    n = len(moves)
    unique = len(set(moves))
    repetitions = n - unique
    has_other = "Other" in moves
    return pd.Series({"n_moves": n, "n_unique": unique, "repetitions": repetitions, "has_other": has_other})

feats = df["move_sequence"].apply(seq_features)
df = pd.concat([df, feats], axis=1)

for group, label in [(failures, "Failures"), (neutral, "Neutral"), (wins, "Wins")]:
    if len(group) == 0:
        continue
    g = df.loc[group.index]
    print(f"{label:10s}  n_moves={g['n_moves'].mean():.1f}  "
          f"n_unique={g['n_unique'].mean():.1f}  "
          f"repetitions={g['repetitions'].mean():.1f}  "
          f"has_other={g['has_other'].mean():.0%}  "
          f"edit_dist={g['edit_distance'].mean():.1f}")

# ── Save ───────────────────────────────────────────────────────────────────────

out_cols = ["writing_id", col_task, "move_sequence", "edit_distance",
            "RAG_validity", "Baseline_validity", "One-shot_validity",
            "rag_vs_oneshot_validity", "rag_vs_baseline_validity",
            "feedback_rag", "feedback_oneshot"]
df[out_cols].sort_values("rag_vs_oneshot_validity").to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV.name}")
