import ast
from pathlib import Path

import pandas as pd

BASE = Path("/Users/macbook/Desktop/AIED cw2")
SEQUENCES_CSV = BASE / "essay_sequences.csv"
OUT_CSV = BASE / "sequence_features_summary.csv"

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def clean_sequence(seq):
    return [move for move in seq if move != "Other"]


def repetition_rate(seq):
    if len(seq) < 2:
        return 0.0
    repeats = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a == b)
    return repeats / (len(seq) - 1)


df = pd.read_csv(SEQUENCES_CSV)
df["move_sequence"] = df["move_sequence"].apply(ast.literal_eval)

rows = []
for _, row in df.iterrows():
    seq = clean_sequence(row["move_sequence"])
    rows.append(
        {
            "writing_id": row["writing_id"],
            "cefr_level": row["cefr_level"],
            "sequence_length": len(seq),
            "move_diversity": len(set(seq)) if seq else 0,
            "repetition_rate": repetition_rate(seq),
        }
    )

feat = pd.DataFrame(rows)
summary = (
    feat.groupby("cefr_level")
    .agg(
        n_essays=("writing_id", "count"),
        sequence_length_mean=("sequence_length", "mean"),
        sequence_length_sd=("sequence_length", "std"),
        move_diversity_mean=("move_diversity", "mean"),
        move_diversity_sd=("move_diversity", "std"),
        repetition_rate_mean=("repetition_rate", "mean"),
        repetition_rate_sd=("repetition_rate", "std"),
    )
    .reindex(CEFR_ORDER)
    .reset_index()
)

summary.to_csv(OUT_CSV, index=False)

print("Sequence-level structural features by CEFR level")
print(summary.round(3).to_string(index=False))
print(f"\nSaved {OUT_CSV}")
