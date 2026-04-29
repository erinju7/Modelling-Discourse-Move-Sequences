import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None  # type: ignore

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE          = "/Users/macbook/Desktop/AIED cw2"
META_FILE     = f"{BASE}/clustering_meta.csv"
LABELLED_CSV  = f"{BASE}/clustering_labelled.csv"
SEQUENCES_CSV = f"{BASE}/essay_sequences.csv"
HEATMAP_FILE  = f"{BASE}/move_distribution_v2.png"

# ── Step 0: Check which cluster IDs 0-43 are missing from the mapping ─────────
# Note: Python dicts keep the LAST value for duplicate keys.
# cluster 36 appears twice below — 'Physical_Description' wins (last entry).
CLUSTER_TO_MOVE = {
    # Social_Opening
    1:  "Social_Opening",
    15: "Social_Opening",
    # Self_Introduction
    2:  "Self_Introduction",
    3:  "Self_Introduction",
    24: "Self_Introduction",
    25: "Self_Introduction",
    # Physical_Description
    4:  "Physical_Description",
    14: "Physical_Description",
    17: "Physical_Description",
    36: "Physical_Description",   # duplicate key; 'Self_Introduction' entry is overridden
    # Daily_Routine
    9:  "Daily_Routine",
    12: "Daily_Routine",
    13: "Daily_Routine",
    19: "Daily_Routine",
    20: "Daily_Routine",
    21: "Daily_Routine",
    # Narrative_Experience
    6:  "Narrative_Experience",
    7:  "Narrative_Experience",
    28: "Narrative_Experience",
    37: "Narrative_Experience",
    40: "Narrative_Experience",
    # Opinion_Evaluation
    18: "Opinion_Evaluation",
    22: "Opinion_Evaluation",
    23: "Opinion_Evaluation",
    29: "Opinion_Evaluation",
    33: "Opinion_Evaluation",
    34: "Opinion_Evaluation",
    39: "Opinion_Evaluation",
    # Information_Reporting
    10: "Information_Reporting",
    11: "Information_Reporting",
    26: "Information_Reporting",
    27: "Information_Reporting",
    30: "Information_Reporting",
    32: "Information_Reporting",
    38: "Information_Reporting",
    41: "Information_Reporting",
    42: "Information_Reporting",
    43: "Information_Reporting",
    # Social_Closing
    0:  "Social_Closing",
    5:  "Social_Closing",
    8:  "Social_Closing",
    16: "Social_Closing",
    31: "Social_Closing",
    35: "Social_Closing",
    # Noise
    -1: "Other",
}

mapped_ids   = set(k for k in CLUSTER_TO_MOVE if k != -1)
all_ids      = set(range(44))
missing_ids  = sorted(all_ids - mapped_ids)

print("── Cluster coverage check ───────────────────────────────────────────────")
print(f"  Clusters 0-43 mapped : {sorted(mapped_ids)}")
if missing_ids:
    print(f"  NOT in dictionary    : {missing_ids}  → will be labelled 'Other'")
else:
    print("  All 44 clusters (0-43) are mapped. Nothing falls to 'Other' except noise (-1).")

# ── Step 1: Load data and apply mapping ───────────────────────────────────────
print("\nLoading clustering_meta.csv ...")
df = pd.read_csv(META_FILE)

cluster_col = "cluster_id_v2" if "cluster_id_v2" in df.columns else "cluster_id"
print(f"  Using cluster column: '{cluster_col}'  ({df[cluster_col].nunique()} unique values)")

df["discourse_move"] = df[cluster_col].map(CLUSTER_TO_MOVE).fillna("Other")

# ── Step 2: Save labelled dataset ─────────────────────────────────────────────
df.to_csv(LABELLED_CSV, index=False)
print(f"  Saved {LABELLED_CSV}")

# ── Step 3: Proportion of sentences per discourse move ────────────────────────
print("\n── Sentence proportions per discourse move ──────────────────────────────")
total = len(df)
move_counts = df["discourse_move"].value_counts()
for move, count in move_counts.items():
    print(f"  {move:<25} {count:>5}  ({100*count/total:.1f}%)")

# ── Step 4: Heatmap ───────────────────────────────────────────────────────────
MOVE_ORDER = [
    "Social_Opening",
    "Social_Closing",
    "Self_Introduction",
    "Physical_Description",
    "Daily_Routine",
    "Narrative_Experience",
    "Opinion_Evaluation",
    "Information_Reporting",
    "Other",
]
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

cross = pd.crosstab(df["discourse_move"], df["cefr_level"])
# Normalise within each CEFR level (column) so values are proportions
cross_norm = cross.div(cross.sum(axis=0), axis=1)
cross_norm = cross_norm.reindex(index=MOVE_ORDER, columns=CEFR_ORDER, fill_value=0)

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(
    cross_norm,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Proportion within CEFR level"},
)
ax.set_title(
    "Discourse Move Distribution by CEFR Level\n(proportion of sentences at each level)",
    fontsize=13,
)
ax.set_xlabel("CEFR Level")
ax.set_ylabel("Discourse Move")
plt.tight_layout()
plt.savefig(HEATMAP_FILE, dpi=150)
plt.close()
print(f"\n  Saved heatmap → {HEATMAP_FILE}")

# ── Step 5: Essay-level move sequences ────────────────────────────────────────
print("\n── Extracting discourse move sequences per essay ────────────────────────")
df_sorted = df.sort_values(["writing_id", "sentence_index"])

sequences = (
    df_sorted
    .groupby(["writing_id", "cefr_level"], sort=False)
    .apply(lambda g: list(g["discourse_move"]))
    .reset_index()
    .rename(columns={0: "move_sequence"})
)
sequences["move_sequence"] = sequences["move_sequence"].apply(str)
sequences.to_csv(SEQUENCES_CSV, index=False)

print(f"  Essays processed: {len(sequences)}")
print(f"  Saved {SEQUENCES_CSV}")

print("\nSample sequences:")
for _, row in sequences.head(8).iterrows():
    print(f"  [{row['cefr_level']}] writing {row['writing_id']}: {row['move_sequence']}")
