import pandas as pd
from pathlib import Path

BASE      = Path("/Users/macbook/Desktop/AIED cw2")
SCORES    = BASE / "feedback_quality_results_final.csv"
RAG_CSV   = BASE / "rag_evaluation_results.csv"
BL_CSV    = BASE / "baseline_feedback.csv"
OS_CSV    = BASE / "oneshot_feedback.csv"
OUT_CSV   = BASE / "failure_analysis.csv"

THRESHOLD = 2.0

# ── Load ──────────────────────────────────────────────────────────────────────

scores_df = pd.read_csv(SCORES)
rag_df    = pd.read_csv(RAG_CSV).set_index("writing_id")
bl_df     = pd.read_csv(BL_CSV).set_index("writing_id")
os_df     = pd.read_csv(OS_CSV).set_index("writing_id")

CONDITIONS = {
    "C1 (Zero-shot)": {
        "dims": ["baseline_specificity", "baseline_helpfulness", "baseline_validity"],
        "feedback_src": bl_df,
        "feedback_col": "baseline_feedback",
        "seq_src": None,
    },
    "C2 (One-shot)": {
        "dims": ["oneshot_specificity", "oneshot_helpfulness", "oneshot_validity"],
        "feedback_src": os_df,
        "feedback_col": "oneshot_feedback",
        "seq_src": None,
    },
    "C3 (RAG)": {
        "dims": ["rag_specificity", "rag_helpfulness", "rag_validity"],
        "feedback_src": rag_df,
        "feedback_col": "feedback",
        "seq_src": rag_df,
    },
}

DIM_LABELS = ["Specificity", "Helpfulness", "Validity"]

# ── Step 1: Identify failures ─────────────────────────────────────────────────

failure_rows = []

for cname, cfg in CONDITIONS.items():
    dims = cfg["dims"]
    for _, row in scores_df.iterrows():
        wid    = row["writing_id"]
        scores = [row[d] for d in dims]
        if any(s <= THRESHOLD for s in scores):
            lowest_idx  = scores.index(min(scores))
            lowest_dim  = DIM_LABELS[lowest_idx]
            feedback_df = cfg["feedback_src"]
            feedback    = feedback_df.loc[wid, cfg["feedback_col"]] if wid in feedback_df.index else ""
            seq         = rag_df.loc[wid, "a1_sequence"] if wid in rag_df.index else "N/A"
            failure_rows.append({
                "condition":       cname,
                "writing_id":      wid,
                "specificity":     scores[0],
                "helpfulness":     scores[1],
                "validity":        scores[2],
                "lowest_dim":      lowest_dim,
                "lowest_score":    min(scores),
                "a1_sequence":     seq,
                "feedback_excerpt": str(feedback)[:200],
            })

failure_df = pd.DataFrame(failure_rows)

# ── Step 2: Failure summary table ─────────────────────────────────────────────

total = len(scores_df)

print("=" * 60)
print(f"{'Condition':<18} | {'Essays <=2':>10} | {'% failure':>10}")
print("-" * 60)

for cname in CONDITIONS:
    n = failure_df[failure_df["condition"] == cname]["writing_id"].nunique()
    pct = n / total * 100
    print(f"{cname:<18} | {f'{n}/{total}':>10} | {pct:>9.0f}%")

print("=" * 60)

# ── Step 3: Per-failure details ───────────────────────────────────────────────

print()
for _, fr in failure_df.iterrows():
    print(f"===== writing_id={fr['writing_id']} | Condition={fr['condition']} =====")
    print(f"Sequence   : {fr['a1_sequence']}")
    print(f"Scores     : Specificity={fr['specificity']:.2f}, "
          f"Helpfulness={fr['helpfulness']:.2f}, Validity={fr['validity']:.2f}")
    print(f"Feedback   : {fr['feedback_excerpt']}")
    print(f"Lowest dim : {fr['lowest_dim']} ({fr['lowest_score']:.2f})")
    print("=" * 54)

# ── Save ──────────────────────────────────────────────────────────────────────

failure_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}  ({len(failure_df)} failure cases)")
