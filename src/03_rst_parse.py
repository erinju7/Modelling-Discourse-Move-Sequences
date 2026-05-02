"""
03_rst_parse.py
Parse all sampled essays with the isanlp_rst parser (Tchewik et al., rstdt model)
and save per-essay RST relation counts.

Run on GPU:
    pip install git+https://github.com/iinemo/isanlp.git isanlp_rst
    python 03_rst_parse.py

Output: rst_results.csv
"""

import json
import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = Path(__file__).parent.parent / "clustering_sample.csv"
OUTPUT_CSV = Path(__file__).parent.parent / "rst_results.csv"
CUDA_DEVICE = 0      # set to -1 for CPU
BATCH_SIZE  = 8      # increase on GPU with more VRAM

# ── Load parser ───────────────────────────────────────────────────────────────
print("Loading isanlp_rst parser (downloading model on first run) ...")
from isanlp_rst.parser import Parser
parser = Parser(
    hf_model_name="tchewik/isanlp_rst_v3",
    hf_model_version="rstdt",
    cuda_device=CUDA_DEVICE,
)
print("Parser ready.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_relations(tree_node) -> list[str]:
    """
    Recursively walk an isanlp_rst DiscourseUnit tree and collect
    all non-span relation labels.
    """
    relations = []
    if tree_node is None:
        return relations
    rel = getattr(tree_node, "relation", None)
    if rel and rel.lower() != "span":
        relations.append(rel.lower())
    for child in (getattr(tree_node, "children", None) or []):
        relations.extend(extract_relations(child))
    return relations


def parse_essay(text: str) -> tuple[list[str], int]:
    """Run the parser on one essay; return (relations, n_edus)."""
    try:
        res = parser(text)
        trees = res.get("rst", [])
        if not trees:
            return [], 0
        # isanlp_rst returns a list of sentence-level trees
        all_rels = []
        n_edus = 0
        for tree in trees:
            all_rels.extend(extract_relations(tree))
            # count leaf EDUs (nodes with no children)
            n_edus += count_leaves(tree)
        return all_rels, n_edus
    except Exception as e:
        print(f"  Parse error: {e}", file=sys.stderr)
        return [], 0


def count_leaves(node) -> int:
    children = getattr(node, "children", None) or []
    if not children:
        return 1
    return sum(count_leaves(c) for c in children)


# ── Load & reconstruct essays ─────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} sentences from {INPUT_CSV.name}")

essays = (
    df.sort_values(["writing_id", "sentence_index"])
      .groupby(["writing_id", "cefr_level"], sort=False)["sentence"]
      .apply(" ".join)
      .reset_index()
)
essays.columns = ["writing_id", "cefr_level", "text"]
print(f"Reconstructed {len(essays)} essays across "
      f"{df['cefr_level'].nunique()} CEFR levels\n")

# ── Parse ─────────────────────────────────────────────────────────────────────
rows = []
failed = 0

for i, row in essays.iterrows():
    relations, n_edus = parse_essay(row["text"])
    if not relations and n_edus == 0:
        failed += 1
    rows.append({
        "writing_id": row["writing_id"],
        "cefr_level": row["cefr_level"],
        "n_edus":     n_edus,
        "relations":  json.dumps(relations),
    })
    done = i + 1
    if done % 50 == 0 or done == len(essays):
        print(f"  {done}/{len(essays)} parsed  (failures: {failed})")

# ── Save ──────────────────────────────────────────────────────────────────────
result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(result_df)} rows to {OUTPUT_CSV}")
print(f"Failures: {failed}")

for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
    sub = result_df[result_df["cefr_level"] == level]
    print(f"  {level}: {len(sub)} essays, avg EDUs = {sub['n_edus'].mean():.1f}")
