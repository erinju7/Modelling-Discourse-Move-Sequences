import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None  # type: ignore

import numpy as np
import pandas as pd

BASE        = "/Users/macbook/Desktop/AIED cw2"
META_FILE   = f"{BASE}/clustering_meta.csv"
EMB_FILE    = f"{BASE}/clustering_embeddings_v2.npy"   # BGE embeddings used for v2 clusters
UMAP_FILE   = f"{BASE}/umap_5d_v2.npy"                 # 5D UMAP used for HDBSCAN
OUT_FILE    = f"{BASE}/cluster_inspection.txt"

CEFR_ORDER  = ["A1", "A2", "B1", "B2", "C1", "C2"]
TOP_N       = 8    # representative sentences per cluster
NOISE_N     = 10   # random noise sentences to show

print("Loading data ...")
meta   = pd.read_csv(META_FILE)
emb_5d = np.load(UMAP_FILE)   # use the same 5D space HDBSCAN clustered in

# Use cluster_id_v2 if present (BGE re-cluster), otherwise fall back
cluster_col = "cluster_id_v2" if "cluster_id_v2" in meta.columns else "cluster_id"
labels      = meta[cluster_col].values
sentences   = meta["sentence"].values
cefr        = meta["cefr_level"].values

unique_clusters = sorted(c for c in set(labels) if c != -1)
print(f"Clusters to inspect: {len(unique_clusters)}  |  noise points: {(labels == -1).sum()}\n")


def cluster_summary(lbl, mask, top_n=TOP_N):
    """Return a formatted string block for one cluster."""
    cluster_embs = emb_5d[mask]
    centroid     = cluster_embs.mean(axis=0)
    dists        = np.linalg.norm(cluster_embs - centroid, axis=1)
    top_idx      = mask[np.argsort(dists)[:top_n]]

    # CEFR breakdown
    level_counts = pd.Series(cefr[mask]).value_counts()
    total        = len(mask)
    level_str    = "  ".join(
        f"{lvl}:{level_counts.get(lvl,0)}({100*level_counts.get(lvl,0)//total}%)"
        for lvl in CEFR_ORDER if level_counts.get(lvl, 0) > 0
    )

    lines = [
        f"{'='*70}",
        f"CLUSTER {lbl}  |  n={total}",
        f"CEFR: {level_str}",
        f"Top {top_n} representative sentences:",
    ]
    for rank, idx in enumerate(top_idx, 1):
        # Clean up annotation noise for readability
        sent = sentences[idx]
        lines.append(f"  {rank}. [{cefr[idx]}] {sent}")
    lines.append("")
    return "\n".join(lines)


# ── Build full output ─────────────────────────────────────────────────────────
output_blocks = [
    f"CLUSTER INSPECTION  —  {len(unique_clusters)} clusters\n"
    f"Embedding: BAAI/bge-large-en-v1.5 with discourse prefix\n"
    f"Cluster column: {cluster_col}\n"
]

for lbl in unique_clusters:
    mask = np.where(labels == lbl)[0]
    output_blocks.append(cluster_summary(lbl, mask))

# ── Noise sample ──────────────────────────────────────────────────────────────
noise_idx = np.where(labels == -1)[0]
rng       = np.random.default_rng(42)
sample    = rng.choice(noise_idx, size=min(NOISE_N, len(noise_idx)), replace=False)

noise_lines = [
    f"{'='*70}",
    f"NOISE POINTS (cluster_id = -1)  |  total noise: {len(noise_idx)}",
    f"Random sample of {NOISE_N}:",
]
for idx in sample:
    noise_lines.append(f"  [{cefr[idx]}] {sentences[idx]}")
output_blocks.append("\n".join(noise_lines))

full_output = "\n".join(output_blocks)

# ── Print and save ────────────────────────────────────────────────────────────
print(full_output)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(full_output)

print(f"\n\nSaved to {OUT_FILE}")
