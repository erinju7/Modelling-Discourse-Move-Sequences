"""
silhouette_eval.py
Compute Silhouette scores for RQ1 clustering across embedding spaces.

Output: silhouette_scores.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
LABELLED_CSV = BASE / "clustering_labelled.csv"
OUT_CSV      = BASE / "silhouette_scores.csv"

SAMPLE_SIZE  = 2000
RANDOM_STATE = 42

SPACES = {
    "Original 1024-dim (Euclidean)": (BASE / "clustering_embeddings_v2.npy", "euclidean", False),
    "Original 1024-dim (Cosine)":    (BASE / "clustering_embeddings_v2.npy", "euclidean", True),
    "UMAP 2D":                       (BASE / "umap_2d.npy",                  "euclidean", False),
    "UMAP 5D":                       (BASE / "umap_5d.npy",                  "euclidean", False),
    "UMAP 5D v2":                    (BASE / "umap_5d_v2.npy",               "euclidean", False),
}

# ── Load cluster labels ───────────────────────────────────────────────────────

df     = pd.read_csv(LABELLED_CSV)
labels = df["cluster_id_v2"].values
mask   = labels != -1   # exclude HDBSCAN noise points

labels_clean = labels[mask]
n_clusters   = len(set(labels_clean))
n_noise      = (~mask).sum()

print(f"Total points : {len(labels)}")
print(f"Noise (-1)   : {n_noise} excluded")
print(f"Clean points : {mask.sum()}")
print(f"Clusters     : {n_clusters}")
print()

# ── Compute Silhouette per space ──────────────────────────────────────────────

rows = []
for name, (path, metric, cosine_norm) in SPACES.items():
    if not path.exists():
        print(f"  SKIP {name} — file not found")
        continue

    embs = np.load(path)[mask]
    if cosine_norm:
        embs = normalize(embs)

    print(f"Computing: {name} ...")
    score = silhouette_score(
        embs, labels_clean,
        metric=metric,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    if score >= 0.5:
        interp = "Strong"
    elif score >= 0.3:
        interp = "Reasonable"
    elif score >= 0.1:
        interp = "Weak"
    else:
        interp = "Poor"

    print(f"  Silhouette = {score:.4f}  ({interp})")
    rows.append({"Space": name, "Silhouette": round(score, 4), "Interpretation": interp})

# ── Save + print summary ──────────────────────────────────────────────────────

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

print()
print("=" * 55)
print(out_df.to_string(index=False))
print("=" * 55)
print(f"\nSaved {OUT_CSV.name}")
print(f"\nNote: Clustering was performed in UMAP 5D v2 space.")
print(f"      Silhouette in that space is the primary validity metric.")
