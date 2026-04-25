import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None  # type: ignore

import numpy as np
import pandas as pd
import umap
import hdbscan
from sentence_transformers import SentenceTransformer

BASE = "/Users/macbook/Desktop/AIED cw2"
META_FILE   = f"{BASE}/clustering_meta.csv"
EMB_OUT_V2  = f"{BASE}/clustering_embeddings_v2.npy"
UMAP_5D_V2  = f"{BASE}/umap_5d_v2.npy"
BATCH_SIZE  = 64
PREFIX = "Represent the rhetorical or communicative function of this sentence: "

# ── Step 1: Re-embed with instruction prefix ──────────────────────────────────
print("Loading metadata ...")
meta = pd.read_csv(META_FILE)
sentences = meta["sentence"].tolist()
print(f"  {len(sentences)} sentences")

print("\nLoading model BAAI/bge-large-en-v1.5 ...")
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

prefixed = [PREFIX + s for s in sentences]

print("Encoding sentences ...")
all_embeddings = []
for start in range(0, len(prefixed), BATCH_SIZE):
    batch = prefixed[start : start + BATCH_SIZE]
    all_embeddings.append(model.encode(batch, show_progress_bar=False, normalize_embeddings=True))
    done = min(start + BATCH_SIZE, len(prefixed))
    if done % 500 < BATCH_SIZE or done == len(prefixed):
        print(f"  {done}/{len(prefixed)} encoded ...")

embeddings_v2 = np.vstack(all_embeddings)
np.save(EMB_OUT_V2, embeddings_v2)
print(f"  Saved clustering_embeddings_v2.npy  shape={embeddings_v2.shape}")

# ── Step 2: UMAP + HDBSCAN ────────────────────────────────────────────────────
print("\nRunning UMAP (5D) ...")
reducer = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                    metric="cosine", random_state=42)
emb_5d = reducer.fit_transform(embeddings_v2)
np.save(UMAP_5D_V2, emb_5d)
print(f"  Saved umap_5d_v2.npy  shape={emb_5d.shape}")

print("Running HDBSCAN ...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=5, metric="euclidean")
labels = clusterer.fit_predict(emb_5d)

meta["cluster_id_v2"] = labels
meta.to_csv(META_FILE, index=False)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points:   {n_noise} ({100*n_noise/len(labels):.1f}%)")

# ── Step 3: Cluster inspection ────────────────────────────────────────────────
cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
cefr_levels = meta["cefr_level"].values

print("\n── Cluster inspection ───────────────────────────────────────────────────")
for lbl in sorted(lbl for lbl in set(labels) if lbl != -1):
    mask = np.where(labels == lbl)[0]
    cluster_embs = emb_5d[mask]
    centroid = cluster_embs.mean(axis=0)
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    closest = mask[np.argsort(dists)[:5]]

    # CEFR distribution
    level_counts = pd.Series(cefr_levels[mask]).value_counts()
    level_pct = {lvl: f"{100*level_counts.get(lvl,0)/len(mask):.0f}%" for lvl in cefr_order}
    level_str = "  ".join(f"{lvl}:{level_pct[lvl]}" for lvl in cefr_order if level_counts.get(lvl,0) > 0)

    print(f"\nCluster {lbl}  (n={len(mask)})  [{level_str}]")
    for idx in closest:
        print(f"  [{cefr_levels[idx]}] {meta.iloc[idx]['sentence']}")
