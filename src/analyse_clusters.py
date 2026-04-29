import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Block TensorFlow from being imported at all — it crashes on non-AVX CPUs
import sys
sys.modules["tensorflow"] = None  # type: ignore
sys.modules["tensorflow_core"] = None  # type: ignore

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap
import hdbscan

BASE = "/Users/macbook/Desktop/AIED cw2"

EMB_FILE  = f"{BASE}/clustering_embeddings.npy"
META_FILE = f"{BASE}/clustering_meta.csv"
OUT_5D    = f"{BASE}/umap_5d.npy"
OUT_2D    = f"{BASE}/umap_2d.npy"
PLOT_FILE = f"{BASE}/cluster_plot.png"

# ── Step 1: Load embeddings ───────────────────────────────────────────────────
print("Loading embeddings ...")
embeddings = np.load(EMB_FILE)
meta = pd.read_csv(META_FILE)
print(f"  Shape: {embeddings.shape}")

# ── Step 1: UMAP reduction ────────────────────────────────────────────────────
print("\nReducing to 5D (for clustering) ...")
reducer_5d = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                        metric="cosine", random_state=42)
emb_5d = reducer_5d.fit_transform(embeddings)
np.save(OUT_5D, emb_5d)
print(f"  Saved umap_5d.npy  {emb_5d.shape}")

print("Reducing to 2D (for visualisation) ...")
reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=42)
emb_2d = reducer_2d.fit_transform(embeddings)
np.save(OUT_2D, emb_2d)
print(f"  Saved umap_2d.npy  {emb_2d.shape}")

# ── Step 2: HDBSCAN clustering ────────────────────────────────────────────────
print("\nRunning HDBSCAN ...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, metric="euclidean")
labels = clusterer.fit_predict(emb_5d)

meta["cluster_id"] = labels
meta.to_csv(META_FILE, index=False)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points:   {n_noise} ({100*n_noise/len(labels):.1f}%)")

# ── Step 3: Visualisation ─────────────────────────────────────────────────────
print("\nGenerating cluster plot ...")
unique_labels = sorted(set(labels))
n_colors = max(n_clusters, 1)
palette = cm.get_cmap("tab20", n_colors)

fig, ax = plt.subplots(figsize=(12, 9))

# Plot noise first (grey, small)
noise_mask = labels == -1
ax.scatter(emb_2d[noise_mask, 0], emb_2d[noise_mask, 1],
           c="lightgrey", s=4, alpha=0.4, label="Noise", zorder=1)

# Plot clusters
for i, lbl in enumerate(lbl for lbl in unique_labels if lbl != -1):
    mask = labels == lbl
    color = palette(i / n_colors)
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=[color], s=10, alpha=0.7, label=f"Cluster {lbl}", zorder=2)

ax.set_title(f"HDBSCAN Clusters on UMAP 2D  ({n_clusters} clusters, {n_noise} noise)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.legend(loc="upper right", fontsize=7, markerscale=2,
          ncol=max(1, n_clusters // 20))
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.close()
print(f"  Saved cluster_plot.png")

# ── Step 4: Cluster inspection ────────────────────────────────────────────────
print("\n── Top 5 representative sentences per cluster ──────────────────────────")
for lbl in sorted(lbl for lbl in unique_labels if lbl != -1):
    mask = np.where(labels == lbl)[0]
    cluster_embs = emb_5d[mask]
    centroid = cluster_embs.mean(axis=0)
    dists = np.linalg.norm(cluster_embs - centroid, axis=1)
    closest_idx = mask[np.argsort(dists)[:5]]

    print(f"\nCluster {lbl}  (n={mask.shape[0]}):")
    for idx in closest_idx:
        print(f"  [{meta.iloc[idx]['cefr_level']}] {meta.iloc[idx]['sentence']}")
