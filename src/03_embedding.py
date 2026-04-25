import os
os.environ["USE_TF"] = "0"          # stop sentence-transformers loading TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

DATASETS = [
    {
        "input":    BASE / "clustering_sample.csv",
        "emb_out":  BASE / "clustering_embeddings.npy",
        "meta_out": BASE / "clustering_meta.csv",
        "label":    "clustering_sample",
    },
    {
        "input":    BASE / "c2_corpus.csv",
        "emb_out":  BASE / "c2_embeddings.npy",
        "meta_out": BASE / "c2_meta.csv",
        "label":    "c2_corpus",
    },
]

print(f"Loading model '{MODEL_NAME}' ...")
model = SentenceTransformer(MODEL_NAME)

for ds in DATASETS:
    print(f"\n── {ds['label']} ──────────────────────────────")
    df = pd.read_csv(ds["input"])
    sentences = df["sentence"].tolist()
    print(f"  Sentences to encode: {len(sentences)}")

    t0 = time.time()
    all_embeddings = []
    last_reported = -1

    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        all_embeddings.append(model.encode(batch, show_progress_bar=False))

        done = min(start + BATCH_SIZE, len(sentences))
        milestone = (done // 500) * 500
        if milestone > last_reported and milestone > 0:
            elapsed = time.time() - t0
            print(f"  {done}/{len(sentences)} sentences encoded  ({elapsed:.1f}s)")
            last_reported = milestone

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0

    np.save(ds["emb_out"], embeddings)
    df.to_csv(ds["meta_out"], index=False)

    print(f"  Done — {len(sentences)} sentences in {elapsed:.1f}s")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved: {ds['emb_out'].name}")
    print(f"  Saved: {ds['meta_out'].name}")
