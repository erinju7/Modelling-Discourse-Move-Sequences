import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_CSV = Path(__file__).parent / "sampled_essays.csv"
OUTPUT_NPY = Path(__file__).parent / "embeddings.npy"
OUTPUT_META = Path(__file__).parent / "sentences_meta.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

df = pd.read_csv(INPUT_CSV)
sentences = df["sentence"].tolist()
print(f"Loaded {len(sentences)} sentences from {INPUT_CSV.name}")

print(f"Loading model '{MODEL_NAME}' ...")
model = SentenceTransformer(MODEL_NAME)

all_embeddings = []
for start in range(0, len(sentences), BATCH_SIZE):
    batch = sentences[start : start + BATCH_SIZE]
    batch_embeddings = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(batch_embeddings)
    done = min(start + BATCH_SIZE, len(sentences))
    if done % 1000 < BATCH_SIZE or done == len(sentences):
        print(f"  Encoded {done}/{len(sentences)} sentences ...")

embeddings = np.vstack(all_embeddings)
np.save(OUTPUT_NPY, embeddings)
print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"Saved embeddings to {OUTPUT_NPY}")

df.to_csv(OUTPUT_META, index=False)
print(f"Saved metadata to {OUTPUT_META}")
