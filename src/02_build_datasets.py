import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

LEVEL_MAP = {"1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1", "6": "C2"}

XML_FILE        = Path(__file__).parent / "EFCAMDAT_Database.xml"
CLUSTERING_CSV  = Path(__file__).parent / "clustering_sample.csv"
C2_CORPUS_CSV   = Path(__file__).parent / "c2_corpus.csv"

N_SAMPLES = 100
SEED = 42

# ── Step 1: parse XML into memory ─────────────────────────────────────────────
print("Parsing XML ...")

essays_by_level = {v: [] for v in LEVEL_MAP.values()}

for event, elem in etree.iterparse(XML_FILE, events=("end",), recover=True):
    if elem.tag != "writing":
        continue

    level_num = elem.get("level", "")
    cefr = LEVEL_MAP.get(level_num)
    if cefr is None:
        elem.clear()
        continue

    writing_id = elem.get("id", "")

    topic_elem = elem.find("topic")
    topic_id   = topic_elem.get("id", "") if topic_elem is not None else ""
    topic_text = (topic_elem.text or "").strip() if topic_elem is not None else ""

    text_elem = elem.find("text")
    text = etree.tostring(text_elem, method="text", encoding="unicode").strip() if text_elem is not None else ""

    if not text:
        elem.clear()
        continue

    essays_by_level[cefr].append({
        "writing_id": writing_id,
        "cefr_level": cefr,
        "topic_id":   topic_id,
        "topic":      topic_text,
        "text":       text,
    })
    elem.clear()

for level, essays in essays_by_level.items():
    print(f"  {level}: {len(essays)} essays")


def essays_to_sentences(essay_list):
    rows = []
    for essay in essay_list:
        sentences = nltk.sent_tokenize(essay["text"])
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                rows.append({
                    "writing_id":     essay["writing_id"],
                    "cefr_level":     essay["cefr_level"],
                    "topic_id":       essay["topic_id"],
                    "sentence_index": idx,
                    "sentence":       sentence,
                })
    return rows


# ── Dataset 1: clustering_sample.csv (100 essays × 6 levels) ─────────────────
print("\nBuilding clustering_sample.csv ...")

rng = np.random.default_rng(SEED)
cluster_rows = []

for level, essays in essays_by_level.items():
    sampled = rng.choice(essays, size=N_SAMPLES, replace=False).tolist()
    cluster_rows.extend(essays_to_sentences(sampled))

cluster_df = pd.DataFrame(cluster_rows)
cluster_df.to_csv(CLUSTERING_CSV, index=False)

print("\nclustering_sample.csv summary:")
for level, grp in cluster_df.groupby("cefr_level"):
    print(f"  {level}: {grp['writing_id'].nunique()} essays, {len(grp)} sentences")


# ── Dataset 2: c2_corpus.csv (all C2 essays) ─────────────────────────────────
print("\nBuilding c2_corpus.csv ...")

c2_rows = essays_to_sentences(essays_by_level["C2"])
c2_df = pd.DataFrame(c2_rows)
c2_df.to_csv(C2_CORPUS_CSV, index=False)

print(f"\nc2_corpus.csv summary:")
print(f"  C2: {c2_df['writing_id'].nunique()} essays, {len(c2_df)} sentences")

print(f"\nSaved: {CLUSTERING_CSV}")
print(f"Saved: {C2_CORPUS_CSV}")
