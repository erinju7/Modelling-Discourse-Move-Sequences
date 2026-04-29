import csv
import nltk
import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

LEVEL_MAP = {"1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1", "6": "C2"}
TARGET_LEVELS = set(LEVEL_MAP.keys())

XML_FILE  = Path(__file__).parent / "EFCAMDAT_Database.xml"
OUTPUT_CSV = Path(__file__).parent / "clustering_sample.csv"
N_SAMPLES = 100
SEED = 42

# ── Step 1: parse XML and collect all essays by level ──────────────────────────
print("Parsing XML (all 6 CEFR levels) ...")

essays_by_level = {v: [] for v in LEVEL_MAP.values()}

for event, elem in etree.iterparse(XML_FILE, events=("end",), recover=True):
    if elem.tag != "writing":
        continue

    level_num = elem.get("level", "")
    if level_num not in TARGET_LEVELS:
        elem.clear()
        continue

    cefr     = LEVEL_MAP[level_num]
    essay_id = elem.get("id", "")

    learner_elem = elem.find("learner")
    learner_id   = learner_elem.get("id", "") if learner_elem is not None else ""

    text_elem = elem.find("text")
    text = etree.tostring(text_elem, method="text", encoding="unicode").strip() if text_elem is not None else ""

    if text:
        essays_by_level[cefr].append({"essay_id": essay_id, "learner_id": learner_id, "cefr_level": cefr, "text": text})

    elem.clear()

for level, essays in essays_by_level.items():
    print(f"  {level}: {len(essays)} essays found")

# ── Step 2: sample 100 essay_ids per level ─────────────────────────────────────
rng = np.random.default_rng(SEED)

rows = []
for level, essays in essays_by_level.items():
    sampled = rng.choice(essays, size=N_SAMPLES, replace=False)
    for essay in sampled:
        sentences = nltk.sent_tokenize(essay["text"])
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence:
                rows.append({
                    "essay_id":       essay["essay_id"],
                    "learner_id":     essay["learner_id"],
                    "cefr_level":     level,
                    "sentence_index": idx,
                    "sentence":       sentence,
                })

# ── Step 3: save and report ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print()
for level, group in df.groupby("cefr_level"):
    print(f"{level}: {group['essay_id'].nunique()} essays, {len(group)} sentences")

print(f"\nSaved to {OUTPUT_CSV}")
