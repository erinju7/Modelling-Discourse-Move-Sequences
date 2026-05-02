"""
07_select_eval_essays.py
Sample 30 A1 evaluation essays from EFCAMDAT that were NOT in the clustering sample.
Selection criteria: 4-8 sentences, ≥2 distinct discourse move types.

Output: eval_essays_n30.csv
"""

import ast, numpy as np, pandas as pd, nltk
from pathlib import Path
from lxml import etree
from sentence_transformers import SentenceTransformer

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
XML_FILE     = BASE / "EFCAMDAT_Database.xml"
LABELLED_CSV = BASE / "clustering_labelled.csv"
EMB_FILE     = BASE / "clustering_embeddings_v2.npy"
OUT_CSV      = BASE / "eval_essays_n30.csv"

N_EVAL         = 30
SEED           = 42
MIN_SENT       = 4
MAX_SENT       = 8
MIN_MOVE_TYPES = 2

A1_TOPIC_TASK_TYPE = {
    1: "narrative", 2: "descriptive", 3: "narrative", 4: "descriptive",
    5: "narrative", 6: "narrative",   7: "descriptive", 8: "descriptive",
}

CLUSTER_TO_MOVE = {
    1:"Social_Opening", 15:"Social_Opening",
    2:"Self_Introduction", 3:"Self_Introduction", 24:"Self_Introduction", 25:"Self_Introduction",
    4:"Physical_Description", 14:"Physical_Description", 17:"Physical_Description", 36:"Physical_Description",
    9:"Daily_Routine", 12:"Daily_Routine", 13:"Daily_Routine", 19:"Daily_Routine", 20:"Daily_Routine", 21:"Daily_Routine",
    6:"Narrative_Experience", 7:"Narrative_Experience", 28:"Narrative_Experience", 37:"Narrative_Experience", 40:"Narrative_Experience",
    18:"Opinion_Evaluation", 22:"Opinion_Evaluation", 23:"Opinion_Evaluation", 29:"Opinion_Evaluation",
    33:"Opinion_Evaluation", 34:"Opinion_Evaluation", 39:"Opinion_Evaluation",
    10:"Information_Reporting", 11:"Information_Reporting", 26:"Information_Reporting", 27:"Information_Reporting",
    30:"Information_Reporting", 32:"Information_Reporting", 38:"Information_Reporting",
    41:"Information_Reporting", 42:"Information_Reporting", 43:"Information_Reporting",
    0:"Social_Closing", 5:"Social_Closing", 8:"Social_Closing", 16:"Social_Closing", 31:"Social_Closing", 35:"Social_Closing",
    -1:"Other",
}

# ── Load RQ1 classifier artifacts ─────────────────────────────────────────────

print("Loading RQ1 artifacts ...")
labelled   = pd.read_csv(LABELLED_CSV)
embeddings = np.load(EMB_FILE)

centroids = {}
for cid in labelled["cluster_id_v2"].unique():
    mask = labelled["cluster_id_v2"].values == cid
    centroids[int(cid)] = embeddings[mask].mean(axis=0)

used_ids = set(labelled["writing_id"].unique())
print(f"  {len(centroids)} cluster centroids")
print(f"  {len(used_ids)} essay IDs excluded (clustering sample)")

PREFIX = "Represent the rhetorical or communicative function of this sentence: "
print("Loading BGE-large-en-v1.5 ...")
bge = SentenceTransformer("BAAI/bge-large-en-v1.5")

# ── Classifier ────────────────────────────────────────────────────────────────

def classify_moves(sentences):
    prefixed = [PREFIX + s for s in sentences]
    embs = bge.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    cluster_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[c] for c in cluster_ids])
    dists = np.linalg.norm(embs[:, None, :] - centroid_matrix[None, :, :], axis=2)
    nearest = np.argmin(dists, axis=1)
    return [CLUSTER_TO_MOVE.get(cluster_ids[n], "Other") for n in nearest]

# ── Parse XML ─────────────────────────────────────────────────────────────────

print(f"\nParsing {XML_FILE.name} ...")
candidates = []
CANDIDATE_TARGET = N_EVAL * 10  # collect 300 candidates then stop

for event, elem in etree.iterparse(XML_FILE, events=("end",), recover=True):
    if elem.tag != "writing":
        continue  # do NOT clear children before parent is processed
    if elem.get("level", "") != "1":
        elem.clear(); continue

    essay_id = int(elem.get("id", 0))
    topic_id = int(elem.get("unit", 0))

    if essay_id in used_ids:
        elem.clear(); continue

    text_elem = elem.find("text")
    text = etree.tostring(text_elem, method="text", encoding="unicode").strip() if text_elem is not None else ""
    elem.clear()

    if not text:
        continue

    sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    if not (MIN_SENT <= len(sentences) <= MAX_SENT):
        continue

    candidates.append({
        "writing_id": essay_id,
        "topic_id":   topic_id,
        "text":       text,
        "sentences":  sentences,
    })

    if len(candidates) >= CANDIDATE_TARGET:
        print(f"  Reached {CANDIDATE_TARGET} candidates — stopping early")
        break

print(f"Found {len(candidates)} candidate A1 essays (4–8 sentences, not in clustering sample)")

# ── Classify and filter ────────────────────────────────────────────────────────

print(f"Classifying {len(candidates)} essays ...")
valid = []

for i, c in enumerate(candidates):
    moves   = classify_moves(c["sentences"])
    distinct = set(m for m in moves if m != "Other")
    if len(distinct) >= MIN_MOVE_TYPES:
        c["move_sequence"]  = moves
        c["n_sentences"]    = len(c["sentences"])
        c["n_move_types"]   = len(distinct)
        c["task_type"]      = A1_TOPIC_TASK_TYPE.get(c["topic_id"], "descriptive")
        valid.append(c)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(candidates)} classified, {len(valid)} valid so far")

print(f"\n{len(valid)} essays pass all filters")

# ── Sample 30 ─────────────────────────────────────────────────────────────────

rng    = np.random.default_rng(SEED)
chosen = rng.choice(len(valid), size=min(N_EVAL, len(valid)), replace=False)
sample = [valid[i] for i in sorted(chosen)]

rows = [{
    "writing_id":    e["writing_id"],
    "topic_id":      e["topic_id"],
    "task_type":     e["task_type"],
    "essay_text":    e["text"],
    "n_sentences":   e["n_sentences"],
    "n_move_types":  e["n_move_types"],
    "move_sequence": str(e["move_sequence"]),
} for e in sample]

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(out_df)} essays to {OUT_CSV}")
print(out_df[["writing_id","topic_id","task_type","n_sentences","n_move_types"]].to_string(index=False))
