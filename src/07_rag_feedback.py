import ast
import editdistance
import ollama
import pandas as pd
from pathlib import Path

BASE        = Path("/Users/macbook/Desktop/AIED cw2")
SEQ_FILE    = BASE / "essay_sequences.csv"
C2_META     = BASE / "c2_meta.csv"
CLUSTER_META = BASE / "clustering_meta.csv"
OUT_FILE    = BASE / "rag_demonstration_v3.txt"

output_lines = []   # we collect everything here, then print + save at the end

def log(text=""):
    """Print a line and store it for saving to file."""
    print(text)
    output_lines.append(text)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Select a representative A1 essay
# ─────────────────────────────────────────────────────────────────────────────
# We want an essay that is typical of A1 writing, not an outlier.
# Criteria:
#   - Between 4 and 8 sentences (typical short essay length)
#   - At least 3 distinct discourse move types (structural variety)
# Among qualifying essays, pick the one with the most unique move types.

print("Loading data ...")
seq_df = pd.read_csv(SEQ_FILE)
seq_df["move_sequence"] = seq_df["move_sequence"].apply(ast.literal_eval)

# Remove 'Other' from every sequence so we compare only meaningful moves
seq_df["moves_clean"] = seq_df["move_sequence"].apply(
    lambda s: [m for m in s if m != "Other"]
)

a1_seq = seq_df[seq_df["cefr_level"] == "A1"].copy()
a1_seq = a1_seq[a1_seq["moves_clean"].apply(len) > 0]

# Apply selection criteria
a1_seq["seq_len"]      = a1_seq["moves_clean"].apply(len)
a1_seq["n_unique"]     = a1_seq["moves_clean"].apply(lambda s: len(set(s)))

candidates = a1_seq[
    (a1_seq["seq_len"] >= 4) &
    (a1_seq["seq_len"] <= 8) &
    (a1_seq["n_unique"] >= 3)
]

# Pick the essay with the most distinct move types; break ties by seq_len
target_row   = candidates.sort_values(["n_unique", "seq_len"], ascending=False).iloc[0]
target_id    = target_row["writing_id"]
target_moves = target_row["moves_clean"]

# Load the actual sentence text for this essay from clustering_meta.csv
# (clustering_meta has sentence-level data with discourse_move labels)
cluster_meta = pd.read_csv(CLUSTER_META)
target_sentences = (
    cluster_meta[cluster_meta["writing_id"] == target_id]
    .sort_values("sentence_index")
)
target_text = " ".join(target_sentences["sentence"].tolist())

log("=" * 70)
log("STEP 1 — SELECTED A1 ESSAY")
log("=" * 70)
log(f"Writing ID   : {target_id}")
log(f"Move sequence: {target_moves}")
log(f"Sequence len : {len(target_moves)}")
log(f"\nEssay text:\n{target_text}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build the C2 retrieval index
# ─────────────────────────────────────────────────────────────────────────────
# Each C2 essay is represented as a pipe-separated string of its discourse moves
# e.g. "Social_Opening|Information_Reporting|Social_Closing"
# We also store the full reconstructed essay text for generating feedback later.

print("\nBuilding C2 retrieval index ...")

c2_seq = seq_df[seq_df["cefr_level"] == "C2"].copy()
c2_seq = c2_seq[c2_seq["moves_clean"].apply(len) > 0]

# Load C2 sentence texts and group them into full essays
c2_meta = pd.read_csv(C2_META)
c2_texts = (
    c2_meta.sort_values(["writing_id", "sentence_index"])
    .groupby("writing_id")["sentence"]
    .apply(lambda s: " ".join(s.dropna().astype(str)))
    .reset_index()
    .rename(columns={"sentence": "full_text"})
)

# Build a list of dicts, one per C2 essay
c2_index = []
for _, row in c2_seq.iterrows():
    wid = row["writing_id"]
    move_str = "|".join(row["moves_clean"])   # pipe-separated sequence string
    text_row = c2_texts[c2_texts["writing_id"] == wid]
    full_text = text_row["full_text"].values[0] if len(text_row) > 0 else ""
    if full_text:
        c2_index.append({
            "writing_id":       wid,
            "move_sequence":    row["moves_clean"],
            "move_sequence_str": move_str,
            "full_text":        full_text,
        })

print(f"  C2 essays indexed: {len(c2_index)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Retrieve the 3 most structurally similar C2 essays
# ─────────────────────────────────────────────────────────────────────────────
# Edit distance measures how many insertions/deletions/substitutions are needed
# to turn one sequence into another.  A distance of 0 = identical structure.
# We treat the move sequence as a list of tokens and compare them directly.

print("Retrieving most similar C2 essays ...")

target_seq_str = "|".join(target_moves)

for entry in c2_index:
    entry["edit_dist"] = editdistance.eval(target_moves, entry["move_sequence"])

# Sort by edit distance (ascending = most similar first)
c2_sorted = sorted(c2_index, key=lambda x: x["edit_dist"])
top3 = c2_sorted[:3]

log("\n" + "=" * 70)
log("STEP 3 — TOP 3 RETRIEVED C2 ESSAYS")
log("=" * 70)
log(f"\nA1 query sequence: {target_seq_str}")

for rank, entry in enumerate(top3, 1):
    log(f"\n── Rank {rank}  (writing_id={entry['writing_id']}, "
        f"edit_distance={entry['edit_dist']}) ──")
    log(f"Move sequence: {entry['move_sequence_str']}")
    log(f"Essay text:\n{entry['full_text']}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Generate feedback with Ollama (qwen2.5:3b)
# ─────────────────────────────────────────────────────────────────────────────
# We pass the A1 essay and the closest C2 essay to the model and ask it to
# generate formative feedback explaining the structural difference and how the
# A1 writer could improve their writing to be more like C2.

print("\nGenerating feedback with Ollama (qwen2.5:3b) ...")

best_c2 = top3[0]

a1_sequence_str = " → ".join(target_moves)
c2_sequence_str = " → ".join(best_c2["move_sequence"])
edit_dist       = best_c2["edit_dist"]

prompt = f"""
You are an ESL writing structure coach. Your job is to give formative feedback on discourse move structure — not grammar, not vocabulary, not content organisation.

STUDENT ESSAY (A1 level):
{target_text}

STUDENT'S DISCOURSE STRUCTURE:
{a1_sequence_str}

ADVANCED EXAMPLE (C2 level, most structurally similar):
{best_c2['full_text']}

C2 DISCOURSE STRUCTURE:
{c2_sequence_str}

Structural edit distance: {edit_dist}

Please give feedback that:
1. Names the student's current discourse move pattern explicitly
2. Explains what move is missing or different compared to the C2 example (focus on structure, not content)
3. Gives ONE specific suggestion on how to add or reorder a discourse move to improve structure
4. Keeps the tone encouraging and simple (A1 level student)
5. Is no longer than 150 words

Do NOT comment on grammar, spelling, vocabulary or content categories.
Focus ONLY on the sequence and variety of discourse moves.
"""

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[{"role": "user", "content": prompt}]
)
feedback = response["message"]["content"]

log("\n" + "=" * 70)
log("STEP 4 — GENERATED FEEDBACK (qwen2.5:3b)")
log("=" * 70)
log(f"\nA1 essay move sequence : {' → '.join(target_moves)}")
log(f"C2 essay move sequence : {' → '.join(best_c2['move_sequence'])}")
log(f"Edit distance          : {best_c2['edit_dist']}")
log(f"\n{feedback}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save everything to a text file
# ─────────────────────────────────────────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\nSaved full output to {OUT_FILE}")
