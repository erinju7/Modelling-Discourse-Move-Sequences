import ast
import re
import editdistance
import ollama
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
SEQ_FILE     = BASE / "essay_sequences.csv"
C2_META      = BASE / "c2_meta.csv"
CLUSTER_META = BASE / "clustering_meta.csv"
OUT_CSV      = BASE / "rag_evaluation_results.csv"

# Keywords used to check if the feedback mentions each discourse move type.
# These are intentionally broad so partial matches are caught.
MOVE_KEYWORDS = {
    "Social_Opening":       ["opening", "greeting", "hi", "hello", "salutation"],
    "Self_Introduction":    ["introduction", "introduce", "name", "origin", "background"],
    "Physical_Description": ["description", "describe", "appearance", "physical", "clothing"],
    "Daily_Routine":        ["routine", "daily", "schedule", "habit", "activity"],
    "Narrative_Experience": ["narrative", "experience", "story", "event", "personal"],
    "Opinion_Evaluation":   ["opinion", "evaluation", "view", "think", "feel", "perspective"],
    "Information_Reporting":["information", "reporting", "report", "fact", "detail"],
    "Social_Closing":       ["closing", "goodbye", "farewell", "thanks", "bye", "regards"],
}

# Prompt for generating structure-focused feedback (same as rag_feedback.py)
def build_prompt(a1_text, a1_seq, c2_text, c2_seq, edit_dist):
    return f"""
You are an ESL writing structure coach. Your job is to give formative feedback on discourse move structure — not grammar, not vocabulary, not content organisation.

STUDENT ESSAY (A1 level):
{a1_text}

STUDENT'S DISCOURSE STRUCTURE:
{" → ".join(a1_seq)}

ADVANCED EXAMPLE (C2 level, most structurally similar):
{c2_text}

C2 DISCOURSE STRUCTURE:
{" → ".join(c2_seq)}

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

# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def missing_moves(a1_seq, c2_seq):
    """Return moves present in C2 but absent from A1 (the structural gap)."""
    return list(set(c2_seq) - set(a1_seq))

def metric1_move_mention(feedback_text, missing):
    """
    Score = 1 if the feedback mentions at least one missing move via keyword,
    0 otherwise.
    """
    if not missing:
        return 1   # no gap → trivially satisfied
    feedback_lower = feedback_text.lower()
    for move in missing:
        for kw in MOVE_KEYWORDS.get(move, []):
            if kw in feedback_lower:
                return 1
    return 0

def metric2_structural_focus(feedback_text):
    """
    Count all discourse-move keyword occurrences in the feedback, divided by
    total word count.  Higher = more structure-focused language.
    """
    words = feedback_text.lower().split()
    if not words:
        return 0.0
    all_keywords = [kw for kws in MOVE_KEYWORDS.values() for kw in kws]
    keyword_hits = sum(1 for w in words if w.strip(".,!?;:\"'") in all_keywords)
    return round(keyword_hits / len(words), 4)

def metric3_edit_reduction(feedback_text, a1_seq, c2_seq, missing):
    """
    If the feedback mentions a missing move, simulate the student adding it to
    their sequence and recalculate edit distance.
    Score = (original_dist - new_dist) / original_dist
    """
    orig_dist = editdistance.eval(a1_seq, c2_seq)
    if orig_dist == 0:
        return 1.0
    feedback_lower = feedback_text.lower()
    for move in missing:
        for kw in MOVE_KEYWORDS.get(move, []):
            if kw in feedback_lower:
                # Simulate student appending the missing move to their sequence
                improved_seq = a1_seq + [move]
                new_dist = editdistance.eval(improved_seq, c2_seq)
                return round((orig_dist - new_dist) / orig_dist, 4)
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Select 10 representative A1 essays
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data ...")
seq_df = pd.read_csv(SEQ_FILE)
seq_df["move_sequence"] = seq_df["move_sequence"].apply(ast.literal_eval)
seq_df["moves_clean"]   = seq_df["move_sequence"].apply(
    lambda s: [m for m in s if m != "Other"]
)

a1_all = seq_df[seq_df["cefr_level"] == "A1"].copy()
a1_all["seq_len"]  = a1_all["moves_clean"].apply(len)
a1_all["n_unique"] = a1_all["moves_clean"].apply(lambda s: len(set(s)))

# Filter: 4-8 sentences, at least 2 distinct move types
candidates = a1_all[
    (a1_all["seq_len"] >= 4) &
    (a1_all["seq_len"] <= 8) &
    (a1_all["n_unique"] >= 2)
].copy()

# Sample 10 essays with random seed 42
sampled = candidates.sample(n=10, random_state=42).reset_index(drop=True)
print(f"  Qualifying A1 essays: {len(candidates)}  |  Sampled: {len(sampled)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build C2 retrieval index (same as rag_feedback.py)
# ─────────────────────────────────────────────────────────────────────────────
print("Building C2 retrieval index ...")

c2_seq_df = seq_df[seq_df["cefr_level"] == "C2"].copy()
c2_seq_df = c2_seq_df[c2_seq_df["moves_clean"].apply(len) > 0]

c2_meta = pd.read_csv(C2_META)
c2_texts = (
    c2_meta.sort_values(["writing_id", "sentence_index"])
    .groupby("writing_id")["sentence"]
    .apply(lambda s: " ".join(s.dropna().astype(str)))
    .reset_index()
    .rename(columns={"sentence": "full_text"})
)

c2_index = []
for _, row in c2_seq_df.iterrows():
    wid = row["writing_id"]
    text_row = c2_texts[c2_texts["writing_id"] == wid]
    full_text = text_row["full_text"].values[0] if len(text_row) > 0 else ""
    if full_text:
        c2_index.append({
            "writing_id":    wid,
            "move_sequence": row["moves_clean"],
            "full_text":     full_text,
        })

print(f"  C2 essays indexed: {len(c2_index)}")

# Load A1 sentence texts from clustering_meta
cluster_meta = pd.read_csv(CLUSTER_META)

# ───────────────────────────────��─────────────────────────────────────────────
# Steps 2-4: Run RAG pipeline + evaluate for each of the 10 essays
# ─────────────────────────────────────────────────────────────────────────────
results = []

for i, row in sampled.iterrows():
    a1_id    = row["writing_id"]
    a1_moves = row["moves_clean"]

    # Get A1 essay text
    a1_sentences = (
        cluster_meta[cluster_meta["writing_id"] == a1_id]
        .sort_values("sentence_index")
    )
    a1_text = " ".join(a1_sentences["sentence"].dropna().astype(str).tolist())

    # Retrieve top 1 C2 essay by edit distance
    for entry in c2_index:
        entry["edit_dist"] = editdistance.eval(a1_moves, entry["move_sequence"])
    best_c2 = min(c2_index, key=lambda x: x["edit_dist"])

    edit_dist = best_c2["edit_dist"]
    c2_moves  = best_c2["move_sequence"]
    c2_text   = best_c2["full_text"]

    print(f"\n[{i+1}/10] essay {a1_id}  |  A1 seq len={len(a1_moves)}  |  edit_dist={edit_dist}")

    # Generate feedback
    prompt   = build_prompt(a1_text, a1_moves, c2_text, c2_moves, edit_dist)
    response = ollama.chat(model="qwen2.5:3b",
                           messages=[{"role": "user", "content": prompt}])
    feedback = response["message"]["content"]

    # Evaluate
    gap   = missing_moves(a1_moves, c2_moves)
    m1    = metric1_move_mention(feedback, gap)
    m2    = metric2_structural_focus(feedback)
    m3    = metric3_edit_reduction(feedback, a1_moves, c2_moves, gap)

    print(f"  Missing moves: {gap}")
    print(f"  M1={m1}  M2={m2:.4f}  M3={m3:.4f}")

    results.append({
        "writing_id":        a1_id,
        "a1_sequence":       " → ".join(a1_moves),
        "c2_sequence":       " → ".join(c2_moves),
        "edit_distance":     edit_dist,
        "missing_moves":     str(gap),
        "feedback":          feedback,
        "metric1_move_mention":      m1,
        "metric2_structural_focus":  m2,
        "metric3_edit_reduction":    m3,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Step 4 + 5: Print results table and save
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved results to {OUT_CSV}")

# Print summary table
print("\n" + "=" * 90)
print(f"{'ID':<12} {'Seq len':>7} {'Edit dist':>9} {'M1':>4} {'M2':>8} {'M3':>8}")
print("=" * 90)
for _, r in results_df.iterrows():
    seq_len = len(r["a1_sequence"].split(" → "))
    print(f"{r['writing_id']:<12} {seq_len:>7} {r['edit_distance']:>9} "
          f"{r['metric1_move_mention']:>4} {r['metric2_structural_focus']:>8.4f} "
          f"{r['metric3_edit_reduction']:>8.4f}")

print("-" * 90)
print(f"{'AVERAGE':<12} {'':>7} {'':>9} "
      f"{results_df['metric1_move_mention'].mean():>4.2f} "
      f"{results_df['metric2_structural_focus'].mean():>8.4f} "
      f"{results_df['metric3_edit_reduction'].mean():>8.4f}")
print("=" * 90)
print("\nMetric legend:")
print("  M1 — Move Mention Accuracy  (1 = missing move mentioned, 0 = not)")
print("  M2 — Structural Focus Score (keyword density in feedback)")
print("  M3 — Edit Distance Reduction Potential (0-1, higher = more helpful)")
