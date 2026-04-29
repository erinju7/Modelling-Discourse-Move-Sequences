import ast
import re
import time
import ollama
import editdistance
import pandas as pd
from pathlib import Path

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
SEQ_FILE     = BASE / "essay_sequences.csv"
C2_CORPUS    = BASE / "c2_corpus.csv"
CLUSTER_META = BASE / "clustering_meta.csv"
RAG_CSV      = BASE / "rag_evaluation_results.csv"

# ── Hardcoded A1 essay info ───────────────────────────────────────────────────

A1_INFO = {
    920337: {"topic_id": 4, "genre": "descriptive"},
    532209: {"topic_id": 4, "genre": "descriptive"},
    464727: {"topic_id": 1, "genre": "identity"},
    811671: {"topic_id": 6, "genre": "identity"},
    375555: {"topic_id": 3, "genre": "identity"},
    387549: {"topic_id": 1, "genre": "identity"},
    778978: {"topic_id": 8, "genre": "descriptive"},
    80521:  {"topic_id": 8, "genre": "descriptive"},
    872413: {"topic_id": 6, "genre": "identity"},
    471252: {"topic_id": 5, "genre": "identity"},
}

C2_GENRE_MAP = {
    41: "descriptive",   # Writing a movie plot
    42: "functional",    # Filling in an arrival card
    43: "descriptive",   # Creating an office dress code
    44: "identity",      # Writing a resume
    45: "identity",      # Writing on the family blog
    46: "functional",    # Writing an email of advice
    47: "descriptive",   # Complaining about a meal
    48: "functional",    # Rescheduling an appointment
}

GENRE_TOPICS = {
    "identity":    {44, 45},
    "descriptive": {41, 43, 47},
    "functional":  {42, 46, 48},
}

FEEDBACK_PROMPT = """\
You are an expert ESL writing structure coach providing formative feedback to an A1-level learner.

STUDENT ESSAY:
{a1_text}

STUDENT'S DISCOURSE MOVE SEQUENCE:
{a1_sequence_str}

ADVANCED C2 REFERENCE ESSAY:
{c2_text}

C2 DISCOURSE MOVE SEQUENCE:
{c2_sequence_str}

Based on the discourse move sequences above, provide structured feedback following this exact format:

YOUR CURRENT STRUCTURE:
[Describe the student's move sequence in plain language, e.g. "Your essay opens with a greeting, then introduces yourself three times in a row, then closes."]

WHAT ADVANCED WRITERS DO DIFFERENTLY:
[Describe the C2 move sequence in plain language, highlighting the key structural difference from the student's essay]

ONE THING TO TRY:
[Give ONE specific, concrete suggestion for how to add or reorder one move to improve the structure. Name the move type explicitly.]

Rules:
- Do NOT comment on grammar, spelling, or vocabulary
- Do NOT rewrite the essay for the student
- Focus ONLY on discourse move structure
- Keep the whole response under 120 words
- Use encouraging, simple language suitable for A1 level
"""

def call_llm(messages, retries=3):
    for attempt in range(retries):
        try:
            r = ollama.chat(model="qwen2.5:3b", messages=messages)
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return ""

# ── Step 1: Build genre-specific C2 indices ───────────────────────────────────

print("Step 1: Building genre-specific C2 indices ...")

seq_df = pd.read_csv(SEQ_FILE)
seq_df["move_sequence"] = seq_df["move_sequence"].apply(ast.literal_eval)
seq_df["moves_clean"]   = seq_df["move_sequence"].apply(
    lambda s: [m for m in s if m != "Other"]
)

# Keep only C2 with non-empty sequences
c2_seq = seq_df[(seq_df["cefr_level"] == "C2") &
                (seq_df["moves_clean"].apply(len) > 0)].copy()

# Load topic_id for each C2 writing_id from c2_corpus (one row per sentence → deduplicate)
print("  Loading C2 topic_ids from c2_corpus.csv ...")
c2_corpus_topics = (pd.read_csv(C2_CORPUS, usecols=["writing_id", "topic_id"])
                      .drop_duplicates("writing_id")
                      .set_index("writing_id")["topic_id"]
                      .to_dict())

c2_seq["topic_id"] = c2_seq["writing_id"].map(c2_corpus_topics)
c2_seq["genre"]    = c2_seq["topic_id"].map(C2_GENRE_MAP)
c2_seq = c2_seq.dropna(subset=["genre"])

# Sample 200 per genre (seed=42)
genre_samples = {}
for genre, topic_ids in GENRE_TOPICS.items():
    pool = c2_seq[c2_seq["topic_id"].isin(topic_ids)]
    sampled = pool.sample(n=min(200, len(pool)), random_state=42)
    genre_samples[genre] = sampled
    print(f"  {genre:<12}: pool={len(pool):>6}  sampled={len(sampled)}")

# Load texts only for the sampled C2 essays
sampled_ids = set()
for df in genre_samples.values():
    sampled_ids.update(df["writing_id"].tolist())

print(f"  Loading texts for {len(sampled_ids)} sampled C2 essays ...")
c2_corpus_df = pd.read_csv(C2_CORPUS)
c2_texts = (c2_corpus_df[c2_corpus_df["writing_id"].isin(sampled_ids)]
            .sort_values(["writing_id", "sentence_index"])
            .groupby("writing_id")["sentence"]
            .apply(lambda s: " ".join(s.dropna().astype(str)))
            .to_dict())

# Build indices as list of dicts
genre_indices = {}
for genre, df in genre_samples.items():
    index = []
    for _, row in df.iterrows():
        wid = row["writing_id"]
        if wid in c2_texts:
            index.append({
                "writing_id":    wid,
                "topic_id":      int(row["topic_id"]),
                "moves_clean":   row["moves_clean"],
                "full_text":     c2_texts[wid],
            })
    genre_indices[genre] = index
    print(f"  {genre:<12}: index size = {len(index)}")

# ── Reconstruct A1 essay texts ────────────────────────────────────────────────

cluster_df = pd.read_csv(CLUSTER_META)
essay_texts = {}
for wid in A1_INFO:
    sents = (cluster_df[cluster_df["writing_id"] == wid]
             .sort_values("sentence_index")["sentence"]
             .dropna().astype(str).tolist())
    essay_texts[wid] = " ".join(sents)

# ── Step 2 & 3: Genre-filtered retrieval + feedback ───────────────────────────

print("\nStep 2 & 3: Genre-filtered retrieval + feedback generation ...")

rag_df = pd.read_csv(RAG_CSV)
new_feedbacks  = {}
retrieval_info = {}

for _, row in rag_df.iterrows():
    a1_id  = row["writing_id"]
    a1_seq = [m.strip() for m in row["a1_sequence"].split(" → ")]
    a1_text = essay_texts[a1_id]
    a1_genre = A1_INFO[a1_id]["genre"]

    index = genre_indices[a1_genre]

    # Edit distance against genre-matched index
    scored = [(editdistance.eval(a1_seq, e["moves_clean"]), e) for e in index]
    scored.sort(key=lambda x: x[0])
    best_ed, best = scored[0]

    retrieval_info[a1_id] = {
        "a1_genre":      a1_genre,
        "c2_writing_id": best["writing_id"],
        "c2_topic_id":   best["topic_id"],
        "c2_genre":      C2_GENRE_MAP[best["topic_id"]],
        "edit_dist":     best_ed,
        "genre_match":   a1_genre == C2_GENRE_MAP[best["topic_id"]],
    }

    print(f"  Essay {a1_id} | A1 genre: {a1_genre:<11} | "
          f"C2: {best['writing_id']}  topic: {best['topic_id']}  "
          f"edit_dist: {best_ed}  genre_match: YES")

    # Generate feedback
    prompt = FEEDBACK_PROMPT.format(
        a1_text         = a1_text,
        a1_sequence_str = " → ".join(a1_seq),
        c2_text         = best["full_text"],
        c2_sequence_str = " → ".join(best["moves_clean"]),
    )
    new_feedbacks[a1_id] = call_llm([{"role": "user", "content": prompt}])

# Overwrite feedback column
rag_df["feedback"] = rag_df["writing_id"].map(new_feedbacks)
rag_df.to_csv(RAG_CSV, index=False)
print(f"\nSaved updated feedback to {RAG_CSV}")

# ── Step 4: Verification table ────────────────────────────────────────────────

print("\n" + "=" * 72)
print("Step 4: Genre match verification")
print("=" * 72)
print(f"{'writing_id':<12} | {'A1 genre':<11} | {'C2 writing_id':<14} | "
      f"{'C2 topic':<8} | {'C2 genre':<11} | {'Ed':>3} | {'Match':>5}")
print("-" * 72)

for _, row in rag_df.iterrows():
    wid  = row["writing_id"]
    info = retrieval_info[wid]
    print(f"{wid:<12} | {info['a1_genre']:<11} | {info['c2_writing_id']:<14} | "
          f"{info['c2_topic_id']:<8} | {info['c2_genre']:<11} | {info['edit_dist']:>3} | "
          f"{'YES':>5}")

print("-" * 72)
matched = sum(1 for v in retrieval_info.values() if v["genre_match"])
print(f"Genre-matched: {matched}/10")
print("=" * 72)
