import time
import ollama
import pandas as pd
from pathlib import Path

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
RAG_CSV      = BASE / "rag_evaluation_results.csv"
CLUSTER_META = BASE / "clustering_meta.csv"
OUT_CSV      = BASE / "oneshot_feedback.csv"

# ── Genre mapping ─────────────────────────────────────────────────────────────

a1_genre = {
    920337: 'descriptive',
    532209: 'descriptive',
    464727: 'identity',
    811671: 'identity',
    375555: 'identity',
    387549: 'identity',
    778978: 'descriptive',
    80521:  'descriptive',
    872413: 'identity',
    471252: 'identity',
}

# ── One-shot examples ─────────────────────────────────────────────────────────

oneshot_examples = {
    'identity': {
        'essay': "Hi! My name is Carlos. I am from Spain. I am 28 years old. I am 28 years old. Nice to meet you!",
        'reasoning': "This essay opens with a greeting and introduces the writer. However, the age information is repeated twice in a row (Self Introduction to Self Introduction), showing limited move variety. The essay would benefit from adding a different move type after the introduction.",
        'feedback': "Your essay has a clear opening and introduction. However, you repeat the same information twice. Try adding a different move after your introduction, such as describing your daily routine or sharing an opinion, to make your structure more varied."
    },
    'descriptive': {
        'essay': "The food was good. The food was very good. The food was delicious. Goodbye!",
        'reasoning': "This essay repeats Opinion Evaluation three times in a row before a closing move, showing limited structural development. A more developed essay would vary the moves by adding specific descriptive details before closing.",
        'feedback': "Your essay shares opinions about the food. However, you repeat the same type of move three times. Try adding a descriptive move with specific details before your closing to add variety to your structure."
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(genre, essay_text):
    ex = oneshot_examples[genre]
    return (
        "You are given an essay written by an ESL learner \n"
        "and the corresponding writing task. Provide \n"
        "formative feedback on the structure and \n"
        "organisation only. Do not comment on grammar \n"
        "or vocabulary. Do not rewrite the essay.\n\n"
        "#### Writing Task: \"Write a short essay in English on the given topic.\"\n\n"
        "Example:\n"
        f"#### Student Essay: \"{ex['essay']}\"\n"
        f"### Reasoning: {ex['reasoning']}\n"
        f"#### Feedback: \"{ex['feedback']}\"\n\n"
        "Now provide feedback for this essay:\n"
        f"#### Student Essay: \"{essay_text}\"\n"
        "#### Feedback:"
    )

def call_llm(messages, retries=3):
    for attempt in range(retries):
        try:
            r = ollama.chat(model="qwen2.5:3b", messages=messages)
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return ""

# ── Load data ─────────────────────────────────────────────────────────────────

rag_df     = pd.read_csv(RAG_CSV)
cluster_df = pd.read_csv(CLUSTER_META)

# Reconstruct essay texts from clustering_meta
essay_texts = {}
for wid in rag_df["writing_id"]:
    sents = (cluster_df[cluster_df["writing_id"] == wid]
             .sort_values("sentence_index")["sentence"]
             .dropna().astype(str).tolist())
    essay_texts[wid] = " ".join(sents)

# ── Generate one-shot feedback ────────────────────────────────────────────────

print("Generating one-shot feedback (qwen2.5:3b) ...\n")

rows = []
for i, wid in enumerate(rag_df["writing_id"]):
    genre = a1_genre[wid]
    essay = essay_texts[wid]

    print(f"[{i+1}/10] writing_id={wid}  genre={genre}")

    prompt   = build_prompt(genre, essay)
    feedback = call_llm([{"role": "user", "content": prompt}])

    print(f"  Feedback: {feedback}\n")

    rows.append({
        "writing_id":       wid,
        "genre":            genre,
        "essay_text":       essay,
        "oneshot_feedback": feedback,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV}")
