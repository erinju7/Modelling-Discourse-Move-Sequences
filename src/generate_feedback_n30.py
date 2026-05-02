"""
generate_feedback_n30.py
Generate RAG, baseline, and one-shot feedback for 30 A1 essays using qwen2.5:7b.
Replaces scripts 07, 08, 09. All three conditions in one pass.

Output: feedback_n30.csv
"""

import ast, os, time, editdistance, ollama
import numpy as np, pandas as pd
from pathlib import Path

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
EVAL_CSV     = BASE / "eval_essays_n30.csv"
SEQ_FILE     = BASE / "essay_sequences.csv"
C2_CORPUS    = BASE / "c2_corpus.csv"
OUT_CSV      = BASE / os.environ.get("OUT_CSV", "feedback_n30.csv")
MAX_ROWS     = int(os.environ.get("MAX_ROWS", "0"))

MODEL = "qwen2.5:7b"

# ── Task-type mappings (narrative / descriptive, following Michel et al. 2019) ─

A1_TOPIC_TASK_TYPE = {
    1: "narrative",
    2: "descriptive",
    3: "narrative",
    4: "descriptive",
    5: "narrative",
    6: "narrative",
    7: "descriptive",
    8: "descriptive",
}

A1_TOPIC_NAMES = {
    1: "Introducing yourself by email",
    2: "Taking inventory in the office",
    3: "Writing an online profile",
    4: "Describing your family in an email",
    5: "Updating your online profile",
    6: "Signing up for a dating website",
    7: "Writing labels for a clothing store",
    8: "Making a dinner party menu",
}

C2_TOPIC_TASK_TYPE = {
    41: "narrative",    42: "descriptive", 43: "descriptive",
    44: "narrative",    45: "narrative",   46: "descriptive",
    47: "descriptive",  48: "descriptive",
}

C2_TOPIC_NAMES = {
    41: "Writing a movie plot",
    42: "Filling in an arrival card",
    43: "Creating an office dress code",
    44: "Writing a resume",
    45: "Writing on the family blog",
    46: "Writing an email of advice",
    47: "Complaining about a meal",
    48: "Rescheduling an appointment",
}

C2_TASK_TYPE_TOPICS = {
    "narrative":   {41, 44, 45},
    "descriptive": {42, 43, 46, 47, 48},
}

# ── One-shot examples (task-type matched) ────────────────────────────────────

ONESHOT = {
    "narrative": {
        "writing_id": 696,
        "task_topic": "Updating your online profile",
        "essay": "Hi! My name's Ronaldo. I'm twenty-nine years old. I'm from Brazil. I live at Ceara State, in northeastern brazilian coast. I'm a fisheries engineer and I have been a graduate student as member of the Animal Ecology Laboratory, at Federal University of Ceara. Best wishes, Ronaldo",
        "feedback": "Your profile is easy to follow because you tell the reader about yourself in a clear order. First your personal details, then your work and study. This is good. One suggestion is to group the information about your job and university together at the end to make it look more organized.",
    },
    "descriptive": {
        "writing_id": 130,
        "task_topic": "Making a dinner party menu",
        "essay": "Hi! This is the menu that I will do to dinner, see if you like: Starter: Vegetables and soup. Drinks: Water, beer, wine and whisk. Main course: Roast beef, rice, noodles and potato chip. Dessert: Ice Cream and maracuj cream.",
        "feedback": "Your menu is easy to read because you put the food into different parts like starter, drinks, main course, and dessert. This helps the reader understand your plan. One suggestion is to write one short sentence at the beginning to tell the reader this is your dinner party menu.",
    },
}

# ── Prompts ───────────────────────────────────────────────────────────────────

RULES = """\
Rules:
- Do NOT comment on grammar, spelling, or vocabulary
- Do NOT rewrite the essay
- Focus ONLY on discourse move structure
- Keep the response under 120 words
- Use encouraging, simple language suitable for A1 level"""

RAG_PROMPT = """\
You are an expert ESL writing structure coach providing formative feedback to an A1-level learner.

A1 Writing Task Topic: "{a1_task_topic}"

Student Essay:
{a1_text}

Student's Discourse Move Sequence:
{a1_sequence}

Retrieved C2 Structural Reference (move sequence only):
{c2_sequence}

Structural summary of the retrieved C2 sequence:
{c2_summary}

Important restriction:
- The retrieved C2 reference may come from a different writing topic.
- Use it only as a contrastive structural hint.
- Do NOT infer topic content, events, goals, or details from the missing C2 essay text.
- Do NOT treat the retrieved C2 sequence as a target pattern that the student should copy exactly.
- Compare the two sequences to notice one useful structural difference.
- All suggestions must be appropriate for the A1 Writing Task Topic.
- Give advice that is suitable for an A1 learner even if it does not mirror the retrieved C2 sequence.

Use the retrieved C2 sequence only to help you notice structural contrasts.
Your goal is not to reproduce the C2 pattern, but to give one helpful,
task-appropriate suggestion for this A1 learner.

Based on the discourse move sequences above, provide feedback in this exact format:

YOUR CURRENT STRUCTURE:
[Describe the student's move sequence in plain language]

WHAT ADVANCED WRITERS DO DIFFERENTLY:
[Briefly describe one structural difference between the student's sequence and the retrieved C2 sequence. Do not imply that the student should copy the C2 pattern exactly.]

ONE THING TO TRY:
[Give ONE simple suggestion that improves the student's structure for the A1 task topic. The suggestion should be learner-appropriate and does not need to reproduce the retrieved C2 sequence. Name the move type explicitly.]

""" + RULES

BASELINE_PROMPT = """\
You are an expert ESL writing structure coach providing formative feedback to an A1-level learner.

A1 Writing Task Topic: "{task_topic}"

Student Essay:
{essay_text}

Provide formative feedback on the structure and organisation of this essay only.

""" + RULES


def oneshot_prompt(task_type, task_topic, essay_text):
    ex = ONESHOT[task_type]
    return (
        'You are an expert ESL writing structure coach providing formative feedback to an A1-level learner.\n\n'
        f'A1 Writing Task Topic: "{task_topic}"\n\n'
        f'Example ({task_type} task type):\n'
        f'Student Essay: "{ex["essay"]}"\n'
        f'Feedback: "{ex["feedback"]}"\n\n'
        'Now provide feedback for this essay:\n'
        f'A1 Writing Task Topic: "{task_topic}"\n'
        f'Student Essay: "{essay_text}"\n\n'
        + RULES + '\n\nFeedback:'
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(prompt, retries=3):
    for attempt in range(retries):
        try:
            r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}"); time.sleep(2)
    return ""


def summarize_move_sequence(seq):
    """Create a compact structural summary without exposing C2 topic content."""
    if not seq:
        return "No move sequence available."

    summary_bits = [f"The sequence contains {len(seq)} moves."]
    unique_moves = []
    for move in seq:
        if move not in unique_moves:
            unique_moves.append(move)
    summary_bits.append("It uses these move types: " + ", ".join(unique_moves) + ".")

    repeated_runs = []
    run_move = seq[0]
    run_len = 1
    for move in seq[1:]:
        if move == run_move:
            run_len += 1
        else:
            if run_len > 1:
                repeated_runs.append(f"{run_move} x{run_len}")
            run_move = move
            run_len = 1
    if run_len > 1:
        repeated_runs.append(f"{run_move} x{run_len}")
    if repeated_runs:
        summary_bits.append("Repeated runs: " + ", ".join(repeated_runs) + ".")

    summary_bits.append(f"It begins with {seq[0]} and ends with {seq[-1]}.")
    return " ".join(summary_bits)


# ── Load eval essays ──────────────────────────────────────────────────────────

print("Loading eval essays ...")
eval_df = pd.read_csv(EVAL_CSV).rename(columns={"genre": "task_type"})
if MAX_ROWS > 0:
    eval_df = eval_df.head(MAX_ROWS)

essay_data = {}
for _, row in eval_df.iterrows():
    moves = ast.literal_eval(row["move_sequence"])
    essay_data[int(row["writing_id"])] = {
        "text":        row["essay_text"],
        "topic_id":    int(row["topic_id"]),
        "task_topic":  A1_TOPIC_NAMES.get(int(row["topic_id"]), f"Topic {int(row['topic_id'])}"),
        "task_type":   row["task_type"],
        "moves":       moves,
        "moves_clean": [m for m in moves if m != "Other"],
    }

sample_ids       = list(essay_data.keys())
task_type_counts = pd.Series([essay_data[w]["task_type"] for w in sample_ids]).value_counts().to_dict()
print(f"Loaded {len(sample_ids)} essays  (task types: {task_type_counts})")

# ── Build task-type retrieval index ───────────────────────────────────────────

print("\nBuilding task-type retrieval index ...")
seq_df = pd.read_csv(SEQ_FILE)
seq_df["move_sequence"] = seq_df["move_sequence"].apply(ast.literal_eval)
seq_df["moves_clean"]   = seq_df["move_sequence"].apply(lambda s: [m for m in s if m != "Other"])
c2_seq = seq_df[(seq_df["cefr_level"] == "C2") & (seq_df["moves_clean"].apply(len) > 0)].copy()

c2_corpus_df = pd.read_csv(C2_CORPUS)
c2_topic_map = (c2_corpus_df.drop_duplicates("writing_id")
                .set_index("writing_id")["topic_id"].to_dict())
c2_seq["topic_id"]  = c2_seq["writing_id"].map(c2_topic_map)
c2_seq["task_type"] = c2_seq["topic_id"].map(C2_TOPIC_TASK_TYPE)
c2_seq = c2_seq.dropna(subset=["task_type"])

c2_texts = (c2_corpus_df.sort_values(["writing_id", "sentence_index"])
            .groupby("writing_id")["sentence"]
            .apply(lambda s: " ".join(s.dropna().astype(str)))
            .to_dict())

task_index = {}
for task_type, topic_ids in C2_TASK_TYPE_TOPICS.items():
    pool    = c2_seq[c2_seq["topic_id"].isin(topic_ids)]
    sampled = pool.sample(n=min(200, len(pool)), random_state=42)
    task_index[task_type] = [
        {"writing_id":  r["writing_id"],
         "topic_id":    int(r["topic_id"]),
         "task_topic":  C2_TOPIC_NAMES.get(int(r["topic_id"]), f"Topic {int(r['topic_id'])}"),
         "moves_clean": r["moves_clean"],
         "full_text":   c2_texts.get(r["writing_id"], "")}
        for _, r in sampled.iterrows() if r["writing_id"] in c2_texts
    ]
    print(f"  {task_type}: {len(task_index[task_type])} C2 essays indexed")

# ── Generate feedback ─────────────────────────────────────────────────────────

print(f"\nGenerating feedback ({MODEL}) ...\n")

rows = []
for i, wid in enumerate(sample_ids):
    info      = essay_data[wid]
    task_type = info["task_type"]
    task_topic = info["task_topic"]
    text      = info["text"]
    a1_seq    = info["moves_clean"]

    scored = [(editdistance.eval(a1_seq, e["moves_clean"]), e) for e in task_index[task_type]]
    scored.sort(key=lambda x: x[0])
    best_ed, best_c2 = scored[0]
    c2_summary = summarize_move_sequence(best_c2["moves_clean"])

    print(f"[{i+1}/{len(sample_ids)}] writing_id={wid}  task_type={task_type}  edit_dist={best_ed}")

    fb_rag      = call_llm(RAG_PROMPT.format(
                      a1_task_topic=task_topic,
                      a1_text=text, a1_sequence=" -> ".join(a1_seq),
                      c2_sequence=" -> ".join(best_c2["moves_clean"]),
                      c2_summary=c2_summary))
    fb_baseline = call_llm(BASELINE_PROMPT.format(task_topic=task_topic, essay_text=text))
    fb_oneshot  = call_llm(oneshot_prompt(task_type, task_topic, text))

    print(f"  RAG:      {fb_rag[:70]}...")
    print(f"  Baseline: {fb_baseline[:70]}...")
    print(f"  One-shot: {fb_oneshot[:70]}...\n")

    rows.append({
        "writing_id":        wid,
        "topic_id":          info["topic_id"],
        "task_topic":        task_topic,
        "task_type":         task_type,
        "essay_text":        text,
        "move_sequence":     " -> ".join(info["moves"]),
        "c2_writing_id":     best_c2["writing_id"],
        "c2_topic_id":       best_c2["topic_id"],
        "c2_task_topic":     best_c2["task_topic"],
        "c2_sequence":       " -> ".join(best_c2["moves_clean"]),
        "c2_structural_summary": c2_summary,
        "edit_distance":     best_ed,
        "feedback_rag":      fb_rag,
        "feedback_baseline": fb_baseline,
        "feedback_oneshot":  fb_oneshot,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(out_df)} rows to {OUT_CSV}")
