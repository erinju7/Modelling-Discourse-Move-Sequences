import re
import time
import ollama
import pandas as pd
from pathlib import Path

BASE          = Path("/Users/macbook/Desktop/AIED cw2")
RAG_CSV       = BASE / "rag_evaluation_results.csv"
CLUSTER_META  = BASE / "clustering_meta.csv"
BASELINE_CSV  = BASE / "baseline_feedback.csv"
COMPARE_CSV   = BASE / "comparison_results.csv"

MOVE_KEYWORDS = {
    "Social_Opening":        ["opening", "greeting", "hi", "hello", "salutation"],
    "Self_Introduction":     ["introduction", "introduce", "name", "origin", "background"],
    "Physical_Description":  ["description", "describe", "appearance", "physical", "clothing"],
    "Daily_Routine":         ["routine", "daily", "schedule", "habit", "activity"],
    "Narrative_Experience":  ["narrative", "experience", "story", "event", "personal"],
    "Opinion_Evaluation":    ["opinion", "evaluation", "view", "think", "feel", "perspective"],
    "Information_Reporting": ["information", "reporting", "report", "fact", "detail"],
    "Social_Closing":        ["closing", "goodbye", "farewell", "thanks", "bye", "regards"],
}

def call_llm(model, messages, retries=3):
    for attempt in range(retries):
        try:
            r = ollama.chat(model=model, messages=messages)
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}")
            time.sleep(2)
    return ""

def score_criterion(essay_text, feedback_text, criterion_prompt, retries=3):
    """Call llama3.2 up to retries times to get a valid 1-5 integer."""
    prompt = criterion_prompt.format(essay_text=essay_text, feedback_text=feedback_text)
    for attempt in range(retries):
        resp = call_llm("llama3.2", [{"role": "user", "content": prompt}])
        m = re.search(r"\b([1-5])\b", resp)
        if m:
            return int(m.group(1))
        print(f"  [score retry {attempt+1}] got: {repr(resp)}")
    return None

# ── Criterion prompts ─────────────────────────────────────────────────────────

SPECIFICITY_PROMPT = (
    "You are evaluating the specificity of writing feedback.\n"
    "Score this feedback on a scale of 1 to 5:\n"
    "1 = completely generic, could apply to any essay\n"
    "2 = mostly generic with one specific reference\n"
    "3 = some specific references to the essay content\n"
    "4 = mostly specific, clearly references this essay's structure\n"
    "5 = highly specific, precisely identifies structural patterns\n"
    "    in this essay and names exact issues\n\n"
    "Only respond with a single integer between 1 and 5. Nothing else.\n\n"
    "Essay: {essay_text}\n"
    "Feedback: {feedback_text}"
)

ACTIONABILITY_PROMPT = (
    "You are evaluating whether writing feedback is actionable.\n"
    "Score this feedback on a scale of 1 to 5:\n"
    "1 = no clear suggestion for improvement\n"
    "2 = vague suggestion that is hard to act on\n"
    "3 = one clear suggestion but not detailed\n"
    "4 = clear and specific suggestion the student can follow\n"
    "5 = precise, concrete suggestion with clear next step\n\n"
    "Only respond with a single integer between 1 and 5. Nothing else.\n\n"
    "Essay: {essay_text}\n"
    "Feedback: {feedback_text}"
)

STRUCTURAL_FOCUS_PROMPT = (
    "You are evaluating whether feedback focuses on discourse\n"
    "structure rather than grammar or vocabulary.\n"
    "Score this feedback on a scale of 1 to 5:\n"
    "1 = focuses entirely on grammar or vocabulary, no structure\n"
    "2 = mostly grammar or vocabulary, brief mention of structure\n"
    "3 = balanced between structure and surface features\n"
    "4 = mostly about structure and organisation\n"
    "5 = entirely focused on discourse structure and organisation\n\n"
    "Only respond with a single integer between 1 and 5. Nothing else.\n\n"
    "Essay: {essay_text}\n"
    "Feedback: {feedback_text}"
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Reconstruct full essay texts for the 10 A1 essays
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1: Reconstructing essay texts ...")

rag_df     = pd.read_csv(RAG_CSV)
cluster_df = pd.read_csv(CLUSTER_META)

essay_texts = {}
for wid in rag_df["writing_id"]:
    sents = (cluster_df[cluster_df["writing_id"] == wid]
             .sort_values("sentence_index")["sentence"]
             .dropna().astype(str).tolist())
    essay_texts[wid] = " ".join(sents)

print(f"  Reconstructed {len(essay_texts)} essays")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Generate vanilla baseline feedback
# ─────────────────────────────────────────────────────────────────────────────
if BASELINE_CSV.exists():
    print("\nStep 2: Loading existing baseline feedback ...")
    baseline_df = pd.read_csv(BASELINE_CSV)
    print(f"  Loaded {len(baseline_df)} baseline feedbacks")
else:
    print("\nStep 2: Generating baseline feedback ...")
    baseline_rows = []
    for i, row in rag_df.iterrows():
        wid   = row["writing_id"]
        essay = essay_texts[wid]
        print(f"  [{i+1}/10] writing_id={wid}")
        prompt = (
            f"You are given an essay written by an ESL learner and the corresponding writing task.\n\n"
            f"#### Writing Task: \"Write a short essay in English on the given topic.\"\n"
            f"### Feedback Task: Provide formative feedback on the structure and organisation of this essay only. "
            f"Do not comment on grammar or vocabulary.\n"
            f"#### Student Essay: \"{essay}\""
        )
        feedback = call_llm("qwen2.5:3b", [{"role": "user", "content": prompt}])
        baseline_rows.append({"writing_id": wid, "essay_text": essay, "baseline_feedback": feedback})

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(BASELINE_CSV, index=False)
    print(f"  Saved {BASELINE_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate feedback quality directly using llama3.2
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Evaluating feedback quality with llama3.2 ...")

result_rows = []
for i, (rag_row, bl_row) in enumerate(zip(rag_df.itertuples(), baseline_df.itertuples())):
    wid   = rag_row.writing_id
    essay = essay_texts[wid]
    print(f"  [{i+1}/10] writing_id={wid}")

    # RAG feedback scores
    rag_fb = rag_row.feedback
    rag_spec   = score_criterion(essay, rag_fb, SPECIFICITY_PROMPT)
    rag_act    = score_criterion(essay, rag_fb, ACTIONABILITY_PROMPT)
    rag_struct = score_criterion(essay, rag_fb, STRUCTURAL_FOCUS_PROMPT)
    print(f"    RAG:      specificity={rag_spec}  actionability={rag_act}  structural_focus={rag_struct}")

    # Baseline feedback scores
    bl_fb  = bl_row.baseline_feedback
    bl_spec   = score_criterion(essay, bl_fb, SPECIFICITY_PROMPT)
    bl_act    = score_criterion(essay, bl_fb, ACTIONABILITY_PROMPT)
    bl_struct = score_criterion(essay, bl_fb, STRUCTURAL_FOCUS_PROMPT)
    print(f"    Baseline: specificity={bl_spec}  actionability={bl_act}  structural_focus={bl_struct}")

    result_rows.append({
        "writing_id":               wid,
        "rag_specificity":          rag_spec,
        "rag_actionability":        rag_act,
        "rag_structural_focus":     rag_struct,
        "baseline_specificity":     bl_spec,
        "baseline_actionability":   bl_act,
        "baseline_structural_focus":bl_struct,
    })

compare_df = pd.DataFrame(result_rows)
compare_df.to_csv(COMPARE_CSV, index=False)
print(f"\n  Saved {COMPARE_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Print comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Comparison table ...")

def winner(rag_val, bl_val):
    if rag_val > bl_val:   return "RAG"
    if bl_val  > rag_val:  return "Base"
    return "Tie"

rag_spec_m   = compare_df["rag_specificity"].mean()
rag_act_m    = compare_df["rag_actionability"].mean()
rag_struct_m = compare_df["rag_structural_focus"].mean()
rag_overall  = (rag_spec_m + rag_act_m + rag_struct_m) / 3

bl_spec_m    = compare_df["baseline_specificity"].mean()
bl_act_m     = compare_df["baseline_actionability"].mean()
bl_struct_m  = compare_df["baseline_structural_focus"].mean()
bl_overall   = (bl_spec_m + bl_act_m + bl_struct_m) / 3

print("\n" + "=" * 60)
print(f"{'Criterion':<22} | {'RAG System':>10} | {'Baseline':>8} | {'Winner':>6}")
print("-" * 60)
print(f"{'Specificity (mean)':<22} | {rag_spec_m:>10.2f} | {bl_spec_m:>8.2f} | {winner(rag_spec_m, bl_spec_m):>6}")
print(f"{'Actionability (mean)':<22} | {rag_act_m:>10.2f} | {bl_act_m:>8.2f} | {winner(rag_act_m, bl_act_m):>6}")
print(f"{'Structural Focus':<22} | {rag_struct_m:>10.2f} | {bl_struct_m:>8.2f} | {winner(rag_struct_m, bl_struct_m):>6}")
print("-" * 60)
print(f"{'Overall mean':<22} | {rag_overall:>10.2f} | {bl_overall:>8.2f} | {winner(rag_overall, bl_overall):>6}")
print("=" * 60)
