import os
import re
import time
import anthropic
import pandas as pd
from pathlib import Path

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

BASE          = Path("/Users/macbook/Desktop/AIED cw2")
RAG_CSV       = BASE / "rag_evaluation_results.csv"
BASELINE_CSV  = BASE / "baseline_feedback.csv"
ONESHOT_CSV   = BASE / "oneshot_feedback.csv"
CLUSTER_META  = BASE / "clustering_meta.csv"
OUT_CSV       = BASE / "feedback_quality_results_final.csv"
COMPARE_CSV   = BASE / "three_way_comparison.csv"

N_RUNS = 3

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Dimension prompts ─────────────────────────────────────────────────────────

SPECIFICITY_PROMPT = """\
Evaluate the specificity of this ESL writing feedback.
Does the feedback address specific issues in this
particular essay, or is it generic advice?

Score 1-5:
1 = completely generic
2 = mostly generic
3 = somewhat specific
4 = mostly specific
5 = highly specific to this essay

Reply with only a single digit 1-5.

Essay: {essay_text}
Feedback: {feedback_text}"""

HELPFULNESS_PROMPT = """\
Evaluate whether this ESL writing feedback is helpful
for the student to improve their writing structure.

Score 1-5:
1 = not helpful
2 = slightly helpful
3 = moderately helpful
4 = helpful
5 = very helpful

Reply with only a single digit 1-5.

Essay: {essay_text}
Feedback: {feedback_text}"""

VALIDITY_PROMPT = """\
Evaluate whether this ESL writing feedback is accurate
and correctly identifies issues in the essay.

Score 1-5:
1 = major errors or misidentification
2 = several inaccuracies
3 = mostly accurate
4 = accurate
5 = fully accurate

Reply with only a single digit 1-5.

Essay: {essay_text}
Feedback: {feedback_text}"""

DIMENSIONS = [
    ("specificity",  SPECIFICITY_PROMPT),
    ("helpfulness",  HELPFULNESS_PROMPT),
    ("validity",     VALIDITY_PROMPT),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_score(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            print(f"  [Claude says]: {text[:80]}")
            match = re.search(r'\b([1-5])\b', text)
            if match:
                time.sleep(1)
                return int(match.group(1))
        except Exception as e:
            print(f"  [error] {e}")
            time.sleep(2)
    return None

def score_ensemble(essay_text, feedback_text, prompt_template):
    prompt = prompt_template.format(essay_text=essay_text, feedback_text=feedback_text)
    scores = [get_score(prompt) for _ in range(N_RUNS)]
    return [s for s in scores if s is not None]

def winner(*means):
    best_val = max(means)
    labels = ["RAG", "One-shot", "Zero-shot"]
    winners = [labels[i] for i, v in enumerate(means) if v == best_val]
    return "/".join(winners)

# ── Load data ─────────────────────────────────────────────────────────────────

rag_df      = pd.read_csv(RAG_CSV)
baseline_df = pd.read_csv(BASELINE_CSV)
oneshot_df  = pd.read_csv(ONESHOT_CSV)
cluster_df  = pd.read_csv(CLUSTER_META)

# Reconstruct essay texts from cluster_meta
essay_texts = {}
for wid in rag_df["writing_id"]:
    sents = (cluster_df[cluster_df["writing_id"] == wid]
             .sort_values("sentence_index")["sentence"]
             .dropna().astype(str).tolist())
    essay_texts[wid] = " ".join(sents)

# Index one-shot feedback by writing_id
oneshot_map = oneshot_df.set_index("writing_id")["oneshot_feedback"].to_dict()

# ── Evaluate all three conditions ─────────────────────────────────────────────

print(f"Evaluating all three conditions (Claude claude-sonnet-4-5, {N_RUNS} runs per dimension) ...\n")

rows = []
for i, (rag_row, bl_row) in enumerate(zip(rag_df.itertuples(), baseline_df.itertuples())):
    wid     = rag_row.writing_id
    essay   = essay_texts[wid]
    rag_fb  = rag_row.feedback
    bl_fb   = bl_row.baseline_feedback
    os_fb   = oneshot_map.get(wid, "")

    print(f"[{i+1}/10] writing_id={wid}")
    result = {"writing_id": wid}

    for dname, prompt_tmpl in DIMENSIONS:
        rag_scores = score_ensemble(essay, rag_fb,  prompt_tmpl)
        bl_scores  = score_ensemble(essay, bl_fb,   prompt_tmpl)
        os_scores  = score_ensemble(essay, os_fb,   prompt_tmpl)

        rs = sum(rag_scores) / len(rag_scores) if rag_scores else float("nan")
        bs = sum(bl_scores)  / len(bl_scores)  if bl_scores  else float("nan")
        os = sum(os_scores)  / len(os_scores)  if os_scores  else float("nan")

        result[f"rag_{dname}"]     = rs
        result[f"baseline_{dname}"] = bs
        result[f"oneshot_{dname}"]  = os

        print(f"  {dname:<20} RAG={rag_scores}→{rs:.2f}  "
              f"Zero-shot={bl_scores}→{bs:.2f}  "
              f"One-shot={os_scores}→{os:.2f}")

    rows.append(result)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

# ── Three-way comparison table ────────────────────────────────────────────────

print("\n" + "=" * 80)
print(f"{'Dimension':<13} | {'RAG mean±std':>14} | {'One-shot ±std':>14} | {'Zero-shot ±std':>15} | {'Best':>10}")
print("-" * 80)

compare_rows = []
totals = {"rag": [], "oneshot": [], "zeroshot": []}

for dname, _ in DIMENSIONS:
    r_vals = out_df[f"rag_{dname}"]
    o_vals = out_df[f"oneshot_{dname}"]
    z_vals = out_df[f"baseline_{dname}"]

    r, r_std = r_vals.mean(), r_vals.std()
    o, o_std = o_vals.mean(), o_vals.std()
    z, z_std = z_vals.mean(), z_vals.std()

    totals["rag"].append(r)
    totals["oneshot"].append(o)
    totals["zeroshot"].append(z)

    best = winner(r, o, z)
    label = dname.title()
    print(f"{label:<13} | {r:>5.2f} ± {r_std:<5.2f}  | {o:>5.2f} ± {o_std:<5.2f}  | {z:>5.2f} ± {z_std:<6.2f} | {best:>10}")

    compare_rows.append({
        "dimension":     dname,
        "rag_mean":      round(r, 3),  "rag_std":      round(r_std, 3),
        "oneshot_mean":  round(o, 3),  "oneshot_std":  round(o_std, 3),
        "zeroshot_mean": round(z, 3),  "zeroshot_std": round(z_std, 3),
        "best":          best,
    })

print("-" * 80)
rag_overall      = sum(totals["rag"])      / len(totals["rag"])
oneshot_overall  = sum(totals["oneshot"])  / len(totals["oneshot"])
zeroshot_overall = sum(totals["zeroshot"]) / len(totals["zeroshot"])
best_overall     = winner(rag_overall, oneshot_overall, zeroshot_overall)

print(f"{'Overall':<13} | {rag_overall:>14.2f} | {oneshot_overall:>14.2f} | {zeroshot_overall:>15.2f} | {best_overall:>10}")
print("=" * 80)

compare_rows.append({
    "dimension":     "overall",
    "rag_mean":      round(rag_overall, 3),      "rag_std":      "",
    "oneshot_mean":  round(oneshot_overall, 3),  "oneshot_std":  "",
    "zeroshot_mean": round(zeroshot_overall, 3), "zeroshot_std": "",
    "best":          best_overall,
})

pd.DataFrame(compare_rows).to_csv(COMPARE_CSV, index=False)
print(f"\nSaved {COMPARE_CSV}")
