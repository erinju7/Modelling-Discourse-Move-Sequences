"""
claude_eval_n30.py
Evaluate 30 essays × 3 conditions using Claude as judge.
Run locally after generate_feedback_n30.py has completed.

Requires: ANTHROPIC_API_KEY environment variable
"""

import os, re, time, anthropic, pandas as pd
from pathlib import Path
from scipy import stats

BASE          = Path("/Users/macbook/Desktop/AIED cw2")
BASELINE_CSV  = BASE / "baseline_feedback_n30.csv"
ONESHOT_CSV   = BASE / "oneshot_feedback_n30.csv"
RAG_CSV       = BASE / "rag_results_n30.csv"
CLUSTER_META  = BASE / "clustering_meta.csv"
OUT_CSV       = BASE / "claude_eval_results_n30.csv"
COMPARE_CSV   = BASE / "claude_comparison_n30.csv"

N_RUNS = 3   # ensemble runs per score for stability
MODEL  = "claude-haiku-4-5-20251001"   # cheaper; swap to sonnet if needed

DIMENSIONS = {
    "specificity": """\
Evaluate the specificity of this ESL writing feedback.
Does the feedback address specific issues in this particular essay, or is it generic?

Score 1-5:
1 = completely generic  2 = mostly generic  3 = somewhat specific
4 = mostly specific     5 = highly specific to this essay

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",

    "helpfulness": """\
Evaluate whether this ESL writing feedback is helpful for the student to improve their writing structure.

Score 1-5:
1 = not helpful  2 = slightly helpful  3 = moderately helpful
4 = helpful      5 = very helpful

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",

    "validity": """\
Evaluate whether this ESL writing feedback accurately identifies issues in the essay.

Score 1-5:
1 = major errors  2 = several inaccuracies  3 = mostly accurate
4 = accurate      5 = fully accurate

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",
}

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))

def get_score(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=10,
                messages=[{"role":"user","content":prompt_text}]
            )
            text = resp.content[0].text.strip()
            m = re.search(r'\b([1-5])\b', text)
            if m:
                time.sleep(0.5)
                return int(m.group(1))
        except Exception as e:
            print(f"  [retry {attempt+1}] {e}"); time.sleep(2)
    return None

def score_essay(essay, feedback):
    scores = {}
    for dim, tmpl in DIMENSIONS.items():
        prompt = tmpl.format(essay=essay, feedback=feedback)
        runs = [get_score(prompt) for _ in range(N_RUNS)]
        valid = [s for s in runs if s is not None]
        scores[dim] = sum(valid)/len(valid) if valid else None
    return scores

# ── Load data ─────────────────────────────────────────────────────────────────
cluster_df  = pd.read_csv(CLUSTER_META)
baseline_df = pd.read_csv(BASELINE_CSV)
oneshot_df  = pd.read_csv(ONESHOT_CSV)
rag_df      = pd.read_csv(RAG_CSV)

essay_texts = (cluster_df.sort_values(["writing_id","sentence_index"])
               .groupby("writing_id")["sentence"]
               .apply(lambda s: " ".join(s.dropna().astype(str))).to_dict())

baseline_map = baseline_df.set_index("writing_id")["baseline_feedback"].to_dict()
oneshot_map  = oneshot_df.set_index("writing_id")["oneshot_feedback"].to_dict()
rag_map      = rag_df.set_index("writing_id")["feedback"].to_dict()

all_ids = sorted(set(rag_map) & set(baseline_map) & set(oneshot_map))
print(f"Evaluating {len(all_ids)} essays with Claude ({MODEL}) ...\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
rows = []
for i, wid in enumerate(all_ids):
    essay = essay_texts.get(wid, "")
    print(f"[{i+1}/{len(all_ids)}] writing_id={wid}")

    rag_s = score_essay(essay, rag_map[wid])
    bl_s  = score_essay(essay, baseline_map[wid])
    os_s  = score_essay(essay, oneshot_map[wid])

    row = {"writing_id": wid}
    for dim in DIMENSIONS:
        row[f"rag_{dim}"]      = rag_s[dim]
        row[f"baseline_{dim}"] = bl_s[dim]
        row[f"oneshot_{dim}"]  = os_s[dim]
        print(f"  {dim:<14} RAG={rag_s[dim]}  Baseline={bl_s[dim]}  One-shot={os_s[dim]}")
    rows.append(row)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

# ── Summary + Wilcoxon tests ──────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Dimension':<14} | {'RAG':>8} | {'One-shot':>8} | {'Baseline':>8} | Best | RAG vs BL p | RAG vs OS p")
print("-"*75)

compare_rows = []
for dim in DIMENSIONS:
    r_vals = out_df[f"rag_{dim}"].dropna()
    o_vals = out_df[f"oneshot_{dim}"].dropna()
    b_vals = out_df[f"baseline_{dim}"].dropna()

    r, o, b = r_vals.mean(), o_vals.mean(), b_vals.mean()
    best = ["RAG","One-shot","Baseline"][[r,o,b].index(max(r,o,b))]

    _, p_rag_bl = stats.wilcoxon(r_vals, b_vals, zero_method="wilcox", alternative="two-sided")
    _, p_rag_os = stats.wilcoxon(r_vals, o_vals, zero_method="wilcox", alternative="two-sided")

    sig_bl = "*" if p_rag_bl < 0.05 else ""
    sig_os = "*" if p_rag_os < 0.05 else ""

    print(f"{dim:<14} | {r:>8.3f} | {o:>8.3f} | {b:>8.3f} | {best:<8} | "
          f"{p_rag_bl:.3f}{sig_bl:<2} | {p_rag_os:.3f}{sig_os}")

    compare_rows.append({
        "dimension": dim,
        "rag_mean": round(r,3), "oneshot_mean": round(o,3), "baseline_mean": round(b,3),
        "best": best,
        "p_rag_vs_baseline": round(p_rag_bl,4), "p_rag_vs_oneshot": round(p_rag_os,4),
        "sig_rag_vs_baseline": p_rag_bl < 0.05, "sig_rag_vs_oneshot": p_rag_os < 0.05,
    })

print("="*75)
pd.DataFrame(compare_rows).to_csv(COMPARE_CSV, index=False)
print(f"Saved {COMPARE_CSV}")
