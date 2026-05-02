"""
09_evaluate_rq2.py
Evaluate generated feedback using Claude Sonnet as judge.
30 essays × 3 conditions × 3 dimensions × 5 runs.

Requires:
    export ANTHROPIC_API_KEY=...

Output:
    scores_rq2_claude.csv
"""

import os, re, time, json
from typing import Optional
import anthropic
import numpy as np, pandas as pd
from pathlib import Path

BASE         = Path("/Users/macbook/Desktop/AIED cw2")
FEEDBACK_CSV = BASE / os.environ.get("FEEDBACK_CSV", "feedback_n30.csv")
OUT_CLAUDE   = BASE / os.environ.get("OUT_CLAUDE", "scores_n30_claude.csv")

N_RUNS       = 5
CLAUDE_MODEL = "claude-sonnet-4-6"

CONDITIONS = {"RAG": "feedback_rag", "Baseline": "feedback_baseline", "One-shot": "feedback_oneshot"}
DIMENSIONS = ["specificity", "helpfulness", "validity"]

JUDGE_PROMPTS = {
    "specificity": """\
Evaluate the specificity of this ESL writing feedback.
Does the feedback address specific issues in this particular essay, or is it generic advice?

Score 1-5:
1 = completely generic (could apply to any essay)
2 = mostly generic
3 = somewhat specific
4 = mostly specific
5 = highly specific to this essay

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",

    "helpfulness": """\
Evaluate whether this ESL writing feedback is helpful for the student to improve their writing structure.

Score 1-5:
1 = not helpful
2 = slightly helpful
3 = moderately helpful
4 = helpful
5 = very helpful

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",

    "validity": """\
Evaluate whether this ESL writing feedback is accurate and correctly identifies issues in the essay.

Score 1-5:
1 = major errors or misidentification
2 = several inaccuracies
3 = mostly accurate
4 = accurate
5 = fully accurate

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",
}

# ── API client ────────────────────────────────────────────────────────────────

claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def extract_score(text: str) -> Optional[int]:
    m = re.search(r'\b([1-5])\b', text.strip())
    return int(m.group(1)) if m else None


def score_claude(prompt: str, retries: int = 3) -> Optional[int]:
    for attempt in range(retries):
        try:
            resp = claude_client.messages.create(
                model=CLAUDE_MODEL, max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            score = extract_score(resp.content[0].text)
            if score is not None:
                time.sleep(0.3)
                return score
        except Exception as e:
            print(f"  [Claude retry {attempt+1}] {e}"); time.sleep(2)
    return None


def ensemble_score(score_fn, essay: str, feedback: str) -> dict:
    """Run N_RUNS evaluations per dimension; return mean scores."""
    result = {}
    for dim in DIMENSIONS:
        prompt = JUDGE_PROMPTS[dim].format(essay=essay, feedback=feedback)
        runs   = [score_fn(prompt) for _ in range(N_RUNS)]
        valid  = [s for s in runs if s is not None]
        result[dim]          = round(sum(valid) / len(valid), 4) if valid else None
        result[f"{dim}_raw"] = json.dumps(runs)
    return result

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(FEEDBACK_CSV)
print(f"Evaluating {len(df)} essays × {len(CONDITIONS)} conditions × {len(DIMENSIONS)} dimensions × {N_RUNS} runs\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Judge: Claude Sonnet")
print(f"{'='*60}\n")

rows = []
for i, row in df.iterrows():
    wid   = row["writing_id"]
    essay = row["essay_text"]
    print(f"[{i+1}/{len(df)}] writing_id={wid}")

    result = {"writing_id": wid}
    for cond, col in CONDITIONS.items():
        feedback = row[col]
        scores   = ensemble_score(score_claude, essay, feedback)
        for key, val in scores.items():
            result[f"{cond}_{key}"] = val
        print(f"  {cond}: " + "  ".join(f"{d}={scores[d]}" for d in DIMENSIONS))

    rows.append(result)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CLAUDE, index=False)
print(f"\nSaved {OUT_CLAUDE.name}")
