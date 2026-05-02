"""
09_evaluate_rq2_tasktopic.py
Evaluate generated feedback using Claude Sonnet as judge, with task topic
included in the judge prompt.

This is intended as a robustness check for the original essay-only judge:
scores_n30_claude.csv.

Output: scores_n30_claude_tasktopic.csv
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd

BASE = Path("/Users/macbook/Desktop/AIED cw2")
FEEDBACK_CSV = BASE / os.environ.get("FEEDBACK_CSV", "feedback_n30.csv")
OUT_CLAUDE = BASE / os.environ.get(
    "OUT_CLAUDE", "scores_n30_claude_tasktopic.csv"
)

N_RUNS = int(os.environ.get("N_RUNS", "5"))
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "0"))

CONDITIONS = {
    "RAG": "feedback_rag",
    "Baseline": "feedback_baseline",
    "One-shot": "feedback_oneshot",
}
DIMENSIONS = ["specificity", "helpfulness", "validity"]

A1_TASK_TOPICS = {
    1: "Introducing yourself by email",
    2: "Taking inventory in the office",
    3: "Writing an online profile",
    4: "Describing your family in an email",
    5: "Updating your online profile",
    6: "Signing up for a dating website",
    7: "Writing labels for a clothing store",
    8: "Making a dinner party menu",
}

JUDGE_PROMPTS = {
    "specificity": """\
Evaluate the specificity of this ESL writing-structure feedback.
Does the feedback address specific issues in this particular essay
and task topic, or is it generic advice?

Context:
- Learner level: A1
- Original task topic: {task_topic}
- Broad task type: {task_type}
- Expected feedback focus: discourse move structure and organisation, not grammar, spelling, or vocabulary.

Score 1-5:
1 = completely generic (could apply to any essay)
2 = mostly generic
3 = somewhat specific
4 = mostly specific
5 = highly specific to this essay and task

Reply with only a single digit 1-5.

Essay: {essay}
Feedback: {feedback}""",

    "helpfulness": """\
Evaluate whether this ESL writing feedback is helpful for the student
to improve their writing structure for the original task topic.

Context:
- Learner level: A1
- Original task topic: {task_topic}
- Broad task type: {task_type}
- Expected feedback focus: discourse move structure and organisation, not grammar, spelling, or vocabulary.

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
Evaluate whether this ESL writing feedback is valid for the original
task topic and accurately identifies structural issues in the essay.
Valid feedback should not suggest changes that conflict with the
communicative purpose of the task.

Context:
- Learner level: A1
- Original task topic: {task_topic}
- Broad task type: {task_type}
- Expected feedback focus: discourse move structure and organisation, not grammar, spelling, or vocabulary.

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

claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def extract_score(text: str) -> Optional[int]:
    match = re.search(r"\b([1-5])\b", text.strip())
    return int(match.group(1)) if match else None


def score_claude(prompt: str, retries: int = 3) -> Optional[int]:
    for attempt in range(retries):
        try:
            response = claude_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            score = extract_score(response.content[0].text)
            if score is not None:
                time.sleep(0.3)
                return score
        except Exception as exc:
            print(f"  [Claude retry {attempt + 1}] {exc}")
            time.sleep(2)
    return None


def ensemble_score(row: pd.Series, feedback: str) -> dict:
    result = {}
    task_topic = row.get(
        "task_topic",
        A1_TASK_TOPICS.get(int(row["topic_id"]), f"Topic {row['topic_id']}"),
    )
    task_type = row.get("task_type", row.get("genre", "unknown"))

    for dimension in DIMENSIONS:
        prompt = JUDGE_PROMPTS[dimension].format(
            task_topic=task_topic,
            task_type=task_type,
            essay=row["essay_text"],
            feedback=feedback,
        )
        runs = [score_claude(prompt) for _ in range(N_RUNS)]
        valid = [score for score in runs if score is not None]
        result[dimension] = round(sum(valid) / len(valid), 4) if valid else None
        result[f"{dimension}_raw"] = json.dumps(runs)

    return result


df = pd.read_csv(FEEDBACK_CSV)
if MAX_ROWS > 0:
    df = df.head(MAX_ROWS)
print(
    f"Evaluating {len(df)} essays x {len(CONDITIONS)} conditions x "
    f"{len(DIMENSIONS)} dimensions x {N_RUNS} runs"
)
print(f"Judge: {CLAUDE_MODEL}")
print(f"Output: {OUT_CLAUDE.name}\n")

rows = []
for index, row in df.iterrows():
    writing_id = row["writing_id"]
    task_topic = row.get(
        "task_topic",
        A1_TASK_TOPICS.get(int(row["topic_id"]), f"Topic {row['topic_id']}"),
    )
    print(f"[{index + 1}/{len(df)}] writing_id={writing_id}  task={task_topic}")

    result = {
        "writing_id": writing_id,
        "topic_id": int(row["topic_id"]),
        "task_topic": task_topic,
        "task_type": row.get("task_type", row.get("genre", "unknown")),
    }

    for condition, column in CONDITIONS.items():
        scores = ensemble_score(row, row[column])
        for key, value in scores.items():
            result[f"{condition}_{key}"] = value
        print(
            f"  {condition}: "
            + "  ".join(f"{dimension}={scores[dimension]}" for dimension in DIMENSIONS)
        )

    rows.append(result)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CLAUDE, index=False)
print(f"\nSaved {OUT_CLAUDE}")
