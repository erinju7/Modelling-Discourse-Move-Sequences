"""
error_analysis_tasktopic.py

Build a manual error-analysis sheet for the final task-aware RQ2 run.
The goal is not to "score" errors automatically, but to surface the
cases where RAG underperforms and provide lightweight heuristic flags
that make manual coding faster.

Outputs:
    error_analysis_tasktopic.csv
"""

from pathlib import Path
import os
import re

import pandas as pd

BASE = Path("/Users/macbook/Desktop/AIED cw2")
FEEDBACK_CSV = BASE / os.environ.get("FEEDBACK_CSV", "feedback_n30.csv")
SCORES_CSV = BASE / os.environ.get("SCORES_CSV", "scores_n30_claude_tasktopic.csv")
OUT_CSV = BASE / os.environ.get("OUT_CSV", "error_analysis_tasktopic.csv")


def norm_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def split_moves(move_sequence):
    return [m.strip() for m in str(move_sequence).split("->") if m.strip()]


def detect_generic_feedback(text):
    txt = norm_text(text).lower()
    generic_markers = [
        "keep up the good work",
        "great job",
        "nice work",
        "good start",
    ]
    return any(marker in txt for marker in generic_markers)


def detect_move_label_claim(text):
    txt = norm_text(text)
    return bool(re.search(r"\b[A-Z][a-z]+_[A-Z][a-z]+\b", txt))


def detect_concrete_example(text):
    txt = norm_text(text)
    return '"' in txt or "for example" in txt.lower() or "like " in txt.lower()


def heuristic_flags(row):
    flags = []
    rag = norm_text(row["feedback_rag"])
    rag_lower = rag.lower()
    move_count = len(split_moves(row["move_sequence"]))
    unique_moves = len(set(split_moves(row["move_sequence"])))
    repetition_count = move_count - unique_moves

    if repetition_count >= 2 and "repeat" in rag_lower:
        flags.append("repetition-focused")
    if "closing" in rag_lower and "email" in str(row["task_topic"]).lower():
        flags.append("closing-emphasis")
    if not detect_concrete_example(rag):
        flags.append("abstract-advice")
    if detect_generic_feedback(rag):
        flags.append("generic-tone")
    if detect_move_label_claim(rag):
        flags.append("explicit-move-label")
    if row["edit_distance"] >= 4:
        flags.append("weaker-retrieval")
    if row["rag_vs_oneshot_validity"] <= -1:
        flags.append("large-validity-gap")
    if row["rag_vs_oneshot_helpfulness"] <= -1:
        flags.append("large-helpfulness-gap")
    return ", ".join(flags)


feedback_df = pd.read_csv(FEEDBACK_CSV)
scores_df = pd.read_csv(SCORES_CSV)
df = feedback_df.merge(scores_df, on=["writing_id", "topic_id", "task_topic", "task_type"])

df["rag_vs_oneshot_validity"] = df["RAG_validity"] - df["One-shot_validity"]
df["rag_vs_oneshot_helpfulness"] = df["RAG_helpfulness"] - df["One-shot_helpfulness"]
df["rag_vs_oneshot_specificity"] = df["RAG_specificity"] - df["One-shot_specificity"]

# Focus on cases where RAG is clearly worse on validity or helpfulness.
focus = df[
    (df["rag_vs_oneshot_validity"] <= -1)
    | (df["rag_vs_oneshot_helpfulness"] <= -1)
].copy()

focus["heuristic_flags"] = focus.apply(heuristic_flags, axis=1)
focus["error_type_primary"] = ""
focus["error_type_secondary"] = ""
focus["notes"] = ""

out_cols = [
    "writing_id",
    "task_topic",
    "task_type",
    "c2_task_topic",
    "edit_distance",
    "move_sequence",
    "c2_sequence",
    "feedback_rag",
    "feedback_oneshot",
    "RAG_specificity",
    "RAG_helpfulness",
    "RAG_validity",
    "One-shot_specificity",
    "One-shot_helpfulness",
    "One-shot_validity",
    "rag_vs_oneshot_specificity",
    "rag_vs_oneshot_helpfulness",
    "rag_vs_oneshot_validity",
    "heuristic_flags",
    "error_type_primary",
    "error_type_secondary",
    "notes",
]

focus = focus[out_cols].sort_values(
    ["rag_vs_oneshot_validity", "rag_vs_oneshot_helpfulness", "edit_distance"]
)
focus.to_csv(OUT_CSV, index=False)

print(f"Saved {OUT_CSV.name} with {len(focus)} candidate cases")
print("\nSuggested coding scheme:")
print("- wrong move label")
print("- task mismatch")
print("- too abstract")
print("- too advanced")
print("- generic feedback")
print("- weak structural diagnosis")
print("- other")
