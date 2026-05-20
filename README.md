# Discourse Move Structures in ESL Learner Writing
## Structural Pattern Analysis and Feedback Generation with Discourse Moves

## Overview

This repository contains the code, analysis scripts, and selected outputs
for a course project on discourse move structure in ESL learner writing
using the EFCAMDAT corpus.

The project has two main components:

1. **Stage 1: Discourse move identification and structural analysis**
   - sentence embeddings
   - UMAP + HDBSCAN clustering
   - qualitative consolidation of raw clusters into eight discourse move
     categories
   - CEFR-level move distributions and sequence-level structural features

2. **Stage 2: Structure-focused feedback generation**
   - zero-shot baseline
   - one-shot prompting with held-out A1 examples and ESL-teacher feedback
   - retrieval-augmented prompting using discourse move sequences as a
     structural retrieval key
   - task-aware evaluation with Claude Sonnet

The repository reflects the **final project pipeline**, including the
revised contrastive RAG prompt, task-aware evaluation, qualitative error
analysis, and sequence-level feature analysis.

---

## At a Glance

- **Corpus:** EFCAMDAT
- **Levels covered:** A1-C2
- **RQ1 output:** 8 discourse move categories from 44 non-noise clusters
- **RQ2 setup:** zero-shot vs one-shot vs retrieval-augmented feedback
- **Evaluation set:** 60 held-out A1 essays
- **Judge model:** Claude Sonnet (task-aware evaluation)
- **Generation model:** Qwen2.5-7B via Ollama

---

## Key Outputs

The most important final outputs are:

- `clustering_meta.csv` - sentence-level metadata with cluster assignments
- `clustering_labelled.csv` - sentence-level discourse move labels
- `essay_sequences.csv` - essay-level move sequences
- `feedback_n60.csv` - generated feedback for the 60-essay RQ2 evaluation set
- `scores_n60_claude_tasktopic.csv` - task-aware judge scores
- `csv/results_table_n60_tasktopic.csv` - descriptive statistics for RQ2
- `csv/wilcoxon_n60_tasktopic.csv` - pairwise Wilcoxon results
- `csv/error_analysis_n60_tasktopic.csv` - error-analysis cases for weaker RAGF outputs
- `csv/error_analysis_n60_summary.csv` - summary of primary RAGF error types
- `csv/sequence_features_summary.csv` - CEFR-level sequence feature summary

Tracked figures are stored in `figs/`, and tracked result tables are stored
in `csv/`.

Because `*.csv` is ignored by default in `.gitignore`, several final result
files are explicitly tracked in this repository.

---

## Installation

```bash
git clone https://github.com/erinju7/Modelling-Discourse-Move-Sequences.git
cd Modelling-Discourse-Move-Sequences
pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

For local feedback generation, install [Ollama](https://ollama.com) and pull
the model used in the final pipeline:

```bash
ollama pull qwen2.5:7b
```

If you only want to inspect the final outputs, you do not need to rerun the
full pipeline; the main result figures and CSV summaries are already tracked
in the repository.

---

## Data

The project uses the **EF Cambridge Open Language Database (EFCAMDAT)**.
Access requires registration:

[https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html](https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html)

After download, place `EFCAMDAT_Database.xml` in the project root before
running the extraction scripts.

---

## Pipeline

### RQ1: Discourse move identification

```bash
python3 src/01_extract_essays.py
python3 src/02_build_datasets.py
python3 src/03_embed_sentences.py
python3 src/04_sample_for_clustering.py
python3 src/05_cluster_discourse_moves.py
python3 src/06_label_discourse_moves.py
python3 src/visualise_umap.py
python3 src/silhouette_eval.py
python3 src/analyse_sequence_features.py
```

Optional inspection scripts:

```bash
python3 src/inspect_clusters.py
python3 src/analyse_a1_sequences.py
python3 src/analyse_c2_sequences.py
python3 src/build_a1_knowledge_graph.py
python3 src/build_c2_knowledge_graph.py
```

### RQ2: Feedback generation and evaluation

```bash
N_EVAL=60 OUT_CSV=eval_essays_n60.csv python3 src/07_select_eval_essays.py
EVAL_CSV=eval_essays_n60.csv OUT_CSV=feedback_n60.csv python3 src/generate_feedback.py
FEEDBACK_CSV=feedback_n60.csv OUT_CLAUDE=scores_n60_claude_tasktopic.csv \
N_RUNS=5 python3 src/09_evaluate_rq2_tasktopic.py
SCORES_CSV=scores_n60_claude_tasktopic.csv \
OUT_TABLE=csv/results_table_n60_tasktopic.csv \
OUT_STATS=csv/wilcoxon_n60_tasktopic.csv \
OUT_PLOT=/tmp/results_n60_tasktopic.png \
python3 src/10_analyse_rq2.py
FEEDBACK_CSV=feedback_n60.csv SCORES_CSV=scores_n60_claude_tasktopic.csv \
OUT_CSV=csv/error_analysis_n60_tasktopic.csv \
python3 src/error_analysis_tasktopic.py
```

---

## Final Experimental Conditions

The final RQ2 evaluation compares three conditions:

- **ZSF: Zero-shot feedback**
  - task topic + learner essay

- **OSF: One-shot feedback**
  - task topic + learner essay + one held-out A1 example with ESL-teacher
    feedback, matched by broad task type

- **RAGF: Retrieval-augmented feedback**
  - task topic + learner essay + learner move sequence + retrieved C2 move
    sequence + short structural summary
  - the retrieved C2 sequence is used as a **contrastive structural hint**,
    not as a full exemplar to imitate

---

## Final Results

### RQ1 Summary

- HDBSCAN produced **44 non-noise clusters**
- **569 / 3275 sentences (17.4%)** were assigned to noise and labelled as
  `Other`
- the remaining **82.6%** were consolidated into **8 core discourse move
  categories**
- silhouette score in 5D UMAP space: **0.619**

Overall move distribution:

- `Information_Reporting`: 17.3%
- `Opinion_Evaluation`: 15.2%
- `Social_Closing`: 10.8%
- `Narrative_Experience`: 10.5%
- `Daily_Routine`: 9.8%
- `Self_Introduction`: 8.9%
- `Physical_Description`: 7.1%
- `Social_Opening`: 3.0%
- `Other`: 17.4%

Sequence-level feature summary:

- A1 had the shortest mean sequence length: **3.01**
- move diversity was higher at A2 (**2.31**) and C1 (**2.29**) than at A1
  (**1.71**)
- repetition rate was lowest at A1 (**0.31**) and higher at B1 (**0.57**) and
  C2 (**0.52**)

These patterns should be interpreted cautiously, since EFCAMDAT prompts vary
across CEFR levels and therefore reflect both proficiency and task design.

### RQ2 Summary

Task-aware Claude evaluation on 60 held-out A1 essays:

| Dimension | RAGF | ZSF | OSF |
|---|---:|---:|---:|
| Specificity | 3.03 ± 0.48 | 2.85 ± 0.59 | 3.33 ± 0.66 |
| Helpfulness | 2.67 ± 0.57 | 2.77 ± 0.60 | 3.11 ± 0.67 |
| Validity | 2.49 ± 0.53 | 2.84 ± 0.68 | 3.21 ± 0.75 |

Main significance results after Bonferroni correction:

- **RAGF vs OSF:** OSF significantly outperformed RAGF on specificity
  (`p_corr = 0.0234`, `r_rb = -0.438`), helpfulness
  (`p_corr = 0.0001`, `r_rb = -0.714`), and validity
  (`p_corr < 0.0001`, `r_rb = -0.867`).
- **ZSF vs OSF:** OSF significantly outperformed ZSF on specificity
  (`p_corr = 0.0002`, `r_rb = -0.568`), helpfulness
  (`p_corr = 0.0189`, `r_rb = -0.474`), and validity
  (`p_corr = 0.0092`, `r_rb = -0.450`).
- **RAGF vs ZSF:** RAGF did not significantly differ from ZSF on
  specificity or helpfulness, but scored significantly lower on validity
  (`p_corr = 0.0087`, `r_rb = -0.600`).

Error analysis of the 38 cases where RAGF scored lower than OSF on
helpfulness or validity identified these primary error types:

- retrieval/task mismatch: **25** cases (65.8%)
- weak structural diagnosis: **9** cases (23.7%)
- too abstract/generic: **1** case (2.6%)
- other: **3** cases (7.9%)

Overall, the final results support one-shot prompting as the strongest
condition in this setup, while suggesting that discourse-move-based retrieval
is informative but fragile: it makes structure explicit, but does not reliably
translate retrieved C2 structural references into task-appropriate A1 feedback.

---

## Notes on Reproducibility

- The final repository contains both scripts and selected generated outputs.
- Some exploratory scripts and older intermediate artifacts remain for
  transparency.
- The report should use the current final result files listed above as the
  canonical source for reported numbers.

---

## License

This project is licensed under the MIT License.
