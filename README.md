# Discourse Move Structures in ESL Learner Writing
## Structural Pattern Analysis and Feedback Generation with Discourse Moves

## Overview

This repository contains the code and analysis pipeline for a dissertation
project on discourse move structure in ESL learner writing using the
EFCAMDAT corpus.

The project has two main components:

1. **RQ1: Discourse move identification and structural analysis**
   - sentence embeddings
   - UMAP + HDBSCAN clustering
   - qualitative consolidation of raw clusters into eight discourse move
     categories
   - CEFR-level move distributions and sequence-level structural features

2. **RQ2: Structure-focused feedback generation**
   - zero-shot baseline
   - one-shot prompting with held-out A1 examples and ESL-teacher feedback
   - retrieval-augmented prompting using discourse move sequences as a
     structural retrieval key
   - task-aware evaluation with Claude Sonnet

The repository now reflects the **final dissertation pipeline**, including
the revised contrastive RAG prompt, task-aware evaluation, error analysis,
and sequence-level feature analysis.

---

## Research Questions

- **RQ1:** Can unsupervised clustering of sentence embeddings identify a
  coherent set of discourse move categories in ESL learner writing?

- **RQ2:** How do discourse move structures differ across CEFR levels, and
  can these structural patterns support feedback generation for A1 learners?

- **RQ3:** Does retrieval-augmented prompting using discourse move sequences
  improve structure-focused feedback compared with zero-shot and one-shot
  baselines?

---

## Repository Structure

```text
.
├── src/
│   ├── 01_extract_essays.py
│   ├── 02_build_datasets.py
│   ├── 03_embed_sentences.py
│   ├── 04_sample_for_clustering.py
│   ├── 05_cluster_discourse_moves.py
│   ├── 06_label_discourse_moves.py
│   ├── 07_select_eval_essays.py
│   ├── 08_generate_baseline_feedback.py
│   ├── 09_oneshot_feedback_generation.py
│   ├── 09_evaluate_rq2.py
│   ├── 09_evaluate_rq2_tasktopic.py
│   ├── 10_analyse_rq2.py
│   ├── 10_evaluate_all_conditions.py
│   ├── analyse_a1_sequences.py
│   ├── analyse_c2_sequences.py
│   ├── analyse_sequence_features.py
│   ├── build_a1_knowledge_graph.py
│   ├── build_c2_knowledge_graph.py
│   ├── claude_eval_n30.py
│   ├── error_analysis_tasktopic.py
│   ├── generate_feedback_n30.py
│   ├── inspect_clusters.py
│   ├── sensitivity_analysis.py
│   ├── silhouette_eval.py
│   └── visualise_umap.py
├── figs/
├── csv/
├── outputs/
├── README.md
└── requirements.txt
```

---

## Main Data Files

These files are generated during the pipeline and are used in the final
analysis:

- `clustering_meta.csv` - sentence-level metadata with cluster assignments
- `clustering_labelled.csv` - sentence-level discourse move labels
- `essay_sequences.csv` - essay-level move sequences
- `csv/feedback_n30.csv` - generated feedback for the 30-essay RQ2 evaluation set
- `csv/scores_n30_claude_tasktopic.csv` - task-aware judge scores
- `csv/results_table_n30_tasktopic.csv` - descriptive statistics for RQ2
- `csv/wilcoxon_rq2_tasktopic.csv` - pairwise Wilcoxon results
- `csv/sensitivity_results_tasktopic.csv` - retrieval-quality sensitivity analysis
- `csv/error_analysis_tasktopic.csv` - manually coded error analysis for weaker RAG cases
- `csv/sequence_features_summary.csv` - CEFR-level sequence feature summary

Final tracked figures are stored in `figs/`, and final tracked result tables
are stored in `csv/`.

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
python3 src/07_select_eval_essays.py
OUT_CSV=csv/feedback_n30.csv python3 src/generate_feedback_n30.py
python3 src/09_evaluate_rq2_tasktopic.py
SCORES_CSV=csv/scores_n30_claude_tasktopic.csv \
OUT_TABLE=csv/results_table_n30_tasktopic.csv \
OUT_PLOT=figs/results_n30_tasktopic.png \
python3 src/10_analyse_rq2.py
SCORES_CSV=csv/scores_n30_claude_tasktopic.csv \
OUT_CSV=csv/sensitivity_results_tasktopic.csv \
python3 src/sensitivity_analysis.py
python3 src/error_analysis_tasktopic.py
```

---

## Final Experimental Conditions

The final RQ2 evaluation compares three conditions:

- **C1 Zero-shot**
  - task topic + learner essay

- **C2 One-shot**
  - task topic + learner essay + one held-out A1 example with ESL-teacher
    feedback, matched by broad task type

- **C3 Retrieval-Augmented Feedback**
  - task topic + learner essay + learner move sequence + retrieved C2 move
    sequence + short structural summary
  - the retrieved C2 sequence is used as a **contrastive structural hint**,
    not as a full exemplar to imitate

---

## Final RQ1 Summary

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

---

## Final RQ2 Summary

Task-aware Claude evaluation on 30 held-out A1 essays:

| Dimension | RAG | Zero-shot | One-shot |
|---|---:|---:|---:|
| Specificity | 3.07 ± 0.37 | 3.01 ± 0.45 | 3.53 ± 0.60 |
| Helpfulness | 2.79 ± 0.45 | 3.03 ± 0.60 | 3.16 ± 0.70 |
| Validity | 2.56 ± 0.53 | 2.91 ± 0.43 | 3.27 ± 0.76 |

Main significance result after Bonferroni correction:

- **Validity: RAG vs One-shot**
  - `p_corr = 0.0009`
  - `r_rb = -0.826`

Sensitivity analysis:

- when restricted to retrievals with `edit_distance <= 3`, the RAG vs
  one-shot validity difference no longer remained significant after
  correction (`p_corr = 0.0555`), although the mean gap persisted

Error analysis of the 19 cases where RAG scored lower than one-shot on
helpfulness or validity showed these primary error types:

- task mismatch: **7** cases (36.8%)
- pedagogically weak contrast: **5** cases (26.3%)
- generic feedback: **4** cases (21.1%)
- inaccurate move labelling: **3** cases (15.8%)

Overall, the final results support one-shot prompting as the strongest
condition in this setup, while suggesting that discourse-move-based retrieval
is informative but fragile.

---

## Notes on Reproducibility

- The final repository contains both scripts and selected generated outputs.
- Some exploratory scripts and older intermediate artifacts remain for
  transparency.
- The dissertation write-up should use the current final result files listed
  above as the canonical source for reported numbers.

---

## Citation

```bibtex
@misc{ju2026discourse,
  title  = {Discourse Move Structures in ESL Learner Writing:
             Structural Pattern Analysis and Feedback Generation with Discourse Moves},
  author = {Ju, Hanyu},
  year   = {2026},
  note   = {Dissertation project}
}
```

---

## License

This project is licensed under the MIT License.
