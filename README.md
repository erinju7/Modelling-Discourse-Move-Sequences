# Discourse Move Structures in ESL Learner Writing
## A Knowledge Graph and RAG-Based Approach to Structural Feedback

---

## Overview

This project investigates discourse move structures in ESL learner writing across CEFR proficiency levels (A1–C2), using a three-stage computational pipeline:

1. **Discourse Move Identification** — Sentences from the EFCAMDAT corpus are encoded with a sentence transformer model (BAAI/bge-large-en-v1.5), reduced with UMAP, and clustered with HDBSCAN to identify 8 recurring discourse move types (e.g. Social_Opening, Information_Reporting, Opinion_Evaluation).

2. **Knowledge Graph Construction** — Essay-level move sequences are extracted for A1 and C2 learners. Bigram transition probabilities are computed and visualised as directed weighted knowledge graphs, revealing structural differences between beginner and advanced writers.

3. **RAG-Based Structural Feedback** — A Retrieval-Augmented Generation system retrieves the most structurally similar C2 essay for a given A1 essay (using edit distance on move sequences) and passes both essays to a local LLM (Qwen2.5:3b via Ollama) to generate formative, structure-focused feedback. The feedback is evaluated across 10 A1 essays using three automated metrics.

**Dataset:** EFCAMDAT (EF Cambridge Open Language Database) — a large-scale longitudinal corpus of EFL learner writing. Full access requires registration (see [Data](#data)).

---

## Research Questions

- **RQ1:** Can unsupervised embedding and clustering methods reliably identify a stable set of discourse moves from ESL learner writing across proficiency levels?

- **RQ2:** How do discourse move transition patterns differ between A1 (beginner) and C2 (advanced) learners, and can these differences be captured in a knowledge graph?

- **RQ3:** Can a RAG system using discourse move sequences as a retrieval key generate accurate and structurally focused formative feedback for A1 learners?

---

## System Architecture

```
EFCAMDAT XML
     │
     ▼
01_data_extraction.py      ← Parse XML, sentence-tokenise essays (A1 + C2)
02_sampling.py             ← Sample 100 essays/level for clustering; extract full C2 corpus
     │
     ▼
03_embedding.py            ← Encode sentences with sentence-transformers
04_clustering.py           ← UMAP dimensionality reduction + HDBSCAN clustering
05_labelling.py            ← Map clusters to 8 discourse move labels; generate heatmap
     │
     ▼
06_knowledge_graph.py      ← Build bigram transition matrix; draw directed knowledge graph
     │
     ▼
07_rag_feedback.py         ← Retrieve most similar C2 essay; generate feedback via Ollama
08_evaluation.py           ← Evaluate feedback across 10 A1 essays (3 automated metrics)
```

---

## Repository Structure

```
.
├── src/
│   ├── 01_data_extraction.py   # Parse EFCAMDAT XML; extract sentences by CEFR level
│   ├── 02_sampling.py          # Build clustering sample (100/level) and full C2 corpus
│   ├── 03_embedding.py         # Sentence embedding with all-MiniLM-L6-v2
│   ├── 04_clustering.py        # Re-embed with BGE-Large; UMAP + HDBSCAN clustering
│   ├── 05_labelling.py         # Assign discourse move labels; generate distribution heatmap
│   ├── 06_knowledge_graph.py   # Bigram transition matrix; directed knowledge graph
│   ├── 07_rag_feedback.py      # RAG pipeline demo: retrieve C2 essay; LLM feedback
│   └── 08_evaluation.py        # Batch evaluate feedback on 10 A1 essays; save results
├── data/
│   └── sample_essays.csv       # 120-sentence sample (20 per CEFR level) for reference
├── outputs/
│   ├── move_distribution_v2.png      # Heatmap of discourse move proportions by CEFR level
│   ├── a1_knowledge_graph.png        # A1 discourse move transition graph
│   ├── c2_knowledge_graph_v3.png     # C2 discourse move transition graph
│   ├── rag_demonstration_v3.txt      # Full RAG pipeline output for one A1 essay
│   └── rag_evaluation_results.csv    # Evaluation metrics for 10 A1 essays
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/erinju7/Modelling-Discourse-Move-Sequences.git
cd Modelling-Discourse-Move-Sequences

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install NLTK tokeniser data (run once)
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Install Ollama and pull the LLM (required for Steps 7 and 8)
# Download Ollama from https://ollama.com
ollama pull qwen2.5:3b
```

---

## Usage

Run scripts in order from the `src/` directory. Scripts read and write files relative to the project root, so run them from the repository root:

```bash
# Step 1 — Extract sentences from EFCAMDAT XML (requires EFCAMDAT_Database.xml)
python3 src/01_data_extraction.py

# Step 2 — Build sampling dataset and full C2 corpus
python3 src/02_sampling.py

# Step 3 — Encode sentences with sentence-transformers
python3 src/03_embedding.py

# Step 4 — Re-embed with BGE-Large, reduce with UMAP, cluster with HDBSCAN
python3 src/04_clustering.py

# Step 5 — Label clusters as discourse moves; generate heatmap
python3 src/05_labelling.py

# Step 6 — Build knowledge graph from move transition probabilities
python3 src/06_knowledge_graph.py

# Step 7 — Run RAG feedback demo for a single A1 essay
python3 src/07_rag_feedback.py

# Step 8 — Batch evaluate RAG feedback on 10 A1 essays
python3 src/08_evaluation.py
```

> **Note:** Steps 1–6 are computationally intensive. Steps 7–8 require Ollama running locally with `qwen2.5:3b` loaded.

---

## Data

### EFCAMDAT Corpus

The full dataset used in this project is the **EF Cambridge Open Language Database (EFCAMDAT)**, a large-scale longitudinal corpus of EFL learner writing at six CEFR proficiency levels (A1–C2).

Access requires free registration at:
**https://ef-lab.mmll.cam.ac.uk/EFCAMDAT.html**

Once registered, download `EFCAMDAT_Database.xml` and place it in the project root directory before running `src/01_data_extraction.py`.

> The XML file is ~1.2 GB and is excluded from this repository via `.gitignore`.

### Sample Data

`data/sample_essays.csv` contains 120 sentences (20 per CEFR level) as a reference for the data format. It is not sufficient to reproduce the full analysis.

### Local LLM

Steps 7 and 8 require **Ollama** running locally with the `qwen2.5:3b` model:
- Install Ollama: https://ollama.com
- Pull the model: `ollama pull qwen2.5:3b`

---

## Key Results

| Research Question | Finding |
|---|---|
| **RQ1 — Discourse Move Identification** | HDBSCAN identified 44 raw clusters, mapped to **8 discourse move types**: Social_Opening, Self_Introduction, Physical_Description, Daily_Routine, Narrative_Experience, Opinion_Evaluation, Information_Reporting, Social_Closing. The distribution varies significantly across CEFR levels (see `outputs/move_distribution_v2.png`). |
| **RQ2 — Knowledge Graph Contrast** | A1 essays predominantly use Social_Opening, Self_Introduction, and Social_Closing. C2 essays show richer transition patterns with higher proportions of Information_Reporting and Opinion_Evaluation, and stronger self-loop tendencies within complex moves. |
| **RQ3 — RAG Feedback Evaluation** | Evaluated over 10 A1 essays: **Move Mention Accuracy (M1) = 0.90**, Structural Focus Score (M2) = keyword density of structure-related language, Edit Distance Reduction Potential (M3) measures actionability of feedback. Full results in `outputs/rag_evaluation_results.csv`. |

---

## Citation

This work is currently under review. Citation will be updated upon publication.

```bibtex
@misc{ju2025discourse,
  title  = {Discourse Move Structures in ESL Learner Writing:
             A Knowledge Graph and RAG-Based Approach to Structural Feedback},
  author = {Ju, Hanyu},
  year   = {2025},
  note   = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Hanyu Ju

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
