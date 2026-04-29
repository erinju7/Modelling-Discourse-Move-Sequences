import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None  # type: ignore

import ast
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE            = "/Users/macbook/Desktop/AIED cw2"
SEQ_FILE        = f"{BASE}/essay_sequences.csv"
TRANSITION_CSV  = f"{BASE}/c2_transition_matrix.csv"
GRAPH_GML       = f"{BASE}/c2_knowledge_graph.gml"
GRAPH_PNG       = f"{BASE}/c2_knowledge_graph.png"
GRAPH_PNG_V2    = f"{BASE}/c2_knowledge_graph_v2.png"
LABELLED_CSV    = f"{BASE}/clustering_labelled.csv"

# 8 canonical discourse move labels
MOVES = [
    "Social_Opening", "Self_Introduction", "Physical_Description",
    "Daily_Routine", "Narrative_Experience", "Opinion_Evaluation",
    "Information_Reporting", "Social_Closing",
]

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract C2 sequences
# ─────────────────────────────────────────────────────────────────────────────
# Load the CSV. The move_sequence column was saved as a Python list repr string,
# e.g. "['Social_Opening', 'Daily_Routine', 'Social_Closing']"
# ast.literal_eval safely parses that string back into a real Python list.

print("Loading essay sequences ...")
df = pd.read_csv(SEQ_FILE)

# Parse the string representation back into a Python list
df["move_sequence"] = df["move_sequence"].apply(ast.literal_eval)

# Keep only C2 essays
c2 = df[df["cefr_level"] == "C2"].copy()
print(f"  Total essays: {len(df)}  |  C2 essays: {len(c2)}")

# Fix 1: Remove 'Other' from every sequence before any analysis.
# This prevents noise sentences from polluting bigrams and graph structure.
c2["move_sequence"] = c2["move_sequence"].apply(
    lambda seq: [m for m in seq if m != "Other"]
)
# Drop essays that become empty after filtering
c2 = c2[c2["move_sequence"].apply(len) > 0].copy()
print(f"  C2 essays after removing 'Other'-only sequences: {len(c2)}")

# Flatten all C2 sequences into a single list to count move frequencies
all_c2_moves = [move for seq in c2["move_sequence"] for move in seq]

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build transition matrix
# ─────────────────────────────────────────────────────────────────────────────
# For each essay, we slide a window of size 2 over the sequence to get
# consecutive move pairs (bigrams).
# Example: [A, B, C] → (A→B), (B→C)

print("Building transition matrix ...")

# Count raw occurrences of each (from_move, to_move) pair
transition_counts = Counter()
for seq in c2["move_sequence"]:
    for from_move, to_move in zip(seq[:-1], seq[1:]):   # consecutive pairs
        transition_counts[(from_move, to_move)] += 1

# Count how many times each source move appears as a "from" move
source_totals = defaultdict(int)
for (from_move, _), count in transition_counts.items():
    source_totals[from_move] += count

# Build rows for the CSV: from_move, to_move, count, probability
rows = []
for (from_move, to_move), count in sorted(transition_counts.items()):
    prob = count / source_totals[from_move]   # P(to | from)
    rows.append({"from_move": from_move, "to_move": to_move,
                 "count": count, "probability": round(prob, 4)})

trans_df = pd.DataFrame(rows).sort_values(["from_move", "probability"],
                                          ascending=[True, False])
trans_df.to_csv(TRANSITION_CSV, index=False)
print(f"  Transition pairs found: {len(trans_df)}")
print(f"  Saved {TRANSITION_CSV}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build directed graph with NetworkX
# ─────────────────────────────────────────────────────────────────────────────
# Nodes = discourse moves.  Edges = transitions with probability > 0.05.
# We attach the transition probability as the "weight" attribute on each edge.

print("\nBuilding knowledge graph ...")
G = nx.DiGraph()
G.add_nodes_from(MOVES)

# Fix 2: Raised threshold to 0.10 — only show the strongest structural transitions
THRESHOLD = 0.10
for _, row in trans_df[trans_df["probability"] > THRESHOLD].iterrows():
    G.add_edge(row["from_move"], row["to_move"],
               weight=row["probability"],
               count=row["count"])

print(f"  Nodes: {G.number_of_nodes()}  |  Edges (prob > {THRESHOLD}): {G.number_of_edges()}")

# Save as GML — a plain-text format readable by Gephi, Cytoscape, etc.
nx.write_gml(G, GRAPH_GML)
print(f"  Saved {GRAPH_GML}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Visualise the graph (v2 — improved layout, labels, node colours)
# ─────────────────────────────────────────────────────────────────────────────
print("\nDrawing graph (v2) ...")

# Fix 3a: Node colour coding based on C2 vs A1 frequency ratio
# Load the full labelled dataset to compute per-level move proportions
labelled = pd.read_csv(LABELLED_CSV)
level_move = labelled.groupby(["cefr_level", "discourse_move"]).size().unstack(fill_value=0)
# Proportion of sentences at each level that belong to each move
level_move_norm = level_move.div(level_move.sum(axis=1), axis=0)

# C2-vs-A1 ratio: log2(C2_proportion / A1_proportion).
# Positive = more common in C2 (darker blue); negative = more common in A1 (lighter).
# Add a small epsilon to avoid division by zero.
eps = 1e-6
c2_ratio = {}
for move in MOVES:
    p_c2 = level_move_norm.loc["C2", move] if move in level_move_norm.columns else eps
    p_a1 = level_move_norm.loc["A1", move] if move in level_move_norm.columns else eps
    c2_ratio[move] = np.log2((p_c2 + eps) / (p_a1 + eps))

# Normalise ratios to [0, 1] for the colourmap
ratio_vals = np.array([c2_ratio.get(m, 0) for m in G.nodes()])
norm_vals  = (ratio_vals - ratio_vals.min()) / (ratio_vals.max() - ratio_vals.min() + eps)
# Blues colourmap: 0.2 (light, A1-dominant) → 0.95 (dark, C2-dominant)
node_colours = plt.cm.Blues(0.2 + 0.75 * norm_vals)

# Node sizes: proportional to C2 frequency
move_freq = Counter(all_c2_moves)
max_freq  = max(move_freq.get(m, 1) for m in G.nodes())
node_sizes = [2000 + 5000 * (move_freq.get(m, 0) / max_freq) for m in G.nodes()]

# Edge styling
edges       = list(G.edges(data=True))
edge_probs  = [d["weight"] for _, _, d in edges]
edge_widths = [1.5 + 7 * p for p in edge_probs]
edge_colours = plt.cm.Oranges([0.35 + 0.65 * p for p in edge_probs])

# Circular layout — nodes evenly spaced around a circle, none fall off the edge
fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.circular_layout(G)

# Draw nodes with per-node colour
nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colours, alpha=0.92, ax=ax)

# Node labels: underscore → newline, white text on a semi-transparent rounded box
short_labels = {m: m.replace("_", "\n") for m in G.nodes()}
node_list = list(G.nodes())
for node, (x, y) in pos.items():
    # Use white text on dark nodes (norm_val > 0.5), black on light nodes
    n_idx = node_list.index(node)
    text_color = "white" if norm_vals[n_idx] > 0.5 else "black"
    ax.text(x, y, short_labels[node],
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="none",
                      edgecolor="none", alpha=0.7),
            zorder=5)

# Draw edges (curved arcs so bidirectional pairs don't overlap)
for (u, v, data), width, colour in zip(edges, edge_widths, edge_colours):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        width=width, edge_color=[colour],
        arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.18",
        ax=ax,
    )

# Edge probability labels
edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=7.5, label_pos=0.35, ax=ax)

ax.set_title(
    "C2 Discourse Move Transition Graph\n"
    "(node size = C2 frequency  |  node colour = C2-vs-A1 dominance  |  edge weight = transition probability)",
    fontsize=13, pad=15,
)
ax.axis("off")

# Colourbar for node colour meaning
sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                            norm=plt.Normalize(vmin=ratio_vals.min(), vmax=ratio_vals.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, location="right")
cbar.set_label("A1-dominant ← → C2-dominant", fontsize=9, labelpad=8)

# Edge colour legend
legend_elements = [
    mpatches.Patch(color=plt.cm.Oranges(0.4), label="Lower probability transition"),
    mpatches.Patch(color=plt.cm.Oranges(0.95), label="Higher probability transition"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(GRAPH_PNG_V2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {GRAPH_PNG_V2}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Insights
# ─────────────────────────────────────────────────────────────────────────────
print("\n── C2 Knowledge Graph Insights ─────────────────────────────────────────")

# Most common FIRST move in C2 essays
first_moves = Counter(seq[0] for seq in c2["move_sequence"] if seq)
print(f"\nMost common first move:")
for move, cnt in first_moves.most_common(3):
    print(f"  {move}: {cnt} essays ({100*cnt/len(c2):.1f}%)")

# Most common LAST move in C2 essays
last_moves = Counter(seq[-1] for seq in c2["move_sequence"] if seq)
print(f"\nMost common last move:")
for move, cnt in last_moves.most_common(3):
    print(f"  {move}: {cnt} essays ({100*cnt/len(c2):.1f}%)")

# Top 5 most frequent transitions
print(f"\nTop 5 transitions by probability:")
top5 = trans_df.nlargest(5, "probability")[["from_move", "to_move", "count", "probability"]]
for _, row in top5.iterrows():
    print(f"  {row['from_move']} → {row['to_move']}  "
          f"(count={row['count']}, prob={row['probability']:.2f})")

# Most common 3-move sequence (trigram)
trigrams = Counter()
for seq in c2["move_sequence"]:
    for tri in zip(seq[:-2], seq[1:-1], seq[2:]):   # sliding window of 3
        trigrams[tri] += 1

print(f"\nTop 5 trigrams (3-move sequences):")
for trigram, cnt in trigrams.most_common(5):
    print(f"  {' → '.join(trigram)}: {cnt} times")
