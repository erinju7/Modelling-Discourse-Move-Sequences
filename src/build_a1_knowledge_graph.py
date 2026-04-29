import os
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None  # type: ignore

import ast
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE            = "/Users/macbook/Desktop/AIED cw2"
SEQ_FILE        = f"{BASE}/essay_sequences.csv"
LABELLED_CSV    = f"{BASE}/clustering_labelled.csv"
TRANSITION_CSV  = f"{BASE}/a1_transition_matrix.csv"
GRAPH_GML       = f"{BASE}/a1_knowledge_graph.gml"
GRAPH_PNG       = f"{BASE}/a1_knowledge_graph.png"

MOVES = [
    "Social_Opening", "Self_Introduction", "Physical_Description",
    "Daily_Routine", "Narrative_Experience", "Opinion_Evaluation",
    "Information_Reporting", "Social_Closing",
]
THRESHOLD = 0.10

# ── Step 1: Load and filter A1 sequences ─────────────────────────────────────
print("Loading essay sequences ...")
df = pd.read_csv(SEQ_FILE)
df["move_sequence"] = df["move_sequence"].apply(ast.literal_eval)

a1 = df[df["cefr_level"] == "A1"].copy()
a1["move_sequence"] = a1["move_sequence"].apply(
    lambda s: [m for m in s if m != "Other"]
)
a1 = a1[a1["move_sequence"].apply(len) > 0].copy()
print(f"  Total essays: {len(df)}  |  A1 essays after filtering: {len(a1)}")

all_a1_moves = [m for seq in a1["move_sequence"] for m in seq]

# ── Step 2: Transition matrix ─────────────────────────────────────────────────
print("Building transition matrix ...")

bigram_counts = Counter()
source_totals = defaultdict(int)
for seq in a1["move_sequence"]:
    for a, b in zip(seq[:-1], seq[1:]):
        bigram_counts[(a, b)] += 1
        source_totals[a] += 1

rows = []
for (from_move, to_move), count in sorted(bigram_counts.items()):
    prob = count / source_totals[from_move]
    rows.append({"from_move": from_move, "to_move": to_move,
                 "count": count, "probability": round(prob, 4)})

trans_df = pd.DataFrame(rows).sort_values(["from_move", "probability"],
                                           ascending=[True, False])
trans_df.to_csv(TRANSITION_CSV, index=False)
print(f"  Transition pairs: {len(trans_df)}  |  Saved {TRANSITION_CSV}")

# ── Step 3: Build directed graph ──────────────────────────────────────────────
print("\nBuilding knowledge graph ...")
G = nx.DiGraph()
G.add_nodes_from(MOVES)

for _, row in trans_df[trans_df["probability"] > THRESHOLD].iterrows():
    G.add_edge(row["from_move"], row["to_move"],
               weight=row["probability"], count=row["count"])

print(f"  Nodes: {G.number_of_nodes()}  |  Edges (prob > {THRESHOLD}): {G.number_of_edges()}")
nx.write_gml(G, GRAPH_GML)
print(f"  Saved {GRAPH_GML}")

# ── Step 4: Visualise ─────────────────────────────────────────────────────────
print("\nDrawing graph ...")

# Node colour: A1-vs-C2 dominance — dark blue = A1-dominant (reversed from C2 graph)
labelled = pd.read_csv(LABELLED_CSV)
level_move = labelled.groupby(["cefr_level", "discourse_move"]).size().unstack(fill_value=0)
level_move_norm = level_move.div(level_move.sum(axis=1), axis=0)

eps = 1e-6
a1_ratio = {}
for move in MOVES:
    p_a1 = level_move_norm.loc["A1", move] if move in level_move_norm.columns else eps
    p_c2 = level_move_norm.loc["C2", move] if move in level_move_norm.columns else eps
    # log2(A1/C2): positive = A1-dominant (dark), negative = C2-dominant (light)
    a1_ratio[move] = np.log2((p_a1 + eps) / (p_c2 + eps))

ratio_vals = np.array([a1_ratio.get(m, 0) for m in G.nodes()])
norm_vals  = (ratio_vals - ratio_vals.min()) / (ratio_vals.max() - ratio_vals.min() + eps)
node_colours = plt.cm.Blues(0.2 + 0.75 * norm_vals)

# Node sizes: proportional to A1 frequency
move_freq = Counter(all_a1_moves)
max_freq  = max(move_freq.get(m, 1) for m in G.nodes())
node_sizes = [2000 + 5000 * (move_freq.get(m, 0) / max_freq) for m in G.nodes()]

# Edge styling
edges        = list(G.edges(data=True))
edge_probs   = [d["weight"] for _, _, d in edges]
edge_widths  = [1.5 + 7 * p for p in edge_probs]
edge_colours = plt.cm.Oranges([0.35 + 0.65 * p for p in edge_probs])

fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.circular_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                       node_color=node_colours, alpha=0.92, ax=ax)

# Text colour: white on dark nodes, black on light
node_list  = list(G.nodes())
short_labels = {m: m.replace("_", "\n") for m in G.nodes()}
for node, (x, y) in pos.items():
    n_idx = node_list.index(node)
    text_color = "white" if norm_vals[n_idx] > 0.5 else "black"
    ax.text(x, y, short_labels[node],
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=text_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="none",
                      edgecolor="none", alpha=0.7),
            zorder=5)

for (u, v, data), width, colour in zip(edges, edge_widths, edge_colours):
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        width=width, edge_color=[colour],
        arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.18",
        ax=ax,
    )

edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=7.5, label_pos=0.35, ax=ax)

ax.set_title(
    "A1 Discourse Move Transition Graph\n"
    "(node size = A1 frequency  |  node colour = A1-vs-C2 dominance  |  edge weight = transition probability)",
    fontsize=13, pad=15,
)
ax.axis("off")

sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                            norm=plt.Normalize(vmin=ratio_vals.min(), vmax=ratio_vals.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, location="right")
cbar.set_label("C2-dominant ← → A1-dominant", fontsize=9, labelpad=8)

legend_elements = [
    mpatches.Patch(color=plt.cm.Oranges(0.4),  label="Lower probability transition"),
    mpatches.Patch(color=plt.cm.Oranges(0.95), label="Higher probability transition"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig(GRAPH_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {GRAPH_PNG}")

# ── Step 5: Insights ──────────────────────────────────────────────────────────
print("\n── A1 Knowledge Graph Insights ─────────────────────────────────────────")
print(f"  Total A1 essays analysed: {len(a1)}")

first_moves = Counter(s[0] for s in a1["move_sequence"])
last_moves  = Counter(s[-1] for s in a1["move_sequence"])
print(f"\nMost common first move:")
for move, cnt in first_moves.most_common(3):
    print(f"  {move}: {cnt} ({100*cnt/len(a1):.1f}%)")

print(f"\nMost common last move:")
for move, cnt in last_moves.most_common(3):
    print(f"  {move}: {cnt} ({100*cnt/len(a1):.1f}%)")

print(f"\nTop 5 transitions by probability:")
for _, row in trans_df.nlargest(5, "probability").iterrows():
    print(f"  {row['from_move']} → {row['to_move']}  "
          f"(count={row['count']}, prob={row['probability']:.2f})")

trigrams = Counter()
for seq in a1["move_sequence"]:
    for tri in zip(seq[:-2], seq[1:-1], seq[2:]):
        trigrams[tri] += 1

if trigrams:
    top_tri, top_count = trigrams.most_common(1)[0]
    print(f"\nMost common trigram: {' → '.join(top_tri)}  (count={top_count})")
else:
    print("\nNo trigrams found (sequences too short).")
