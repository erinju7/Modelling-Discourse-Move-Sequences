import ast
from collections import Counter, defaultdict
import pandas as pd

df = pd.read_csv("/Users/macbook/Desktop/AIED cw2/essay_sequences.csv")
df["move_sequence"] = df["move_sequence"].apply(ast.literal_eval)

c2 = df[df["cefr_level"] == "C2"].copy()
c2["move_sequence"] = c2["move_sequence"].apply(lambda s: [m for m in s if m != "Other"])
c2 = c2[c2["move_sequence"].apply(len) > 0]

# First and last moves
first = Counter(s[0] for s in c2["move_sequence"])
last  = Counter(s[-1] for s in c2["move_sequence"])

print(f"Most common first move: {first.most_common(1)[0]}")
print(f"Most common last move:  {last.most_common(1)[0]}")

# Bigrams
bigram_counts = Counter()
source_totals = defaultdict(int)
for seq in c2["move_sequence"]:
    for a, b in zip(seq[:-1], seq[1:]):
        bigram_counts[(a, b)] += 1
        source_totals[a] += 1

print("\nTop 5 bigram transitions:")
for (a, b), count in bigram_counts.most_common(5):
    prob = count / source_totals[a]
    print(f"  {a} → {b}  (count={count}, prob={prob:.2f})")

# Trigrams
trigrams = Counter()
for seq in c2["move_sequence"]:
    for tri in zip(seq[:-2], seq[1:-1], seq[2:]):
        trigrams[tri] += 1

top_tri, top_count = trigrams.most_common(1)[0]
print(f"\nMost common trigram: {' → '.join(top_tri)}  (count={top_count})")
