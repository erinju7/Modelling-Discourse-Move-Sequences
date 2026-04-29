import csv
import nltk
from pathlib import Path
from lxml import etree

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# EFCAMDAT level attribute is numeric: 1=A1, 2=A2, 3=B1, 4=B2, 5=C1, 6=C2
LEVEL_MAP = {"1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1", "6": "C2"}
TARGET_LEVELS = {"1", "6"}  # A1 and C2

XML_FILE = Path(__file__).parent / "EFCAMDAT_Database.xml"
OUTPUT_CSV = Path(__file__).parent / "essays_a1_c2.csv"

counts = {"A1": 0, "C2": 0}
rows = []

print(f"Parsing {XML_FILE} ...")

# lxml recover=True silently repairs unclosed/mismatched tags (e.g. <br>)
for event, elem in etree.iterparse(XML_FILE, events=("end",), recover=True):
    if elem.tag != "writing":
        continue

    level_num = elem.get("level", "")
    if level_num not in TARGET_LEVELS:
        elem.clear()
        continue

    cefr = LEVEL_MAP[level_num]
    essay_id = elem.get("id", "")

    learner_elem = elem.find("learner")
    learner_id = learner_elem.get("id", "") if learner_elem is not None else ""

    text_elem = elem.find("text")
    # etree.tostring with method="text" extracts all text, stripping nested XML tags
    if text_elem is not None:
        text = etree.tostring(text_elem, method="text", encoding="unicode").strip()
    else:
        text = ""

    if not text:
        elem.clear()
        continue

    sentences = nltk.sent_tokenize(text)
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:
            rows.append({
                "essay_id": essay_id,
                "learner_id": learner_id,
                "cefr_level": cefr,
                "sentence_index": idx,
                "sentence": sentence,
            })

    counts[cefr] += 1
    elem.clear()  # free memory

print(f"\nEssays found — A1: {counts['A1']}, C2: {counts['C2']}")
print(f"Total sentences: {len(rows)}")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["essay_id", "learner_id", "cefr_level", "sentence_index", "sentence"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {OUTPUT_CSV}")
