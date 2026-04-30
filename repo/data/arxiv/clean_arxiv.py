"""
clean_arxiv.py

Processes the raw arXiv JSONL snapshot into a 30k-paper CSV for benchmarking.

Input:  data/arxiv/arxiv-metadata-oai-snapshot.json  (Kaggle download)
Output: data/arxiv/arxiv_clean.csv

Usage (from repo root):
    python data/arxiv/clean_arxiv.py
"""

import os
import sys
import json
import random
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
random.seed(42)

# Resolve repo root regardless of where the script is called from
current_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(current_dir, "src")):
    parent = os.path.abspath(os.path.join(current_dir, ".."))
    if parent == current_dir:
        raise FileNotFoundError("Could not find repo root.")
    current_dir = parent

RAW_PATH = os.path.join(current_dir, "data", "arxiv", "arxiv-metadata-oai-snapshot.json")
OUT_PATH  = os.path.join(current_dir, "data", "arxiv", "arxiv_clean.csv")
N_SAMPLES = 30000

if not os.path.exists(RAW_PATH):
    print(f"Raw file not found: {RAW_PATH}")
    print("Download it from https://www.kaggle.com/datasets/Cornell-University/arxiv")
    sys.exit(1)

print(f"Reading {RAW_PATH} ...")

records = []
with open(RAW_PATH, "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        abstract   = entry.get("abstract", "").strip().replace("\n", " ")
        categories = entry.get("categories", "").strip()

        if not abstract or not categories:
            continue

        # Use the first listed category, which must have a dot (e.g. "cs.AI")
        first_cat = categories.split()[0]
        if "." not in first_cat:
            continue

        cat0, cat1 = first_cat.split(".", 1)

        records.append({
            "topic":      abstract,
            "category_0": cat0,
            "category_1": cat1,
        })

print(f"Found {len(records):,} valid papers.")

if len(records) > N_SAMPLES:
    records = random.sample(records, N_SAMPLES)
    print(f"Sampled {N_SAMPLES:,} papers.")

df = pd.DataFrame(records)
df = df.dropna().reset_index(drop=True)
df = df[df["topic"].apply(lambda x: isinstance(x, str) and x.strip() != "")].reset_index(drop=True)

df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df):,} rows to {OUT_PATH}")
