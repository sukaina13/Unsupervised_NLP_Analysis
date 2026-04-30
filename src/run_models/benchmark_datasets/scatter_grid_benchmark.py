"""
scatter_grid_benchmark.py

Produces a scatter grid for benchmark datasets:
- Rows: rcv1, arxiv, amazon, wos, dbpedia (5 datasets with matching label/embedding alignment)
- Columns: PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP (6 DR methods)
- Points colored by top-level category (category_0)
- One figure per embedding model (MiniLM, Qwen)

Run from repo root:
    python src/run_models/benchmark_datasets/scatter_grid_benchmark.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# path setup
current = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current) != "src" and current != os.path.dirname(current):
    current = os.path.dirname(current)
src_dir = current
os.chdir(src_dir)
sys.path.insert(0, src_dir)

EMBEDDING_MODELS = [
    ("MiniLM", "cache/sentence-transformers/all-MiniLM-L6-v2_reduced_2d"),
    ("Qwen",   "cache/Qwen/Qwen3-Embedding-0.6B_reduced_2d"),
]

OUT_DIR = "../results/summary_figures"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", "TriMAP"]
METHOD_LABELS = {"tSNE": "t-SNE"}

VIS_SUBSAMPLE = None  # set to an int (e.g. 5000) to subsample for readability; None = all points

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#6A4C93", "#80B918", "#FF6B6B",
    "#4CC9F0", "#F72585", "#023E8A", "#8338EC",
    "#FB5607", "#3A86FF", "#06D6A0",
]


# ========================
# Data loaders (mirror eval_pipeline.py exactly)
# ========================

def load_arxiv():
    df = pd.read_csv('../data/arxiv/arxiv_clean.csv')
    df = df.dropna().reset_index(drop=True)
    return df


def load_rcv1():
    df = pd.read_csv('../data/rcv1/rcv1.csv')
    df = df.drop_duplicates(subset='topic', keep=False).reset_index(drop=True)
    df = df.drop_duplicates(subset='item_id', keep=False).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    df = df[df['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)
    return df


def load_amazon():
    amz_40 = pd.read_csv('../data/amazon/train_40k.csv')
    amz_10 = pd.read_csv('../data/amazon/val_10k.csv')
    df = pd.concat([amz_40, amz_10])
    df = df.drop_duplicates(subset='Title', keep=False).reset_index(drop=True)
    df = df.drop_duplicates(subset='productId', keep=False).reset_index(drop=True)
    df = df.dropna().reset_index(drop=True)
    df = df[df['Title'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)
    df = df.rename(columns={'Cat1': 'category_0'})
    return df


def load_wos():
    raw = pd.read_excel('../data/WebOfScience/Data.xlsx')
    df = pd.DataFrame([
        {'topic': str(r['keywords']), 'category_0': r['Domain'], 'category_1': r['area']}
        for _, r in raw.iterrows()
    ])
    return df


def load_dbpedia():
    def clean_dbpedia(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    db = pd.read_csv('../data/dbpedia/DBPEDIA_test.csv')
    db = db.rename(columns={"text": "topic", "l1": "category_0", "l2": "category_1", "l3": "category_2"})
    db['topic'] = db['topic'].astype(str).apply(clean_dbpedia)
    return db


DATASETS = [
    ("rcv1",    load_rcv1,    1566,  "RCV1"),
    ("arxiv",   load_arxiv,   29966, "arXiv"),
    ("amazon",  load_amazon,  14824, "Amazon"),
    ("wos",     load_wos,     46985, "WoS"),
    ("dbpedia", load_dbpedia, 60794, "DBpedia"),
]


# ========================
# Plotting
# ========================

def make_scatter_grid(model_label, cache_dir, out_suffix):
    n_rows = len(DATASETS)
    n_cols = len(METHODS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.0, n_rows * 3.5),
                             facecolor="white")
    fig.patch.set_facecolor("white")

    all_categories = {}  # dataset -> categories list, for legend figure

    for row_idx, (dataset, loader, n_full, row_label) in enumerate(DATASETS):
        print(f"  Loading {dataset}...")
        df = loader()
        if len(df) != n_full:
            df = df.iloc[:n_full].reset_index(drop=True)

        labels_raw = df['category_0'].values
        categories = sorted(df['category_0'].unique())
        all_categories[row_label] = categories
        cat2idx = {c: i for i, c in enumerate(categories)}
        color_idx = np.array([cat2idx[l] for l in labels_raw])

        n_actual = len(df)
        # subsample same indices across all methods for comparability
        np.random.seed(42)
        if VIS_SUBSAMPLE is not None and n_actual > VIS_SUBSAMPLE:
            vis_idx = np.random.choice(n_actual, VIS_SUBSAMPLE, replace=False)
        else:
            vis_idx = np.arange(n_actual)

        colors_vis = [PALETTE[color_idx[i] % len(PALETTE)] for i in vis_idx]

        for col_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            npy_path = os.path.join(cache_dir, f"{method}_2d_{dataset}_full{n_full}.npy")

            if not os.path.exists(npy_path):
                ax.set_visible(False)
                print(f"    Missing: {npy_path}")
                continue

            coords = np.load(npy_path)[vis_idx]
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colors_vis, s=4, alpha=0.4, linewidths=0)
            for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
                lo, hi = np.percentile(coords[:, dim], [1, 99])
                pad = (hi - lo) * 0.05
                setter(lo - pad, hi + pad)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            ax.set_facecolor("#f0f0f0")

            if row_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=13, fontweight='bold', pad=6)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=12, fontweight='bold', labelpad=6)

    plt.suptitle(f"Benchmark Scatter Grid ({model_label})", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"fig_scatter_grid_benchmark_{out_suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # save legend as a separate figure
    max_cats = max(len(c[:15]) for c in all_categories.values())
    legend_fig, axes_leg = plt.subplots(1, len(all_categories),
                                        figsize=(len(all_categories) * 3.5, max_cats * 0.4 + 1))
    legend_fig.patch.set_facecolor("white")
    for ax_leg, (row_label, categories) in zip(axes_leg, all_categories.items()):
        legend_cats = categories[:15]
        title = row_label if len(categories) <= 15 else f"{row_label} (top 15/{len(categories)})"
        patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=str(c))
                   for i, c in enumerate(legend_cats)]
        ax_leg.legend(handles=patches, fontsize=11, frameon=True, framealpha=0.95,
                      edgecolor="#CCCCCC", title=title, title_fontsize=12,
                      loc="upper left", ncol=1)
        ax_leg.axis("off")
    legend_path = os.path.join(OUT_DIR, f"fig_scatter_grid_benchmark_{out_suffix}_legend.png")
    legend_fig.savefig(legend_path, dpi=200, bbox_inches='tight', facecolor="white")
    plt.close(legend_fig)
    print(f"Saved legend: {legend_path}")


# ========================
# Main
# ========================

for model_label, cache_dir in EMBEDDING_MODELS:
    print(f"\n{'='*50}")
    print(f"Model: {model_label}")
    print('='*50)
    out_suffix = "minilm" if "MiniLM" in model_label else "qwen"
    make_scatter_grid(model_label, cache_dir, out_suffix)
