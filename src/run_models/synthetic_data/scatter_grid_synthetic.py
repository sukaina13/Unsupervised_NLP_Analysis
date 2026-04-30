"""
scatter_grid_synthetic.py

Produces a Figure-2-style scatter grid for synthetic datasets with both embedding models:
- Rows: 8 (4 configs × 2 models: MiniLM top half, Qwen bottom half)
- Columns: 6 DR methods (PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP)
- Points colored by top-level category (category 0)

Run from repo root:
    python src/run_models/scatter_grid_synthetic.py
"""

import os
import sys
import csv
import numpy as np
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

LABEL_DIR = "../data/synthetic/generated_data"
OUT_DIR   = "../results/summary_figures"
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", "TriMAP"]
METHOD_LABELS = {"tSNE": "t-SNE"}

CONFIGS = [
    (
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub3_depth5_synonyms0_random",
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub3_depth5_synonyms0_noise0.0_random",
        "Ecosystems (d)",
    ),
    (
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub5_depth3_synonyms0_random",
        "Energy_Ecosystems_and_Humans_hierarchy_t1.0_maxsub5_depth3_synonyms0_noise0.0_random",
        "Ecosystems (s)",
    ),
    (
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub3_depth5_synonyms0_random",
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub3_depth5_synonyms0_noise0.0_random",
        "Fisheries (d)",
    ),
    (
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub5_depth3_synonyms0_random",
        "Offshore_energy_impacts_on_fisheries_hierarchy_t1.0_maxsub5_depth3_synonyms0_noise0.0_random",
        "Fisheries (s)",
    ),
]

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#6A4C93", "#80B918", "#FF6B6B",
    "#4CC9F0", "#F72585",
]

# Extended palette for category 1 (up to 27 unique values)
PALETTE_EXT = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#80B918", "#FF6B6B", "#4CC9F0", "#F72585",
    "#264653", "#E76F51", "#A8DADC", "#F1FAEE", "#1D3557",
    "#52B788", "#B7E4C7", "#D62828", "#023E8A", "#F77F00",
    "#FCBF49", "#EAE2B7", "#003049", "#8338EC", "#3A86FF",
    "#FB5607", "#FFBE0B",
]

def load_labels(label_csv_stem, cat_level=0):
    path = os.path.join(LABEL_DIR, f"{label_csv_stem}.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    col = f"category {cat_level}"
    fallback = "category 0"
    return [r[col] if r[col] else r[fallback] for r in rows]

def encode_labels(labels):
    unique = sorted(set(labels))
    mapping = {v: i for i, v in enumerate(unique)}
    return np.array([mapping[l] for l in labels]), unique

n_cols = len(METHODS)

def make_scatter_grid(model_name, reduction_dir, out_filename, title,
                      cat_level=0):
    """
    cat_level: category level used to color ALL panels (0 or 1).
    cat_level=0 produces the primary figures; cat_level=1 is the
    exploratory second-level coloring for all methods.
    """
    n_rows = len(CONFIGS)
    col_width  = 4.0 if cat_level == 0 else 5.0
    row_height = 3.5 if cat_level == 0 else 5.0
    pt_size    = 12  if cat_level == 0 else 18
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * col_width, n_rows * row_height),
                             facecolor="white")
    fig.patch.set_facecolor("white")

    pal = PALETTE_EXT if cat_level != 0 else PALETTE

    for r, (stem, label_stem, row_label) in enumerate(CONFIGS):
        labels  = load_labels(label_stem, cat_level=cat_level)
        encoded, unique_cats = encode_labels(labels)
        point_colors = [pal[i % len(pal)] for i in encoded]

        for c, method in enumerate(METHODS):
            ax = axes[r, c]
            ax.set_facecolor("#F7F7F7")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")
                spine.set_linewidth(0.8)

            npy_path = os.path.join(reduction_dir, f"{method}_2d_{stem}.npy")

            if not os.path.exists(npy_path):
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            else:
                x2d = np.load(npy_path)
                ax.scatter(x2d[:, 0], x2d[:, 1], c=point_colors,
                           s=pt_size, alpha=0.85, linewidths=0, rasterized=True)
                for dim, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
                    lo, hi = np.percentile(x2d[:, dim], [1, 99])
                    pad = (hi - lo) * 0.05
                    setter(lo - pad, hi + pad)

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#cccccc")
            ax.set_facecolor("#f0f0f0")

            if r == 0:
                ax.set_title(METHOD_LABELS.get(method, method),
                             fontsize=12, fontweight="bold", pad=6)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, fontweight="bold", labelpad=8)

        if cat_level == 0:
            # per-row legend: fits cleanly with 5-6 top-level categories
            patches = [
                mpatches.Patch(color=pal[i % len(pal)], label=cat)
                for i, cat in enumerate(unique_cats)
            ]
            axes[r, -1].legend(
                handles=patches,
                loc="center left",
                bbox_to_anchor=(1.03, 0.5),
                fontsize=11,
                frameon=True,
                framealpha=0.9,
                edgecolor="#CCCCCC",
                title="Category",
                title_fontsize=12,
            )

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.005)

    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    out_path = os.path.join(OUT_DIR, out_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    if cat_level == 1:
        # save legend as a separate figure
        eco_labels  = sorted(set(
            lbl for _, label_stem, _ in CONFIGS[:2]
            for lbl in load_labels(label_stem, cat_level=cat_level)
        ))
        fish_labels = sorted(set(
            lbl for _, label_stem, _ in CONFIGS[2:]
            for lbl in load_labels(label_stem, cat_level=cat_level)
        ))

        legend_fig, axes_leg = plt.subplots(1, 2, figsize=(18, max(len(eco_labels), len(fish_labels)) * 0.35 + 1))
        legend_fig.patch.set_facecolor("white")

        for ax_leg, labels, title in zip(axes_leg,
                                         [eco_labels, fish_labels],
                                         ["Ecosystems - Level-2 Categories",
                                          "Fisheries - Level-2 Categories"]):
            patches = [mpatches.Patch(color=pal[i % len(pal)], label=lbl)
                       for i, lbl in enumerate(labels)]
            ax_leg.legend(handles=patches, fontsize=11, frameon=True,
                          framealpha=0.95, edgecolor="#CCCCCC",
                          title=title, title_fontsize=12,
                          loc="upper left", ncol=1)
            ax_leg.axis("off")

        legend_path = os.path.join(OUT_DIR, out_filename.replace(".png", "_legend.png"))
        legend_fig.savefig(legend_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(legend_fig)
        print(f"Saved legend: {legend_path}")
    print(f"Saved: {out_path}")

for model_name, reduction_dir in EMBEDDING_MODELS:
    slug = "minilm" if "MiniLM" in model_name else "qwen"
    # primary figures: all methods colored by top-level category (cat0)
    make_scatter_grid(
        model_name=model_name,
        reduction_dir=reduction_dir,
        out_filename=f"fig2_scatter_grid_{slug}.png",
        title=f"Synthetic Dataset Visualizations ({model_name})",
        cat_level=0,
    )
    # exploratory figures: all methods colored by second-level category (cat1)
    make_scatter_grid(
        model_name=model_name,
        reduction_dir=reduction_dir,
        out_filename=f"fig2_scatter_grid_{slug}_cat1.png",
        title=f"Synthetic Dataset Visualizations ({model_name}): Level-2 Categories",
        cat_level=1,
    )
