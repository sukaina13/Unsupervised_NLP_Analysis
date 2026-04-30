"""
recompute_tsne_sklearn.py

Recomputes t-SNE for all 12 synthetic configs x 2 embedding models using
scikit-learn's TSNE, replacing any previously cached .npy files (whether
produced by cuML on HPCC or an earlier sklearn run).

Motivation: cuML TSNE on HPCC produced degenerate near-zero output for some
smaller configs. To ensure all configs are processed identically, we use
sklearn TSNE with consistent parameters for all 24 files.

Parameters used (same for all configs):
    - PCA preprocessing: 50 components, random_state=67
    - TSNE: init='pca', learning_rate='auto', max_iter=2000,
            perplexity=min(30, n_samples // 4), random_state=67

Metrics are recomputed and CSVs updated after each config.

Run from repo root (src/ directory):
    python run_models/synthetic_data/recompute_tsne_sklearn.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'src':
    parent = os.path.abspath(os.path.join(current_dir, '..'))
    if parent == current_dir:
        raise FileNotFoundError("src/ not found in directory tree.")
    current_dir = parent
os.chdir(current_dir)
sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import TSNE as skTSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

# ========================
# Config
# ========================

embedding_models = [
    ("sentence-transformers/all-MiniLM-L6-v2",
     "cache/sentence-transformers/all-MiniLM-L6-v2_embeddings",
     "cache/sentence-transformers/all-MiniLM-L6-v2_reduced_2d",
     "cache/sentence-transformers/all-MiniLM-L6-v2_results"),
    ("Qwen/Qwen3-Embedding-0.6B",
     "cache/Qwen/Qwen3-Embedding-0.6B_embeddings",
     "cache/Qwen/Qwen3-Embedding-0.6B_reduced_2d",
     "cache/Qwen/Qwen3-Embedding-0.6B_results"),
]

configs = [
    ("Energy_Ecosystems_and_Humans",          5, 3, 0.0),
    ("Energy_Ecosystems_and_Humans",          5, 3, 0.25),
    ("Energy_Ecosystems_and_Humans",          5, 3, 0.5),
    ("Energy_Ecosystems_and_Humans",          3, 5, 0.0),
    ("Energy_Ecosystems_and_Humans",          3, 5, 0.25),
    ("Energy_Ecosystems_and_Humans",          3, 5, 0.5),
    ("Offshore_energy_impacts_on_fisheries",  5, 3, 0.0),
    ("Offshore_energy_impacts_on_fisheries",  5, 3, 0.25),
    ("Offshore_energy_impacts_on_fisheries",  5, 3, 0.5),
    ("Offshore_energy_impacts_on_fisheries",  3, 5, 0.0),
    ("Offshore_energy_impacts_on_fisheries",  3, 5, 0.25),
    ("Offshore_energy_impacts_on_fisheries",  3, 5, 0.5),
]

RANDOM_STATE  = 67
PCA_COMPONENTS = 50

# ========================
# Helpers
# ========================

def build_stem(theme, max_sub, depth, add_noise, t=1.0, synonyms=0, branching="random"):
    if add_noise > 0:
        return (f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}"
                f"_synonyms{synonyms}_noise{add_noise}_{branching}")
    return f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}"

def compute_tsne(x_high):
    n = x_high.shape[0]
    n_pca = min(PCA_COMPONENTS, x_high.shape[1])
    x_pca = skPCA(n_components=n_pca, random_state=RANDOM_STATE).fit_transform(x_high)
    perplexity = min(30, n // 4)
    tsne = skTSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        max_iter=2000,
        perplexity=perplexity,
        random_state=RANDOM_STATE,
    )
    return tsne.fit_transform(x_pca)

def compute_continuity(x_high, x_low, n_neighbors=15):
    n = x_high.shape[0]
    d_high   = pairwise_distances(x_high)
    d_low    = pairwise_distances(x_low)
    rank_low = np.argsort(np.argsort(d_low, axis=1), axis=1)
    continuity = 0.0
    for i in range(n):
        neighbors_high = set(np.argsort(d_high[i])[1:n_neighbors+1])
        neighbors_low  = set(np.argsort(d_low[i])[1:n_neighbors+1])
        for j in neighbors_high - neighbors_low:
            continuity += rank_low[i, j] - n_neighbors
    norm = 2.0 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))
    return 1 - norm * continuity

def compute_demap(x_high, x_low_2d, k_min=3, k_max=15):
    for k in range(k_min, k_max + 1):
        knn = kneighbors_graph(x_high, n_neighbors=k, mode='distance', include_self=False)
        geo = shortest_path(knn, directed=False)
        if not np.any(np.isinf(geo)):
            print(f"    DEMaP using K={k} (min connected)")
            break
    if np.any(np.isinf(geo)):
        max_finite = np.nanmax(geo[np.isfinite(geo)])
        geo[np.isinf(geo)] = 1 + max_finite
    idx      = np.triu_indices(x_high.shape[0], k=1)
    geo_flat = geo[idx]
    euc_flat = pairwise_distances(x_low_2d)[idx]
    return spearmanr(geo_flat, euc_flat)[0]

def compute_metrics(x_high, x_low):
    t  = trustworthiness(x_high, x_low, n_neighbors=15)
    c  = compute_continuity(x_high, x_low, n_neighbors=15)
    sp = spearmanr(pairwise_distances(x_high).flatten(),
                   pairwise_distances(x_low).flatten())[0]
    dm = compute_demap(x_high, x_low)
    return t, c, sp, dm

# ========================
# Main loop
# ========================

for model_name, embed_dir, reduction_dir, results_dir in embedding_models:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    for theme, max_sub, depth, add_noise in configs:
        stem = build_stem(theme, max_sub, depth, add_noise)
        print(f"\n  --- {stem} ---")

        embed_path = os.path.join(embed_dir, f"{stem}_embed.npy")
        if not os.path.exists(embed_path):
            print(f"  Embeddings not found, skipping.")
            continue

        x_high = np.load(embed_path)
        n = x_high.shape[0]
        print(f"  n={n}, dim={x_high.shape[1]}, perplexity=min(30, {n}//4)={min(30, n//4)}")

        # recompute t-SNE with sklearn (overwrites any existing file)
        tsne_path = os.path.join(reduction_dir, f"tSNE_2d_{stem}.npy")
        print(f"  Computing sklearn t-SNE...")
        x_tsne = compute_tsne(x_high)
        xr = x_tsne[:, 0].max() - x_tsne[:, 0].min()
        yr = x_tsne[:, 1].max() - x_tsne[:, 1].min()
        print(f"  Output range: x={xr:.2f}, y={yr:.2f}")
        np.save(tsne_path, x_tsne)
        print(f"  Saved: {tsne_path}")

        # recompute metrics for t-SNE
        print(f"  Computing metrics...")
        t, c, sp, dm = compute_metrics(x_high, x_tsne)
        print(f"  tSNE: T={t:.4f}, C={c:.4f}, Spearman={sp:.4f}, DEMaP={dm:.4f}")

        # update CSV — replace tSNE row (or add if missing)
        csv_path = os.path.join(results_dir, f"viz_metrics_{stem}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df["Method"] != "tSNE"]  # drop old tSNE row
        else:
            df = pd.DataFrame()

        new_row = pd.DataFrame([{
            "Method":               "tSNE",
            "Trustworthiness":      round(t,  4),
            "Continuity":           round(c,  4),
            "Spearman Correlation": round(sp, 4),
            "DEMaP":                round(dm, 4),
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"  Updated CSV: {csv_path}")

print(f"\n{'='*60}")
print("All configs complete.")
print(f"{'='*60}")
