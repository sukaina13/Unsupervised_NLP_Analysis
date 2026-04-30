"""
Visualization Quality Metrics: Synthetic Data (Batch Script)

Runs Trustworthiness, Continuity, Spearman Correlation, and DEMaP
for all synthetic dataset configs. Results and Shepard diagrams are
saved to results/viz_metrics/{embedding_model}/.

Usage:
    python src/run_models/synthetic_data/viz_metrics_script.py
"""

import os
import sys

# navigate to src/
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
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script use
import matplotlib.pyplot as plt
import phate
import pacmap
import trimap
import umap as umap_pkg
from cuml.manifold import TSNE as cuTSNE
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

# ========================
# Config
# ========================

embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen3-Embedding-0.6B",
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

# ========================
# Helper functions
# ========================

def build_stem(theme, max_sub, depth, add_noise, t=1.0, synonyms=0, branching="random"):
    if add_noise > 0:
        return f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}"
    return f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}"

def load_or_compute_2d(name, path, compute_fn):
    if os.path.exists(path):
        print(f"  Loading cached 2D {name} from {path}...")
        return np.load(path)
    print(f"  Computing 2D {name}...")
    result = compute_fn()
    np.save(path, result)
    print(f"  Saved to {path}")
    return result

def compute_continuity(x_high, x_low, n_neighbors=15):
    n = x_high.shape[0]
    d_high   = pairwise_distances(x_high)
    d_low    = pairwise_distances(x_low)
    rank_low = np.argsort(np.argsort(d_low, axis=1), axis=1)
    continuity = 0.0
    for i in range(n):
        neighbors_high = set(np.argsort(d_high[i])[1:n_neighbors+1])
        neighbors_low  = set(np.argsort(d_low[i])[1:n_neighbors+1])
        missing = neighbors_high - neighbors_low
        for j in missing:
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

def plot_shepard(x_high, x_low, name, stem, sample_size=500):
    indices = np.random.choice(len(x_high), min(sample_size, len(x_high)), replace=False)
    d_high  = pairwise_distances(x_high[indices]).flatten()
    d_low   = pairwise_distances(x_low[indices]).flatten()
    d_high  = d_high / np.max(d_high)
    d_low   = d_low  / np.max(d_low)
    plt.figure(figsize=(6, 6))
    plt.scatter(d_high, d_low, alpha=0.1, s=1, color='teal')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)
    plt.title(f"Shepard Diagram: {name}")
    plt.xlabel("High-Dimensional Distance (Normalized)")
    plt.ylabel("Low-Dimensional Distance (Normalized)")
    plt.tight_layout()
    filename = os.path.join(shepard_dir, f"shepard_{stem}_{name.lower()}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

# ========================
# Main loop
# ========================

for embedding_model in embedding_models:
    provider         = embedding_model.split("/")[0]  # "sentence-transformers" or "Qwen"
    embedding_dir    = f"cache/{embedding_model}_embeddings"
    reduction_2d_dir = f"cache/{embedding_model}_reduced_2d"
    results_dir      = f"../results/viz_metrics/{provider}"
    shepard_dir      = f"../results/shepard_diagrams/{provider}"

    os.makedirs(reduction_2d_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(shepard_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Embedding model: {embedding_model}")
    print(f"{'='*60}")

    for theme, max_sub, depth, add_noise in configs:
        stem = build_stem(theme, max_sub, depth, add_noise)
        print(f"\n  --- Config: {stem} ---")

        embed_path = f"{embedding_dir}/{stem}_embed.npy"
        if not os.path.exists(embed_path):
            print(f"  Embeddings not found at {embed_path}, skipping.")
            continue

        x_high = np.load(embed_path)
        print(f"  Embeddings shape: {x_high.shape}")

        x_high_sub = x_high
        print(f"  Using full dataset ({x_high.shape[0]} points)")

        # compute 2D reductions
        reductions = {}
        reductions["PCA"] = load_or_compute_2d(
            "PCA", f"{reduction_2d_dir}/PCA_2d_{stem}.npy",
            lambda: skPCA(n_components=2, random_state=67).fit_transform(x_high_sub)
        )
        reductions["UMAP"] = load_or_compute_2d(
            "UMAP", f"{reduction_2d_dir}/UMAP_2d_{stem}.npy",
            lambda: umap_pkg.UMAP(n_components=2, min_dist=0.05, n_neighbors=10, random_state=67).fit_transform(x_high_sub)
        )
        reductions["PHATE"] = load_or_compute_2d(
            "PHATE", f"{reduction_2d_dir}/PHATE_2d_{stem}.npy",
            lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=2).fit_transform(x_high_sub)
        )
        reductions["PaCMAP"] = load_or_compute_2d(
            "PaCMAP", f"{reduction_2d_dir}/PaCMAP_2d_{stem}.npy",
            lambda: pacmap.PaCMAP(n_components=2, random_state=67).fit_transform(x_high_sub)
        )
        reductions["TriMAP"] = load_or_compute_2d(
            "TriMAP", f"{reduction_2d_dir}/TriMAP_2d_{stem}.npy",
            lambda: trimap.TRIMAP(n_dims=2).fit_transform(x_high_sub)
        )
        reductions["tSNE"] = load_or_compute_2d(
            "tSNE", f"{reduction_2d_dir}/tSNE_2d_{stem}.npy",
            lambda: np.array(cuTSNE(n_components=2).fit_transform(
                skPCA(n_components=50, random_state=67).fit_transform(x_high_sub)
            ))
        )

        # compute metrics
        stats = []
        for name, x_low_2d in reductions.items():
            t_score = trustworthiness(x_high_sub, x_low_2d, n_neighbors=15)
            c_score = compute_continuity(x_high_sub, x_low_2d, n_neighbors=15)
            d_high_flat = pairwise_distances(x_high_sub).flatten()
            d_low_flat  = pairwise_distances(x_low_2d).flatten()
            spearman_corr, _ = spearmanr(d_high_flat, d_low_flat)
            print(f"  Computing DEMaP for {name}...")
            demap_score = compute_demap(x_high_sub, x_low_2d)
            stats.append({
                "Method": name,
                "Trustworthiness": round(t_score, 4),
                "Continuity": round(c_score, 4),
                "Spearman Correlation": round(spearman_corr, 4),
                "DEMaP": round(demap_score, 4)
            })
            print(f"  {name}: T={t_score:.4f}, C={c_score:.4f}, Spearman={spearman_corr:.4f}, DEMaP={demap_score:.4f}")

        # save metrics CSV
        output_path = os.path.join(results_dir, f"viz_metrics_{stem}.csv")
        pd.DataFrame(stats).to_csv(output_path, index=False)
        print(f"  Saved metrics to {output_path}")

        # save Shepard diagrams
        for name, x_low_2d in reductions.items():
            f = plot_shepard(x_high_sub, x_low_2d, name, stem)
            print(f"  Saved Shepard: {f}")

print(f"\n{'='*60}")
print("All configs complete.")
print(f"{'='*60}")
