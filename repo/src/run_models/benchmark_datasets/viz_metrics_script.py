"""
Visualization Quality Metrics: Benchmark Datasets (Batch Script)

Runs Trustworthiness, Continuity, Spearman Correlation, and DEMaP
for all benchmark datasets. Results and Shepard diagrams are saved
to results/viz_metrics/{embedding_model}/.

2D reductions are computed on the full dataset. For large datasets
(> SUBSAMPLE_THRESHOLD points), metrics are computed on N_SUBSAMPLES
random subsamples of SUBSAMPLE_SIZE points and the CSV reports
mean ± std across subsamples.

Usage:
    # run all datasets, all models
    python src/run_models/benchmark_datasets/viz_metrics_script.py
    # run a single dataset
    python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset arxiv
    # run a single model
    python src/run_models/benchmark_datasets/viz_metrics_script.py --model sentence-transformers/all-MiniLM-L6-v2
    # run a single dataset + single model
    python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset arxiv --model Qwen/Qwen3-Embedding-0.6B
"""

import os
import sys
import argparse

# navigate to src/
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != 'src':
    parent = os.path.abspath(os.path.join(current_dir, '..'))
    if parent == current_dir:
        raise FileNotFoundError("src/ not found in directory tree.")
    current_dir = parent
os.chdir(current_dir)
sys.path.insert(0, current_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None,
                    help="Run a single dataset (e.g. arxiv). Omit to run all.")
parser.add_argument("--model", type=str, default=None,
                    help="Run a single embedding model (e.g. sentence-transformers/all-MiniLM-L6-v2). Omit to run all.")
args = parser.parse_args()

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

all_embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen3-Embedding-0.6B",
]
embedding_models = [args.model] if args.model else all_embedding_models

all_datasets = ["rcv1", "arxiv", "amazon", "dbpedia", "wos"]
datasets = [args.dataset] if args.dataset else all_datasets

SUBSAMPLE_THRESHOLD = 10_000   # use subsampling for datasets larger than this
SUBSAMPLE_SIZE      = 10_000   # points per subsample
N_SUBSAMPLES        = 30       # number of subsamples (mean ± std reported)

# ========================
# Helper functions
# ========================

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

def compute_metrics_once(x_high_sub, x_low_sub):
    t_score = trustworthiness(x_high_sub, x_low_sub, n_neighbors=15)
    c_score = compute_continuity(x_high_sub, x_low_sub, n_neighbors=15)
    d_high_flat = pairwise_distances(x_high_sub).flatten()
    d_low_flat  = pairwise_distances(x_low_sub).flatten()
    spearman_corr, _ = spearmanr(d_high_flat, d_low_flat)
    demap_score = compute_demap(x_high_sub, x_low_sub)
    return t_score, c_score, spearman_corr, demap_score

def plot_shepard(x_high, x_low, name, dataset, sample_size=500):
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
    filename = os.path.join(shepard_dir, f"shepard_{dataset}_{name.lower()}.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    return filename

# ========================
# Main loop
# ========================

cache_base = "intermediate_data" if os.path.isdir("intermediate_data") else "cache"

for embedding_model in embedding_models:
    provider         = embedding_model.split("/")[0]  # "sentence-transformers" or "Qwen"
    embedding_dir    = f"{cache_base}/{embedding_model}_embeddings"
    reduction_2d_dir = f"{cache_base}/{embedding_model}_reduced_2d"
    results_dir      = f"../results/viz_metrics/{provider}"
    shepard_dir      = f"../results/shepard_diagrams/{provider}"

    os.makedirs(reduction_2d_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(shepard_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Embedding model: {embedding_model}")
    print(f"{'='*60}")

    for dataset in datasets:
        print(f"\n  --- Dataset: {dataset} ---")

        embed_path = f"{embedding_dir}/{dataset}.npy"
        if not os.path.exists(embed_path):
            print(f"  Embeddings not found at {embed_path}, skipping.")
            continue

        x_high = np.load(embed_path)
        print(f"  Embeddings shape: {x_high.shape}")

        x_high_full = x_high
        print(f"  Using full dataset ({x_high.shape[0]} points) for 2D reductions")

        suffix = f"{dataset}_full{len(x_high_full)}"

        # compute 2D reductions on full dataset
        reductions = {}
        reductions["PCA"] = load_or_compute_2d(
            "PCA", f"{reduction_2d_dir}/PCA_2d_{suffix}.npy",
            lambda: skPCA(n_components=2, random_state=67).fit_transform(x_high_full)
        )
        reductions["UMAP"] = load_or_compute_2d(
            "UMAP", f"{reduction_2d_dir}/UMAP_2d_{suffix}.npy",
            lambda: umap_pkg.UMAP(n_components=2, min_dist=0.05, n_neighbors=10, random_state=67).fit_transform(x_high_full)
        )
        reductions["PHATE"] = load_or_compute_2d(
            "PHATE", f"{reduction_2d_dir}/PHATE_2d_{suffix}.npy",
            lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=2).fit_transform(x_high_full)
        )
        reductions["PaCMAP"] = load_or_compute_2d(
            "PaCMAP", f"{reduction_2d_dir}/PaCMAP_2d_{suffix}.npy",
            lambda: pacmap.PaCMAP(n_components=2, random_state=67).fit_transform(x_high_full)
        )
        reductions["TriMAP"] = load_or_compute_2d(
            "TriMAP", f"{reduction_2d_dir}/TriMAP_2d_{suffix}.npy",
            lambda: trimap.TRIMAP(n_dims=2).fit_transform(x_high_full)
        )
        reductions["tSNE"] = load_or_compute_2d(
            "tSNE", f"{reduction_2d_dir}/tSNE_2d_{suffix}.npy",
            lambda: np.array(cuTSNE(n_components=2).fit_transform(
                skPCA(n_components=50, random_state=67).fit_transform(x_high_full)
            ))
        )

        n_total = x_high_full.shape[0]
        use_subsampling = n_total > SUBSAMPLE_THRESHOLD

        if use_subsampling:
            print(f"  Dataset has {n_total} points — using {N_SUBSAMPLES}x subsamples of {SUBSAMPLE_SIZE} for metrics")
        else:
            print(f"  Dataset has {n_total} points — computing metrics on full dataset")

        # load existing cached metrics if available
        output_path = os.path.join(results_dir, f"viz_metrics_{dataset}.csv")
        if os.path.exists(output_path):
            cached_df = pd.read_csv(output_path)
            cached_methods = set(cached_df["Method"].tolist())
            stats = cached_df.to_dict("records")
            print(f"  Loaded cached metrics for: {cached_methods}")
        else:
            cached_methods = set()
            stats = []

        # compute metrics
        for name, x_low_2d in reductions.items():
            if name in cached_methods:
                print(f"  Skipping {name} (already cached)")
                continue
            print(f"  Computing metrics for {name}...")

            if use_subsampling:
                t_scores, c_scores, sp_scores, demap_scores = [], [], [], []
                for s in range(N_SUBSAMPLES):
                    idx = np.random.choice(n_total, SUBSAMPLE_SIZE, replace=False)
                    t, c, sp, dm = compute_metrics_once(x_high_full[idx], x_low_2d[idx])
                    t_scores.append(t)
                    c_scores.append(c)
                    sp_scores.append(sp)
                    demap_scores.append(dm)
                    if (s + 1) % 10 == 0:
                        print(f"    subsample {s+1}/{N_SUBSAMPLES} done")

                stats.append({
                    "Method":                   name,
                    "Trustworthiness_mean":      round(np.mean(t_scores),     4),
                    "Trustworthiness_std":       round(np.std(t_scores),      4),
                    "Continuity_mean":           round(np.mean(c_scores),     4),
                    "Continuity_std":            round(np.std(c_scores),      4),
                    "Spearman_Correlation_mean": round(np.mean(sp_scores),    4),
                    "Spearman_Correlation_std":  round(np.std(sp_scores),     4),
                    "DEMaP_mean":                round(np.mean(demap_scores), 4),
                    "DEMaP_std":                 round(np.std(demap_scores),  4),
                    "n_subsamples":              N_SUBSAMPLES,
                    "subsample_size":            SUBSAMPLE_SIZE,
                })
                print(f"  {name}: T={np.mean(t_scores):.4f}±{np.std(t_scores):.4f}, "
                      f"C={np.mean(c_scores):.4f}±{np.std(c_scores):.4f}, "
                      f"Spearman={np.mean(sp_scores):.4f}±{np.std(sp_scores):.4f}, "
                      f"DEMaP={np.mean(demap_scores):.4f}±{np.std(demap_scores):.4f}")
            else:
                t, c, sp, dm = compute_metrics_once(x_high_full, x_low_2d)
                stats.append({
                    "Method":                   name,
                    "Trustworthiness_mean":      round(t,  4),
                    "Trustworthiness_std":       0.0,
                    "Continuity_mean":           round(c,  4),
                    "Continuity_std":            0.0,
                    "Spearman_Correlation_mean": round(sp, 4),
                    "Spearman_Correlation_std":  0.0,
                    "DEMaP_mean":                round(dm, 4),
                    "DEMaP_std":                 0.0,
                    "n_subsamples":              1,
                    "subsample_size":            n_total,
                })
                print(f"  {name}: T={t:.4f}, C={c:.4f}, Spearman={sp:.4f}, DEMaP={dm:.4f}")

            # save after each method (incremental caching)
            pd.DataFrame(stats).to_csv(output_path, index=False)
            print(f"  Saved metrics to {output_path}")

        # save Shepard diagrams (sampled from full dataset)
        for name, x_low_2d in reductions.items():
            f = plot_shepard(x_high_full, x_low_2d, name, dataset)
            print(f"  Saved Shepard: {f}")

print(f"\n{'='*60}")
print("All datasets complete.")
print(f"{'='*60}")
