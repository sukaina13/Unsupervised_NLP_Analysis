"""
RCV1 Hyperparameter Sweep

Sweeps dimensionality reduction hyperparameters for the RCV1 dataset and saves
results in a format compatible with notebooks/parameter_selection.ipynb.

Each row in the output CSV corresponds to one
(embedding_model, reduction_method, reduction_params, cluster_method, cluster_params, level)
combination, with columns: FM, Rand, ARI, AMI, Dendrogram_Purity, LCA_F1.

Usage:
    python src/run_models/benchmark_datasets/rcv1_param_sweep.py
"""

import os
import sys
import re
import json
import hashlib
import warnings
import pickle
import argparse
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

target_folder = "src"
current_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir
os.chdir(current_dir)
sys.path.insert(0, current_dir)

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import fcluster, to_tree, linkage
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from tqdm import tqdm

import phate
import pacmap
import cuml
from cuml.decomposition import PCA as cuPCA
from cuml.manifold import TSNE as cuTSNE
from cuml.manifold import UMAP as cuUMAP
from cuml.cluster import HDBSCAN as cuHDBSCAN

from custom_packages.fowlkes_mallows import FowlkesMallows
from custom_packages.dendrogram_purity import dendrogram_purity
from custom_packages.lca_f1 import lca_f1
from custom_packages.graph_utils import clusternode_to_anytree
from run_models.benchmark_datasets.build_ground_truth_trees import build_ground_truth_tree

warnings.filterwarnings("ignore")
np.random.seed(67)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────
# Parameter grids
# ─────────────────────────────────────────

UMAP_GRID = [
    {"n_components": nc, "n_neighbors": nn, "min_dist": md}
    for nc in [50, 100, 300]
    for nn in [5, 15, 30]
    for md in [0.01, 0.1, 0.5]
]

PHATE_GRID = [
    {"n_components": nc, "decay": d}
    for nc in [50, 100, 300]
    for d in [10, 20, 40]
]

PCA_GRID = [
    {"n_components": nc}
    for nc in [50, 100, 300]
]

PaCMAP_GRID = [
    {"n_components": nc}
    for nc in [50, 100, 300]
]

TSNE_GRID = [
    {"n_components": 2}
]

REDUCTION_GRIDS = {
    "UMAP":   UMAP_GRID,
    "PHATE":  PHATE_GRID,
    "PCA":    PCA_GRID,
    "PaCMAP": PaCMAP_GRID,
    "tSNE":   TSNE_GRID,
}

CLUSTER_METHODS = ["Agglomerative", "HDBSCAN"]

EMBEDDING_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def params_to_key(params: dict) -> str:
    """Stable short string key from a params dict for use in filenames."""
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def load_rcv1():
    rcv1 = pd.read_csv("../data/rcv1/rcv1.csv")
    rcv1 = rcv1.drop_duplicates(subset="topic", keep=False).reset_index(drop=True)
    rcv1 = rcv1.drop_duplicates(subset="item_id", keep=False).reset_index(drop=True)
    rcv1 = rcv1.dropna().reset_index(drop=True)
    rcv1 = rcv1[rcv1["topic"].apply(lambda x: isinstance(x, str) and x.strip() != "")].reset_index(drop=True)
    return rcv1


def get_embeddings(texts, model_id, batch_size=8):
    model = SentenceTransformer(
        model_id,
        model_kwargs={"attn_implementation": "sdpa", "device_map": "auto"} if "Qwen" in model_id else {},
        tokenizer_kwargs={"padding_side": "left"} if "Qwen" in model_id else {},
    )
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def run_reduction(method: str, params: dict, embeddings: np.ndarray, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"  Loading cached {method} {params} from {cache_path}")
        return np.load(cache_path)

    print(f"  Running {method} {params}...")
    if method == "UMAP":
        result = cuUMAP(**params).fit_transform(embeddings)
    elif method == "PHATE":
        result = phate.PHATE(n_jobs=-2, random_state=67, t="auto", n_pca=None, **params).fit_transform(embeddings)
    elif method == "PCA":
        result = cuPCA(**params).fit_transform(embeddings)
    elif method == "PaCMAP":
        result = pacmap.PaCMAP(random_state=67, **params).fit_transform(embeddings)
    elif method == "tSNE":
        result = cuTSNE(**params).fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    if hasattr(result, "to_output"):
        result = result.to_output("numpy")
    elif not isinstance(result, np.ndarray):
        result = np.array(result)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, result)
    return result


def build_linkage(cluster_method: str, embed_data: np.ndarray, cache_path: str):
    if os.path.exists(cache_path):
        print(f"  Loading cached linkage from {cache_path}")
        Z = np.load(cache_path)
    else:
        if cluster_method == "Agglomerative":
            print("  Building ward linkage...")
            Z = linkage(embed_data, method="ward")
        elif cluster_method == "HDBSCAN":
            print("  Running cuML HDBSCAN...")
            model = cuHDBSCAN(min_cluster_size=5, min_samples=1)
            model.fit(embed_data)
            Z = model.single_linkage_tree_.to_numpy()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, Z)

    tree, _ = to_tree(Z, rd=True)
    return Z, tree


def evaluate_combo(
    embed_data, cluster_method, cluster_levels, topic_dict,
    gt_tree_root, linkage_cache_path
):
    """Run one (reduced_embedding, cluster_method) combo at all cluster levels."""
    Z, tree = build_linkage(cluster_method, embed_data, linkage_cache_path)
    pred_tree = clusternode_to_anytree(tree)

    rows = []
    for level in cluster_levels:
        labels = fcluster(Z, level, criterion="maxclust")

        available_levels = np.array(sorted(topic_dict.keys()))
        closest_level = min(available_levels, key=lambda k: abs(k - level))
        topic_series = topic_dict[closest_level]
        valid_idx = ~pd.isna(topic_series)
        target_lst = topic_series[valid_idx]
        label_lst = labels[valid_idx]

        try:
            fm_score = FowlkesMallows.Bk({level: target_lst}, {level: label_lst})[level]["FM"]
        except Exception:
            fm_score = np.nan

        rand  = rand_score(target_lst, label_lst)
        ari   = adjusted_rand_score(target_lst, label_lst)
        ami   = adjusted_mutual_info_score(target_lst, label_lst)
        dp    = dendrogram_purity(pred_tree, topic_series)
        lca   = lca_f1(pred_tree, gt_tree_root, topic_series) if gt_tree_root is not None else np.nan

        rows.append({
            "level":              level,
            "FM":                 fm_score,
            "Rand":               rand,
            "ARI":                ari,
            "AMI":                ami,
            "Dendrogram_Purity":  dp,
            "LCA_F1":             lca,
        })

    return rows


# ─────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────

def main():
    print("Loading RCV1...")
    data = load_rcv1()
    print(f"RCV1 shape: {data.shape}")

    # Ground truth tree (built once, row indices = leaf IDs)
    gt_tree_path = "cache/ground_truth_trees/rcv1_tree.pkl"
    if os.path.exists(gt_tree_path):
        with open(gt_tree_path, "rb") as f:
            saved = pickle.load(f)
        gt_tree_root = saved["root"]
        print("Loaded cached ground truth tree.")
    else:
        print("Building ground truth tree...")
        gt_df = data.rename(columns={c: c.replace(" ", "_") for c in data.columns})
        gt_tree_root, _ = build_ground_truth_tree(gt_df, depth=2)
        os.makedirs(os.path.dirname(gt_tree_path), exist_ok=True)
        with open(gt_tree_path, "wb") as f:
            pickle.dump({"root": gt_tree_root, "node_map": {}}, f)

    # topic_dict: unique-count → label array
    topic_dict = {}
    for col in data.columns:
        if re.match(r"^category[_ ]\d+$", col):
            topic_dict[len(data[col].unique())] = np.array(data[col])

    cluster_levels = sorted(topic_dict.keys())
    print(f"Cluster levels: {cluster_levels}")

    all_rows = []

    for embedding_model in EMBEDDING_MODELS:
        print(f"\n{'='*60}\nEmbedding model: {embedding_model}\n{'='*60}")

        embed_dir = f"cache/{embedding_model}_embeddings"
        os.makedirs(embed_dir, exist_ok=True)
        embed_path = f"{embed_dir}/rcv1.npy"

        if os.path.exists(embed_path):
            print("Loading cached embeddings...")
            embeddings = np.load(embed_path)
        else:
            print("Generating embeddings...")
            embeddings = get_embeddings(data["topic"], embedding_model)
            np.save(embed_path, embeddings)

        reduction_cache_dir = f"cache/{embedding_model}_sweep_reduced/rcv1"
        linkage_cache_dir   = f"cache/{embedding_model}_sweep_linkage/rcv1"
        os.makedirs(reduction_cache_dir, exist_ok=True)
        os.makedirs(linkage_cache_dir,   exist_ok=True)

        # Build list of all (method, params) combos
        combos = [
            (method, params)
            for method, grid in REDUCTION_GRIDS.items()
            for params in grid
        ]

        for reduction_method, reduction_params in tqdm(combos, desc=f"Reductions ({embedding_model.split('/')[-1]})"):
            key = params_to_key(reduction_params)
            reduction_cache = f"{reduction_cache_dir}/{reduction_method}_{key}.npy"

            try:
                reduced = run_reduction(reduction_method, reduction_params, embeddings, reduction_cache)
            except Exception as e:
                print(f"  ERROR in {reduction_method} {reduction_params}: {e}")
                continue

            for cluster_method in CLUSTER_METHODS:
                linkage_cache = f"{linkage_cache_dir}/{reduction_method}_{key}_{cluster_method}.npy"

                try:
                    level_rows = evaluate_combo(
                        reduced, cluster_method, cluster_levels,
                        topic_dict, gt_tree_root, linkage_cache,
                    )
                except Exception as e:
                    print(f"  ERROR in clustering {cluster_method}: {e}")
                    continue

                for row in level_rows:
                    row.update({
                        "embedding_model":   embedding_model,
                        "reduction_method":  reduction_method,
                        "reduction_params":  str(reduction_params),
                        "cluster_method":    cluster_method,
                        "cluster_params":    "{}",
                    })
                    all_rows.append(row)

    results_df = pd.DataFrame(all_rows, columns=[
        "embedding_model", "reduction_method", "reduction_params",
        "cluster_method", "cluster_params", "level",
        "FM", "Rand", "ARI", "AMI", "Dendrogram_Purity", "LCA_F1",
    ])

    os.makedirs("../results", exist_ok=True)
    out_path = "../results/rcv1_param_sweep_scores.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSweep complete. Results saved to {out_path}")
    print(f"Total rows: {len(results_df)}")


if __name__ == "__main__":
    main()
