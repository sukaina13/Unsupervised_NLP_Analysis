"""
Generalized Benchmark Evaluation Pipeline

This script provides a unified pipeline for running dimensionality reduction and clustering
experiments on different benchmark datasets. The dataset is specified via command-line argument.

Usage:
    python eval_pipeline.py --dataset <dataset_name>

    Available datasets: amazon, dbpedia, arxiv, rcv1, wos

Example:
    python eval_pipeline.py --dataset dbpedia
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys
from dotenv import load_dotenv
import json
import argparse
load_dotenv()

target_folder = "src"

# Use __file__ to get the script's actual location, not the terminal's CWD
current_dir = os.path.dirname(os.path.abspath(__file__))

while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

os.chdir(current_dir)
sys.path.insert(0, current_dir)


# ===================
# Standard Libraries
# ===================
import importlib
import os
import re
from pathlib import Path
import warnings
from collections import defaultdict
import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# ===================
# Data Manipulation
# ===================
import numpy as np
import pandas as pd

# ====================
# Embeddings
# ====================
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Dimensionality Reduction
# ==========================
import phate
import pacmap
import trimap

# cuML GPU-accelerated dimensionality reduction
import cuml
from cuml.decomposition import PCA as cuPCA
from cuml.manifold import TSNE as cuTSNE
from cuml.manifold import UMAP as cuUMAP

# ========================
# Clustering
# ========================
from custom_packages.diffusion_condensation import DiffusionCondensation as dc
from scipy.cluster.hierarchy import fcluster, to_tree, linkage

# cuML GPU-accelerated clustering
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
from cuml.cluster import HDBSCAN as cuHDBSCAN


# ======================
# Evaluation Metrics
# ======================
from custom_packages.fowlkes_mallows import FowlkesMallows
from custom_packages.dendrogram_purity import dendrogram_purity
from custom_packages.lca_f1 import lca_f1
from custom_packages.graph_utils import clusternode_to_anytree, build_ground_truth_tree
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score

from tqdm import tqdm

# ==============
# Global Config
# ==============
np.random.seed(67)
warnings.filterwarnings("ignore")

# =====================================
# Dendrogram Purity Sampling Config
# =====================================
# Reload modules if needed
importlib.reload(phate)


# =====================================
# Dataset-Specific Loading Functions
# =====================================

def load_amazon():
    """Load and preprocess Amazon dataset."""
    amz_40 = pd.read_csv("../data/amazon/train_40k.csv")
    amz_10 = pd.read_csv("../data/amazon/val_10k.csv")

    amz = pd.concat([amz_40, amz_10])
    amz = amz.drop_duplicates(subset='Title', keep=False).reset_index(drop=True)
    amz = amz.drop_duplicates(subset='productId', keep=False).reset_index(drop=True)

    amz = amz.rename(columns={'Title': 'topic'})
    amz = amz.rename(columns={'Cat1': 'category_0'})
    amz = amz.rename(columns={'Cat2': 'category_1'})
    amz = amz.rename(columns={'Cat3': 'category_2'})

    amz = amz.dropna().reset_index(drop=True)
    amz = amz[amz['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

    amz.to_csv("../data/amazon/amz_data.csv")

    return amz


def load_dbpedia():
    """Load and preprocess DBpedia dataset."""
    def clean_dbpedia(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    db = pd.read_csv('../data/dbpedia/DBPEDIA_test.csv')

    db = db.rename(columns={"text": "topic"})
    db = db.rename(columns={"l1": "category_0"})
    db = db.rename(columns={"l2": "category_1"})
    db = db.rename(columns={"l3": "category_2"})

    db['topic'] = db['topic'].astype(str).apply(clean_dbpedia)

    print(db.iloc[0])

    return db


def load_arxiv():
    """Load and preprocess arXiv dataset."""
    arx = pd.read_csv("../data/arxiv/arxiv_clean.csv")
    arx = arx.dropna().reset_index(drop=True)
    return arx


def load_rcv1():
    """Load and preprocess RCV1 dataset."""
    rcv1 = pd.read_csv('../data/rcv1/rcv1.csv')

    rcv1 = rcv1.drop_duplicates(subset='topic', keep=False).reset_index(drop=True)
    rcv1 = rcv1.drop_duplicates(subset='item_id', keep=False).reset_index(drop=True)

    rcv1 = rcv1.dropna().reset_index(drop=True)
    rcv1 = rcv1[rcv1['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

    rcv1.to_csv("../data/rcv1/rcv1.csv")

    return rcv1


def load_wos():
    """Load and preprocess Web of Science dataset."""
    wos = pd.read_excel('../data/WebOfScience/Data.xlsx')

    new = []
    for i, row in wos.iterrows():
        result = {}
        result['topic'] = str(row['keywords'])
        result['category_0'] = row['Domain']
        result['category_1'] = row['area']
        new.append(result)

    wos = pd.DataFrame(new)

    return wos


# =====================================
# Dataset Configuration Dictionary
# =====================================

DATASET_CONFIGS = {
    "amazon": {
        "load_function": load_amazon,
        "depth": 3,
        "short": "amz",
        "results_filename": "amazon_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["Raw", "PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", "TriMAP"],
    },
    "dbpedia": {
        "load_function": load_dbpedia,
        "depth": 3,
        "short": "db",
        "results_filename": "db_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["Raw", "PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", 'TriMAP'],
    },
    "arxiv": {
        "load_function": load_arxiv,
        "depth": 2,
        "short": "arx",
        "results_filename": "arxiv_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["Raw", "PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", 'TriMAP'],
    },
    "rcv1": {
        "load_function": load_rcv1,
        "depth": 2,
        "short": "rcv1",
        "results_filename": "rcv1_clustering_scores.csv",
        "batch_size": 2,
        "reduction_methods": ["Raw", "PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", 'TriMAP'],
    },
    "wos": {
        "load_function": load_wos,
        "depth": 2,
        "short": "wos",
        "results_filename": "wos_clustering_scores.csv",
        "batch_size": 64,
        "reduction_methods": ["Raw", "PHATE", "PCA", "UMAP", "tSNE", "PaCMAP", 'TriMAP'],
    },
}


# =====================================
# Core Pipeline Functions
# =====================================

def get_embeddings(texts, model_id, batch_size=32):
    """
    Generate embeddings using SentenceTransformer.

    Args:
        texts: List or Series of text inputs
        model_id: Model identifier for SentenceTransformer
        batch_size: Batch size for encoding

    Returns:
        numpy array of embeddings
    """
    print("Using device:", device)
    print(f"Number of texts: {len(texts)}")

    model = SentenceTransformer(
        model_id,
        model_kwargs={"attn_implementation": "sdpa", "device_map": "auto"} if "Qwen" in model_id else {},
        tokenizer_kwargs={"padding_side": "left"} if "Qwen" in model_id else {},
        device="cuda:0"
    )

    # Print token statistics
    tok = model.tokenizer(texts.tolist(), truncation=False, padding=False)
    lens = [len(x) for x in tok['input_ids']]
    print(f"Total tokens: {sum(lens):,}")
    print(f"Avg tokens: {sum(lens)/len(lens):.1f}")
    print(f"Max tokens: {max(lens)}")

    print("Generating embeddings...")
    embeddings = model.encode(
        texts.to_list(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda:0"
    )

    return embeddings


def apply_dimensionality_reduction(embeddings, reduction_dir, embed_filename, reduction_methods):
    """
    Apply dimensionality reduction methods to embeddings.

    Args:
        embeddings: Input embeddings (numpy array)
        reduction_dir: Directory to save reduced embeddings
        embed_filename: Filename prefix for saved embeddings
        reduction_methods: List of reduction methods to apply

    Returns:
        Dictionary mapping method names to reduced embeddings
    """
    embedding_methods = {}

    # Raw embeddings (no dimensionality reduction) — pass through directly
    if "Raw" in reduction_methods:
        print("Using raw embeddings (no dimensionality reduction)...")
        embedding_methods["Raw"] = embeddings

    # Define all possible reduction tasks
    all_reduction_tasks = {
        "PHATE": {
            "path": f"{reduction_dir}/PHATE_{embed_filename}.npy",
            "run": lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None).fit_transform(embeddings)
        },
        "PCA": {
            "path": f"{reduction_dir}/PCA_{embed_filename}.npy",
            "run": lambda: cuPCA(n_components=300).fit_transform(embeddings)
        },
        "UMAP": {
            "path": f"{reduction_dir}/UMAP_{embed_filename}.npy",
            "run": lambda: cuUMAP(n_components=300, min_dist=.05, n_neighbors=10).fit_transform(embeddings)
        },
        "tSNE": {
            "path": f"{reduction_dir}/tSNE_{embed_filename}.npy",
            "run": lambda: cuTSNE(n_components=2).fit_transform(embeddings)
        },
        "PaCMAP": {
            "path": f"{reduction_dir}/PaCMAP_{embed_filename}.npy",
            "run": lambda: pacmap.PaCMAP(n_components=300, random_state=67).fit_transform(embeddings)
        },
        "TriMAP": {
            "path": f"{reduction_dir}/TriMAP_{embed_filename}.npy",
            "run": lambda: trimap.TRIMAP(n_dims=300).fit_transform(embeddings)
        }
    }

    # Filter to only requested methods
    reduction_tasks = {k: v for k, v in all_reduction_tasks.items() if k in reduction_methods}

    for method_name, task in reduction_tasks.items():
        if os.path.exists(task["path"]):
            print(f"Loading cached {method_name} from {task['path']}...")
            result = np.load(task["path"])
        else:
            print(f"Running {method_name}...")
            result = task["run"]()

            # Handle cuML GPU to CPU conversion if necessary
            if hasattr(result, 'to_output'):
                result = result.to_output('numpy')
            elif not isinstance(result, np.ndarray):
                result = np.array(result)

            np.save(task["path"], result)
            print(f"Saved {method_name} to {task['path']}")

        embedding_methods[method_name] = result

    return embedding_methods



def cluster_combo(embedding_model, dim_reduction_method, cluster_method, reduced_embeddings,
                   cluster_levels, topic_dict, short,
                   gt_tree_root=None, gt_node_map=None):
    embed_data = reduced_embeddings[embedding_model][dim_reduction_method]
    combo_scores = {"FM": [], "Rand": [], "ARI": [], "AMI": [], "Dendrogram Purity": [], "LCA_F1": []}

    print(f"\n{'='*60}")
    print(f"Processing Embedding Method: {dim_reduction_method}")
    print(f"Clustering Method: {cluster_method}")
    print(f"Embedding shape: {embed_data.shape}")
    print(f"{'='*60}")

    method_prefix = {"Agglomerative": "Agg", "HDBSCAN": "HDB", "DC": "DC"}[cluster_method]
    scores_dir = os.path.join(f"cache/{embedding_model}_scores", short, dim_reduction_method)
    os.makedirs(scores_dir, exist_ok=True)

    def score_paths(level):
        return {
            "fm":   os.path.join(scores_dir, f"{method_prefix}_{level}_fm.npy"),
            "rand": os.path.join(scores_dir, f"{method_prefix}_{level}_rand.npy"),
            "ari":  os.path.join(scores_dir, f"{method_prefix}_{level}_ari.npy"),
            "ami":  os.path.join(scores_dir, f"{method_prefix}_{level}_ami.npy"),
            "dp":   os.path.join(scores_dir, f"{method_prefix}_{level}_dp.npy"),
            "lca":  os.path.join(scores_dir, f"{method_prefix}_{level}_lca_f1.npy"),
        }


    # Short-circuit: if all scores for every level are cached, load and return immediately.
    if all(all(os.path.exists(p) for p in score_paths(level).values()) for level in cluster_levels):
        print(f"All scores cached for {dim_reduction_method} / {cluster_method}, loading from cache...")
        for level in cluster_levels:
            paths = score_paths(level)
            combo_scores["FM"].append(float(np.load(paths["fm"])))
            combo_scores["Rand"].append(float(np.load(paths["rand"])))
            combo_scores["ARI"].append(float(np.load(paths["ari"])))
            combo_scores["AMI"].append(float(np.load(paths["ami"])))
            combo_scores["Dendrogram Purity"].append(float(np.load(paths["dp"])))
            combo_scores["LCA_F1"].append(float(np.load(paths["lca"])))
        return embedding_model, dim_reduction_method, cluster_method, combo_scores

    # Build the full linkage tree once per embedding-clustering method combination.
    # All three methods now use fcluster (or DC's get_labels) to cut at each level.
    tree = None
    Z = None
    dc_model = None

    if cluster_method == "Agglomerative":
        linkage_path = os.path.join(
            f"cache/{embedding_model}_linkage", short, dim_reduction_method,
            "Agg_linkage.npy"
        )

        if os.path.exists(linkage_path):
            print(f"Loading cached Agglomerative linkage from {linkage_path}")
            Z = np.load(linkage_path)
            if len(Z) + 1 != len(embed_data):
                print(f"Cached linkage size mismatch ({len(Z)+1} vs {len(embed_data)}), recomputing...")
                Z = linkage(embed_data, method='ward')
                np.save(linkage_path, Z)
        else:
            print("Building ward linkage tree for Agglomerative Clustering...")
            Z = linkage(embed_data, method='ward')
            os.makedirs(os.path.dirname(linkage_path), exist_ok=True)
            np.save(linkage_path, Z)
            print(f"Saved linkage matrix to {linkage_path}")

        tree, _ = to_tree(Z, rd=True)

    elif cluster_method == "HDBSCAN":
        linkage_path = os.path.join(
            f"cache/{embedding_model}_linkage", short, dim_reduction_method,
            "HDBSCAN_linkage.npy"
        )

        if os.path.exists(linkage_path):
            print(f"Loading cached HDBSCAN linkage from {linkage_path}")
            Z = np.load(linkage_path)
            if len(Z) + 1 != len(embed_data):
                print(f"Cached linkage size mismatch ({len(Z)+1} vs {len(embed_data)}), recomputing...")
                model = cuHDBSCAN(min_cluster_size=5, min_samples=1)
                model.fit(embed_data)
                Z = model.single_linkage_tree_.to_numpy()
                np.save(linkage_path, Z)
            tree, _ = to_tree(Z, rd=True)
        else:
            print("Using cuML HDBSCAN (GPU)...")
            model = cuHDBSCAN(min_cluster_size=5, min_samples=1)
            model.fit(embed_data)

            Z = model.single_linkage_tree_.to_numpy()

            os.makedirs(os.path.dirname(linkage_path), exist_ok=True)
            np.save(linkage_path, Z)
            print(f"Saved linkage matrix to {linkage_path}")

            tree, _ = to_tree(Z, rd=True)

    elif cluster_method == "DC":
        print(f"Running Diffusion Condensation for {dim_reduction_method}")
        dc_model = dc(min_clusters=1, max_iterations=5000, k=10, alpha=3)
        dc_model.fit(embed_data)
        tree = dc_model.tree_

    # Convert full predicted tree to anytree once; used for both dendrogram purity and LCA-F1.
    pred_tree = None
    if tree is not None:
        pred_tree = clusternode_to_anytree(tree)

    # Iterate through cluster levels
    for level in cluster_levels:
        print(f"Testing cluster level: {level}")

        paths = score_paths(level)

        # Load cached scores for this level if available
        if all(os.path.exists(p) for p in paths.values()):
            print(f"Loading cached scores for level {level}...")
            combo_scores["FM"].append(float(np.load(paths["fm"])))
            combo_scores["Rand"].append(float(np.load(paths["rand"])))
            combo_scores["ARI"].append(float(np.load(paths["ari"])))
            combo_scores["AMI"].append(float(np.load(paths["ami"])))
            combo_scores["Dendrogram Purity"].append(float(np.load(paths["dp"])))
            combo_scores["LCA_F1"].append(float(np.load(paths["lca"])))
            continue

        # Compute labels for this level
        if cluster_method in ("Agglomerative", "HDBSCAN"):
            labels = fcluster(Z, level, criterion='maxclust')
            print(f"{cluster_method} fcluster cut at {level}. Unique labels: {len(np.unique(labels))}")
        elif cluster_method == "DC":
            dc_model.get_labels(n_clusters=level)
            labels = dc_model.labels_
            print(f"DC tree cut at {level}. Unique labels: {len(np.unique(labels))}")

        # Match to closest ground truth level
        available_levels = np.array(sorted(topic_dict.keys()))
        closest_level = min(available_levels, key=lambda k: abs(k - level))
        print(f"Ground truth: Using closest level {closest_level} (requested: {level})")

        topic_series = topic_dict[closest_level]
        valid_idx = (~pd.isna(topic_series))
        target_lst = topic_series[valid_idx]
        label_lst = labels[valid_idx]

        try:
            fm_score = FowlkesMallows.Bk({level: target_lst}, {level: label_lst})[level]['FM']
        except Exception:
            fm_score = np.nan
            print("WARNING: FM score computation failed!")

        rand = rand_score(target_lst, label_lst)
        ari = adjusted_rand_score(target_lst, label_lst)
        ami = adjusted_mutual_info_score(target_lst, label_lst)

        if pred_tree is not None:
            dp = dendrogram_purity(pred_tree, topic_series)
        else:
            dp = np.nan
        if pred_tree is not None and gt_tree_root is not None:
            lca_f1_score = lca_f1(pred_tree, gt_tree_root, topic_series)
        else:
            lca_f1_score = np.nan

        # Cache all scores
        np.save(paths["fm"],   np.array(fm_score))
        np.save(paths["rand"], np.array(rand))
        np.save(paths["ari"],  np.array(ari))
        np.save(paths["ami"],  np.array(ami))
        np.save(paths["dp"],   np.array(dp))
        np.save(paths["lca"],  np.array(lca_f1_score))
        lca_str = f"{lca_f1_score:.4f}" if not np.isnan(lca_f1_score) else "NaN"
        print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}, "
              f"DP: {dp:.4f}, LCA_F1: {lca_str}")

        combo_scores["FM"].append(fm_score)
        combo_scores["Rand"].append(rand)
        combo_scores["ARI"].append(ari)
        combo_scores["AMI"].append(ami)
        combo_scores["Dendrogram Purity"].append(dp)
        combo_scores["LCA_F1"].append(lca_f1_score)

    return embedding_model, dim_reduction_method, cluster_method, combo_scores



def run_pipeline(dataset_name):
    """
    Run the complete evaluation pipeline for a specified dataset.

    Args:
        dataset_name: Name of the dataset to process (e.g., 'amazon', 'dbpedia')
    """
    # Validate dataset name
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*80}")
    print(f"Running pipeline for dataset: {dataset_name.upper()}")
    print(f"{'='*80}\n")

    # Load dataset using dataset-specific function
    print("Loading dataset...")
    data = config["load_function"]()
    print(f"Dataset shape: {data.shape}\n")

    # Embedding models to test
    embedding_model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    reduced_embeddings = {}  # Will store {model_name: {reduction_method: embeddings}}

    # Prepare data once (same for all embedding models)
    topic_data = data.reset_index(drop=True)

    # Build ground truth tree
    print("Building ground truth tree...")
    gt_tree_root, gt_node_map = build_ground_truth_tree(data, config["depth"])

    # Build topic_dict from ground truth categories
    topic_dict = {}
    for col in topic_data.columns:
        if re.match(r'^category_\d+$', col):
            unique_count = len(topic_data[col].unique())
            topic_dict[unique_count] = np.array(topic_data[col])

    # Determine cluster levels from hierarchy depth
    depth = config["depth"]
    print(f"Depth: {depth}")
    print(f"Building cluster levels by counting unique categories at each level...\n")

    cluster_levels = []
    for i in reversed(range(0, depth)):
        if f'category_{i}' in topic_data.columns:
            unique_count = len(topic_data[f'category_{i}'].unique())
            print(f"Level {i} (category_{i}): {unique_count} unique categories")
            cluster_levels.append(unique_count)

    print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

    # Process each embedding model
    for embedding_model in embedding_model_names:
        print(f"\n{'='*60}")
        print(f"Processing embedding model: {embedding_model}")
        print(f"{'='*60}\n")

        os.makedirs(f'cache/{embedding_model}_results', exist_ok=True)

        reduction_dir = f"cache/{embedding_model}_reduced_embeddings"
        os.makedirs(reduction_dir, exist_ok=True)

        embedding_dir = f"cache/{embedding_model}_embeddings"
        os.makedirs(embedding_dir, exist_ok=True)

        embedding_path = f"{embedding_dir}/{config['short']}.npy"

        # Generate or load embeddings
        if os.path.exists(embedding_path):
            print(f"Loading existing embeddings from {embedding_path}")
            embedding_list = np.load(embedding_path)
        else:
            print("Generating embeddings...")
            embedding_list = get_embeddings(
                data['topic'],
                model_id=embedding_model,
                batch_size=config["batch_size"]
            )
            np.save(embedding_path, embedding_list)
            print(f"Saved embeddings to {embedding_path}")

        embeddings = np.array(embedding_list)

        # Apply dimensionality reduction
        embedding_methods_for_model = apply_dimensionality_reduction(
            embeddings=embeddings,
            reduction_dir=reduction_dir,
            embed_filename=config["short"],
            reduction_methods=config["reduction_methods"]
        )

        # Store the final dict for the global reduced_embeddings
        reduced_embeddings[embedding_model] = embedding_methods_for_model

    # Run clustering and evaluation
    scores_all = defaultdict(lambda: defaultdict(list))

    combo_params = [
        (embedding_model, dim_reduction_method, cluster_method)
        for embedding_model in reduced_embeddings.keys()
        for dim_reduction_method in reduced_embeddings[embedding_model].keys()
        for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]
    ]

    # Run each combo sequentially
    combo_results = []
    for embedding_model, dim_reduction_method, cluster_method in tqdm(combo_params, desc="Processing embedding-clustering combos"):
        result = cluster_combo(
            embedding_model,
            dim_reduction_method,
            cluster_method,
            reduced_embeddings,
            cluster_levels,
            topic_dict,
            short=config["short"],
            gt_tree_root=gt_tree_root,
            gt_node_map=gt_node_map,
        )
        combo_results.append(result)

    for embedding_model, dim_reduction_method, cluster_method, combo_scores in combo_results:
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["FM"] = combo_scores["FM"]
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["Rand"] = combo_scores["Rand"]
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["ARI"] = combo_scores["ARI"]
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["AMI"] = combo_scores["AMI"]
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["Dendrogram Purity"] = combo_scores["Dendrogram Purity"]
        scores_all[(embedding_model, dim_reduction_method, cluster_method)]["LCA_F1"] = combo_scores["LCA_F1"]

    print(f"\n{'='*60}")
    print("All clustering and evaluation complete!")
    print(f"{'='*60}")

    # Save results to CSV
    rows = []

    for (embedding_model, dim_reduction_method, cluster_method), score_dict in scores_all.items():
        n_levels = len(score_dict["FM"])
        for i in range(n_levels):
            rows.append({
                "embedding_model": embedding_model,
                "reduction_method": dim_reduction_method,
                "cluster_method": cluster_method,
                "level": cluster_levels[i],
                "FM": score_dict["FM"][i],
                "Rand": score_dict["Rand"][i],
                "ARI": score_dict["ARI"][i],
                "AMI": score_dict["AMI"][i],
                "Dendrogram_Purity": score_dict["Dendrogram Purity"][i],
                "LCA_F1": score_dict["LCA_F1"][i],
            })

    # Create DataFrame
    scores_df = pd.DataFrame(rows)

    # Sort for easier viewing
    print(scores_df)
    print(scores_df.columns)
    scores_df = scores_df.sort_values(
        by=["embedding_model", "reduction_method", "cluster_method", "level"]
    ).reset_index(drop=True)

    # Save results
    os.makedirs("../results/clustering/benchmark", exist_ok=True)
    results_path = f"../results/clustering/benchmark/{config['results_filename']}"
    scores_df.to_csv(results_path, index=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Pipeline complete for dataset: {dataset_name.upper()}\n")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation pipeline on specified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Available datasets:
            amazon    - Amazon product categories (3 levels)
            dbpedia   - DBpedia ontology topics (3 levels)
            arxiv     - arXiv paper categories (2 levels)
            rcv1      - Reuters RCV1 news categories (2 levels)
            wos       - Web of Science publications (2 levels)

            Example usage:
            python eval_pipeline.py --dataset dbpedia
            python eval_pipeline.py --dataset amazon
                    """
                )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to process'
    )

    args = parser.parse_args()

    # Run pipeline for specified dataset
    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()
