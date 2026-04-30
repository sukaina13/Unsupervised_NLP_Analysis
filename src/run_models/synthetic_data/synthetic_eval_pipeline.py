# ========================
# Environment Configuration
# ========================
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import sys

target_folder = "src"

current_dir = os.path.dirname(os.path.abspath(__file__))

while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

os.chdir(current_dir)
sys.path.insert(0, current_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================
# Standard Libraries
# ========================
import importlib
import re
import warnings

# ========================
# Data Manipulation
# ========================
import numpy as np
import pandas as pd

# ===============================
# Machine Learning & Clustering
# ===============================
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from scipy.cluster.hierarchy import fcluster, to_tree, linkage
from collections import defaultdict
import torch

# ===========================
# Dimensionality Reduction
# ===========================
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

# cuML GPU-accelerated clustering
from cuml.cluster import HDBSCAN as cuHDBSCAN

# ========================
# NLP & Transformers
# ========================
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Parallel Processing
# ========================
from tqdm import tqdm

# ========================
# Evaluation Metrics
# ========================
from custom_packages.fowlkes_mallows import FowlkesMallows
from custom_packages.dendrogram_purity import dendrogram_purity
from custom_packages.lca_f1 import lca_f1
from custom_packages.graph_utils import clusternode_to_anytree, apted_distance, build_ground_truth_tree


# ===================
# Global Config
# ===================
np.random.seed(67)
warnings.filterwarnings("ignore")
importlib.reload(phate)

# ===================
# Embedding Functions
# ===================
def get_embeddings(texts, model):
    print("Using device:", device)
    model = SentenceTransformer(model, device=device)
    print("Generating embeddings...")
    embeddings = model.encode(
        texts.to_list(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


# ===================
# Clustering Runner
# ===================
def safe_run_combo(embedding_model, embed_name, cluster_method, embed_data, cluster_levels, topic_dict, theme, t, max_sub, depth, synonyms, branching, add_noise, gt_tree_root=None):
    """Run clustering on reduced embeddings and evaluate against ground truth."""
    combo_scores = {"FM": [], "Rand": [], "ARI": [], "AMI": [], "Dendrogram Purity": [], "LCA_F1": [], "TED": None}
    try:
        print(f"\n{'='*60}")
        print(f"Processing Embedding Method: {embed_name}")
        print(f"Clustering Method: {cluster_method}")
        print(f"Embedding shape: {embed_data.shape}")
        print(f"{'='*60}")

        if float(add_noise) > 0:
            path_prefix = f"{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}"
        else:
            path_prefix = f"{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}"

        method_prefix = {"Agglomerative": "Agg", "HDBSCAN": "HDB", "DC": "DC"}[cluster_method]
        scores_dir = os.path.join(f"cache/{embedding_model}_scores", path_prefix, embed_name)
        os.makedirs(scores_dir, exist_ok=True)
        ted_cache_path = os.path.join(scores_dir, f"{method_prefix}_ted.npy")

        def score_paths(level):
            return {
                "fm":        os.path.join(scores_dir, f"{method_prefix}_{level}_fm.npy"),
                "rand":      os.path.join(scores_dir, f"{method_prefix}_{level}_rand.npy"),
                "ari":       os.path.join(scores_dir, f"{method_prefix}_{level}_ari.npy"),
                "ami":       os.path.join(scores_dir, f"{method_prefix}_{level}_ami.npy"),
                "dp":        os.path.join(scores_dir, f"{method_prefix}_{level}_dp.npy"),
                "lca":       os.path.join(scores_dir, f"{method_prefix}_{level}_lca_f1.npy"),
            }

        # Short-circuit: if all scores for every level are cached, load and return immediately.
        if (all(all(os.path.exists(p) for p in score_paths(level).values()) for level in cluster_levels)
                and os.path.exists(ted_cache_path)):
            print(f"All scores cached for {embed_name} / {cluster_method}, loading from cache...")
            combo_scores["TED"] = float(np.load(ted_cache_path))
            for level in cluster_levels:
                paths = score_paths(level)
                combo_scores["FM"].append(float(np.load(paths["fm"])))
                combo_scores["Rand"].append(float(np.load(paths["rand"])))
                combo_scores["ARI"].append(float(np.load(paths["ari"])))
                combo_scores["AMI"].append(float(np.load(paths["ami"])))
                combo_scores["Dendrogram Purity"].append(float(np.load(paths["dp"])))
                combo_scores["LCA_F1"].append(float(np.load(paths["lca"])))
            return embedding_model, embed_name, cluster_method, combo_scores

        tree = None
        Z = None
        dc_model = None

        if cluster_method == "Agglomerative":
            linkage_path = os.path.join(
                f"cache/{embedding_model}_linkage", path_prefix, embed_name,
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
                f"cache/{embedding_model}_linkage", path_prefix, embed_name,
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
            print(f"Running Diffusion Condensation for {embed_name}")
            dc_model = dc(min_clusters=1, max_iterations=5000, k=10, alpha=3)
            dc_model.fit(embed_data)
            tree = dc_model.tree_

        # Convert full predicted tree to anytree once; used for dendrogram purity, LCA-F1, and TED.
        pred_tree = clusternode_to_anytree(tree) if tree is not None else None

        # Tree Edit Distance (computed once per method, not per level)
        if pred_tree is not None and gt_tree_root is not None:
            if os.path.exists(ted_cache_path):
                print(f"Loading cached TED from {ted_cache_path}")
                combo_scores["TED"] = float(np.load(ted_cache_path))
            else:
                print("Computing Tree Edit Distance...")
                ted_score = apted_distance(pred_tree, gt_tree_root)
                np.save(ted_cache_path, np.array(ted_score))
                combo_scores["TED"] = ted_score
                print(f"TED: {ted_score:.1f}")
        else:
            combo_scores["TED"] = np.nan

        for level in cluster_levels:
            print(f"Testing cluster level: {level}")

            paths = score_paths(level)

            if all(os.path.exists(p) for p in paths.values()):
                print(f"Loading cached scores for level {level}...")
                combo_scores["FM"].append(float(np.load(paths["fm"])))
                combo_scores["Rand"].append(float(np.load(paths["rand"])))
                combo_scores["ARI"].append(float(np.load(paths["ari"])))
                combo_scores["AMI"].append(float(np.load(paths["ami"])))
                combo_scores["Dendrogram Purity"].append(float(np.load(paths["dp"])))
                combo_scores["LCA_F1"].append(float(np.load(paths["lca"])))
                continue

            if cluster_method in ("Agglomerative", "HDBSCAN"):
                labels = fcluster(Z, level, criterion='maxclust')
                print(f"{cluster_method} fcluster cut at {level}. Unique labels: {len(np.unique(labels))}")
            elif cluster_method == "DC":
                dc_model.get_labels(n_clusters=level)
                labels = dc_model.labels_
                print(f"DC tree cut at {level}. Unique labels: {len(np.unique(labels))}")

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

        return embedding_model, embed_name, cluster_method, combo_scores
    except Exception as e:
        print(f"Error in combo ({embedding_model}, {embed_name}, {cluster_method}): {e}")
        return embedding_model, embed_name, cluster_method, combo_scores


# ========================
# Config
# ========================

CONFIGS = [
    # (theme, max_sub, depth, add_noise)
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.0),
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.25),
    ("Energy_Ecosystems_and_Humans",         5, 3, 0.5),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.0),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.25),
    ("Energy_Ecosystems_and_Humans",         3, 5, 0.5),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.0),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.25),
    ("Offshore_energy_impacts_on_fisheries", 5, 3, 0.5),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.0),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.25),
    ("Offshore_energy_impacts_on_fisheries", 3, 5, 0.5),
]

T = 1.0
SYNONYMS = 0
BRANCHING = "random"


# ====================
# Main Pipeline
# ====================

def run_eval_pipeline(theme, max_sub, depth, add_noise, t=T, synonyms=SYNONYMS, branching=BRANCHING):
    print(f"\n{'='*80}")
    print(f"Running eval pipeline: {theme} | depth={depth} | max_sub={max_sub} | noise={add_noise}")
    print(f"{'='*80}\n")

    filename = f'../data/synthetic/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Data file not found: {filename}\nRun generate.py first.")

    print(f"Reading {filename}...")
    topic_data_original = pd.read_csv(filename)

    topic_data_original = topic_data_original.dropna(subset=[f'category {depth - 1}']).reset_index(drop=True)
    deepest_col = f'category {depth - 1}'
    topic_data_original = topic_data_original[
        topic_data_original['topic'] != topic_data_original[deepest_col]
    ].reset_index(drop=True)
    print(f"Filtered to {len(topic_data_original)} rows with full depth-{depth} labels and non-trivial topics.")

    embedding_model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    embedding_models = {}

    shuffle_idx = np.random.RandomState(seed=67).permutation(len(topic_data_original))
    topic_data = topic_data_original.iloc[shuffle_idx].reset_index(drop=True)

    topic_dict = {}
    for col in topic_data.columns:
        if re.match(r'^category \d+$', col):
            unique_count = len(topic_data[col].unique())
            topic_dict[unique_count] = np.array(topic_data[col])

    print(f"Depth: {depth}")
    cluster_levels = []
    for i in reversed(range(0, depth)):
        unique_count = len(topic_data[f'category {i}'].unique())
        print(f"Level {i} (category {i}): {unique_count} unique categories")
        cluster_levels.append(unique_count)
    print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

    # Process each embedding model
    for embedding_model in embedding_model_names:
        print(f"\n{'='*60}")
        print(f"Processing embedding model: {embedding_model}")
        print(f"{'='*60}\n")

        os.makedirs(f'cache/{embedding_model}_results', exist_ok=True)
        os.makedirs(f"cache/{embedding_model}_embeddings", exist_ok=True)

        if float(add_noise) > 0:
            embed_file = f'cache/{embedding_model}_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy'
        else:
            embed_file = f'cache/{embedding_model}_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy'

        if not os.path.exists(embed_file):
            embedding_list = get_embeddings(topic_data_original['topic'], model=embedding_model)
            np.save(embed_file, embedding_list)
        else:
            embedding_list = np.load(embed_file)
            if len(embedding_list) != len(topic_data_original):
                print(f"Cached embeddings size mismatch ({len(embedding_list)} vs {len(topic_data_original)}), regenerating...")
                embedding_list = get_embeddings(topic_data_original['topic'], model=embedding_model)
                np.save(embed_file, embedding_list)

        reduction_dir = f"cache/{embedding_model}_reduced_embeddings"
        os.makedirs(reduction_dir, exist_ok=True)

        data = np.array(embedding_list)[shuffle_idx]
        embeddings = np.array(data)

        if float(add_noise) > 0:
            embed_filename = f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}"
        else:
            embed_filename = f"{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}"

        reduction_tasks = {
            "PHATE": {
                "path": f"{reduction_dir}/PHATE_{embed_filename}_embed.npy",
                "run": lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None).fit_transform(data),
            },
            "PCA": {
                "path": f"{reduction_dir}/PCA_{embed_filename}_embed.npy",
                "run": lambda: cuPCA(n_components=300).fit_transform(embeddings),
            },
            "UMAP": {
                "path": f"{reduction_dir}/UMAP_{embed_filename}_embed.npy",
                "run": lambda: cuUMAP(n_components=300, min_dist=.05, n_neighbors=10).fit_transform(embeddings),
            },
            "tSNE": {
                "path": f"{reduction_dir}/tSNE_{embed_filename}_embed.npy",
                "run": lambda: cuTSNE(n_components=2).fit_transform(embeddings),
            },
            "PaCMAP": {
                "path": f"{reduction_dir}/PaCMAP_{embed_filename}_embed.npy",
                "run": lambda: pacmap.PaCMAP(n_components=300, random_state=67).fit_transform(embeddings),
            },
            "TriMAP": {
                "path": f"{reduction_dir}/TriMAP_{embed_filename}_embed.npy",
                "run": lambda: trimap.TRIMAP(n_dims=300).fit_transform(embeddings),
            },
        }

        embedding_methods_for_model = {"Raw": embeddings}
        for method_name, task in reduction_tasks.items():
            if os.path.exists(task["path"]):
                print(f"Loading cached {method_name} from {task['path']}...")
                result = np.load(task["path"])
            else:
                print(f"Running {method_name}...")
                result = task["run"]()
                if hasattr(result, 'to_output'):
                    result = result.to_output('numpy')
                elif not isinstance(result, np.ndarray):
                    result = np.array(result)
                np.save(task["path"], result)
                print(f"Saved {method_name} to {task['path']}")
            embedding_methods_for_model[method_name] = result

        embedding_models[embedding_model] = embedding_methods_for_model

    print("Building ground truth tree for synthetic data...")
    gt_df = topic_data.rename(columns={f'category {i}': f'category_{i}' for i in range(depth)})
    gt_tree_root, _ = build_ground_truth_tree(gt_df, depth)
    print(f"Ground truth tree built. Root id: {gt_tree_root.name}")

    scores_all = defaultdict(lambda: defaultdict(list))

    combo_params = [
        (embedding_model, embed_name, cluster_method)
        for embedding_model in embedding_models.keys()
        for embed_name in embedding_models[embedding_model].keys()
        for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]
    ]

    combo_results = []
    for embedding_model, embed_name, cluster_method in tqdm(combo_params, desc="Processing embedding-clustering combos"):
        embed_data = embedding_models[embedding_model][embed_name]
        result = safe_run_combo(embedding_model, embed_name, cluster_method, embed_data, cluster_levels, topic_dict, theme, t, max_sub, depth, synonyms, branching, add_noise, gt_tree_root=gt_tree_root)
        combo_results.append(result)

    for embedding_model, embed_name, cluster_method, combo_scores in combo_results:
        scores_all[(embedding_model, embed_name, cluster_method)]["FM"] = combo_scores["FM"]
        scores_all[(embedding_model, embed_name, cluster_method)]["Rand"] = combo_scores["Rand"]
        scores_all[(embedding_model, embed_name, cluster_method)]["ARI"] = combo_scores["ARI"]
        scores_all[(embedding_model, embed_name, cluster_method)]["AMI"] = combo_scores["AMI"]
        scores_all[(embedding_model, embed_name, cluster_method)]["Dendrogram Purity"] = combo_scores["Dendrogram Purity"]
        scores_all[(embedding_model, embed_name, cluster_method)]["LCA_F1"] = combo_scores["LCA_F1"]
        scores_all[(embedding_model, embed_name, cluster_method)]["TED"] = combo_scores["TED"]

    print(f"\n{'='*60}")
    print("All clustering and evaluation complete!")
    print(f"{'='*60}")

    rows = []
    for (embedding_model, embed_name, cluster_method), score_dict in scores_all.items():
        n_levels = len(score_dict["FM"])
        for i in range(n_levels):
            rows.append({
                "embedding_model": embedding_model,
                "reduction_method": embed_name,
                "cluster_method": cluster_method,
                "level": cluster_levels[i],
                "FM": score_dict["FM"][i],
                "Rand": score_dict["Rand"][i],
                "ARI": score_dict["ARI"][i],
                "AMI": score_dict["AMI"][i],
                "Dendrogram_Purity": score_dict["Dendrogram Purity"][i],
                "LCA_F1": score_dict["LCA_F1"][i],
                "TED": score_dict["TED"],
            })

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(by=["embedding_model", "reduction_method", "cluster_method", "level"]).reset_index(drop=True)

    os.makedirs("../results/clustering/synthetic", exist_ok=True)
    if float(add_noise) > 0:
        output_file = f"../results/clustering/synthetic/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_clustering_scores.csv"
    else:
        output_file = f"../results/clustering/synthetic/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_clustering_scores.csv"

    scores_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


TOPIC_MAP = {
    "energy":    "Energy_Ecosystems_and_Humans",
    "fisheries": "Offshore_energy_impacts_on_fisheries",
}

MAX_SUB_MAP = {3: 5, 5: 3}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic data evaluation pipeline.")
    parser.add_argument("--topic",  choices=["energy", "fisheries"], default=None,
                        help="Topic theme: 'energy' or 'fisheries'. Omit to run all configs.")
    parser.add_argument("--depth",  type=int, choices=[3, 5], default=None,
                        help="Hierarchy depth: 3 or 5. Omit to run all configs.")
    parser.add_argument("--noise",  type=float, choices=[0.0, 0.25, 0.5], default=None,
                        help="Noise level: 0, 0.25, or 0.5. Omit to run all configs.")
    args = parser.parse_args()

    if args.topic is not None or args.depth is not None or args.noise is not None:
        if not (args.topic and args.depth and args.noise is not None):
            parser.error("--topic, --depth, and --noise must all be specified together.")
        theme   = TOPIC_MAP[args.topic]
        max_sub = MAX_SUB_MAP[args.depth]
        run_eval_pipeline(theme, max_sub, args.depth, args.noise)
    else:
        for theme, max_sub, depth, add_noise in CONFIGS:
            run_eval_pipeline(theme, max_sub, depth, add_noise)
