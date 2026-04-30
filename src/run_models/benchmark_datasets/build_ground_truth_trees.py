"""
Build ground truth hierarchical trees from benchmark datasets.

Creates multi-way trees (non-binary) from hierarchical category labels using anytree.
"""

import os
import sys
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from anytree import Node

# Path setup - same as eval_pipeline.py
target_folder = "src"
current_dir = os.path.dirname(os.path.abspath(__file__))

while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

os.chdir(current_dir)
sys.path.insert(0, current_dir)


# =====================================
# Dataset-Specific Loading Functions
# (Copied from eval_pipeline.py)
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

    return db


def load_arxiv():
    """Load and preprocess arXiv dataset."""
    arx = pd.read_csv("../data/arxiv/arxiv_clean.csv")
    arx = arx.dropna().reset_index(drop=True)
    return arx


def load_rcv1():
    """Load and preprocess RCV1 dataset."""
    rcv1 = pd.read_csv('../data/rcv1/rcv1.csv')

    # Rename columns with spaces to underscores
    rcv1 = rcv1.rename(columns={'category 0': 'category_0', 'category 1': 'category_1'})

    rcv1 = rcv1.drop_duplicates(subset='topic', keep=False).reset_index(drop=True)
    rcv1 = rcv1.drop_duplicates(subset='item_id', keep=False).reset_index(drop=True)

    rcv1 = rcv1.dropna().reset_index(drop=True)
    rcv1 = rcv1[rcv1['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

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
# Dataset Configuration
# =====================================

DATASET_CONFIGS = {
    "amazon": {"load_function": load_amazon, "depth": 3},
    "dbpedia": {"load_function": load_dbpedia, "depth": 3},
    "arxiv": {"load_function": load_arxiv, "depth": 2},
    "rcv1": {"load_function": load_rcv1, "depth": 2},
    "wos": {"load_function": load_wos, "depth": 2},
}


# =====================================
# Tree Building Functions
# =====================================

def build_ground_truth_tree(df, depth):
    """
    Build a multi-way tree from ground truth hierarchical labels using anytree.

    Iterates row by row, creating leaf nodes for each data point, then
    builds internal nodes bottom-up from finest to coarsest level.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with category_0, category_1, ... columns (coarsest to finest)
    depth : int
        Number of hierarchy levels

    Returns
    -------
    root : anytree.Node
        Root node of the tree (node.name is the node id)
    node_map : dict
        Mapping from node id (name) to anytree.Node
    """
    n_samples = len(df)

    # Get category columns (category_0 is coarsest, category_{depth-1} is finest)
    category_columns = [f'category_{i}' for i in range(depth) if f'category_{i}' in df.columns]
    n_levels = len(category_columns)

    if n_levels == 0:
        raise ValueError("No category columns found in dataframe")

    # Node ID counter - leaves are 0 to n_samples-1, internal nodes start at n_samples
    next_internal_id = n_samples

    # Create leaf nodes for each data point (row index = leaf node id)
    # Parent will be set later
    leaf_nodes = {i: Node(name=i) for i in range(n_samples)}
    node_map = dict(leaf_nodes)

    # Group leaves by their full category path (finest level)
    # category path = tuple of (cat_0, cat_1, ..., cat_{n_levels-1})
    finest_groups = defaultdict(list)

    for idx in range(n_samples):
        cat_path = tuple(df[col].iloc[idx] for col in category_columns)
        finest_groups[cat_path].append(idx)

    # Build internal nodes bottom-up
    # Start from finest level: group leaves by full category path
    current_level_nodes = {}  # cat_path -> anytree.Node

    for cat_path, leaf_indices in finest_groups.items():
        # Create internal node for this finest-level category group
        internal_node = Node(name=next_internal_id)
        next_internal_id += 1
        node_map[internal_node.name] = internal_node

        # Set leaf nodes as children (by assigning parent)
        for leaf_idx in leaf_indices:
            leaf_nodes[leaf_idx].parent = internal_node

        current_level_nodes[cat_path] = internal_node

    # Iterate upward through hierarchy levels (from level n_levels-2 down to 0)
    for level in range(n_levels - 2, -1, -1):
        # Group current level nodes by their parent category path (truncate to level+1)
        parent_groups = defaultdict(list)

        for cat_path, node in current_level_nodes.items():
            parent_path = cat_path[:level + 1]
            parent_groups[parent_path].append(node)

        # Create parent nodes
        next_level_nodes = {}

        for parent_path, child_nodes in parent_groups.items():
            if len(child_nodes) == 1:
                # Only one child - promote it up without creating new node
                next_level_nodes[parent_path] = child_nodes[0]
            else:
                # Multiple children - create new parent node
                parent_node = Node(name=next_internal_id)
                next_internal_id += 1
                node_map[parent_node.name] = parent_node

                for child in child_nodes:
                    child.parent = parent_node

                next_level_nodes[parent_path] = parent_node

        current_level_nodes = next_level_nodes

    # Create root node if needed (multiple top-level categories)
    if len(current_level_nodes) == 1:
        root = list(current_level_nodes.values())[0]
    else:
        root = Node(name=next_internal_id)
        node_map[root.name] = root
        for node in current_level_nodes.values():
            node.parent = root

    return root, node_map


def get_leaves(node):
    """
    Get all leaf node ids under a node using iterative DFS.

    Parameters
    ----------
    node : anytree.Node
        Starting node

    Returns
    -------
    leaves : list
        List of leaf node ids (names)
    """
    leaves = []
    stack = [node]

    while stack:
        current = stack.pop()
        if current.is_leaf:
            leaves.append(current.name)
        else:
            for child in current.children:
                stack.append(child)

    return leaves


def build_maps(root):
    """
    Build node map and parent map for an anytree in a single traversal.

    Parameters
    ----------
    root : anytree.Node
        Root of the tree

    Returns
    -------
    node_map : dict
        Mapping from node id (name) to anytree.Node
    parent_map : dict
        Mapping from node id to parent node id (root maps to None)
    """
    node_map = {}
    parent_map = {root.name: None}
    stack = [root]

    while stack:
        node = stack.pop()
        node_map[node.name] = node
        for child in node.children:
            parent_map[child.name] = node.name
            stack.append(child)

    return node_map, parent_map


GROUND_TRUTH_TREE_DIR = "cache/ground_truth_trees"


def save_ground_truth_tree(root, node_map, dataset_name):
    """
    Save a ground truth tree to cache/ground_truth_trees/.

    Parameters
    ----------
    root : anytree.Node
        Root node of the ground truth tree
    node_map : dict
        Mapping from node id to anytree.Node
    dataset_name : str
        Dataset name used as the filename key
    """
    os.makedirs(GROUND_TRUTH_TREE_DIR, exist_ok=True)
    path = os.path.join(GROUND_TRUTH_TREE_DIR, f"{dataset_name}_tree.pkl")
    with open(path, "wb") as f:
        pickle.dump({"root": root, "node_map": node_map}, f)
    print(f"Saved ground truth tree to {path}")


def load_cached_ground_truth_tree(dataset_name):
    """
    Load a cached ground truth tree from cache/ground_truth_trees/.

    Parameters
    ----------
    dataset_name : str
        Dataset name to look up

    Returns
    -------
    (root, node_map) if cache exists, else None
    """
    path = os.path.join(GROUND_TRUTH_TREE_DIR, f"{dataset_name}_tree.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded cached ground truth tree from {path}")
    return data["root"], data["node_map"]


def load_dataset_and_build_tree(dataset_name, use_cache=True):
    """
    Load a benchmark dataset and build its ground truth tree.

    Parameters
    ----------
    dataset_name : str
        One of: 'amazon', 'dbpedia', 'arxiv', 'rcv1', 'wos'
    use_cache : bool
        If True, load from cache if available (default: True)

    Returns
    -------
    root : anytree.Node
        Root node of the ground truth tree
    node_map : dict
        Mapping from node id to anytree.Node
    df : pd.DataFrame
        The loaded dataset
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    df = config["load_function"]()
    depth = config["depth"]

    if use_cache:
        cached = load_cached_ground_truth_tree(dataset_name)
        if cached is not None:
            root, node_map = cached
            return root, node_map, df

    root, node_map = build_ground_truth_tree(df, depth)
    save_ground_truth_tree(root, node_map, dataset_name)

    return root, node_map, df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build ground truth tree from benchmark dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset to build tree for')
    args = parser.parse_args()

    print(f"Building ground truth tree for {args.dataset}...")
    root, node_map, df = load_dataset_and_build_tree(args.dataset)

    print(f"Dataset size: {len(df)}")
    print(f"Total nodes: {len(node_map)}")
    print(f"Root node id: {root.name}")

    # Count leaves
    leaves = get_leaves(root)
    print(f"Leaf nodes: {len(leaves)}")

    # Print tree structure summary
    def count_children_at_depth(node, depth=0, counts=None):
        if counts is None:
            counts = defaultdict(list)
        counts[depth].append(len(node.children))
        for child in node.children:
            if not child.is_leaf:
                count_children_at_depth(child, depth + 1, counts)
        return counts

    counts = count_children_at_depth(root)
    print("\nTree structure summary (children per node at each depth):")
    for depth in sorted(counts.keys()):
        children_counts = counts[depth]
        print(f"  Depth {depth}: {len(children_counts)} nodes, avg children: {np.mean(children_counts):.1f}, "
              f"min: {min(children_counts)}, max: {max(children_counts)}")
