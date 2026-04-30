"""
graph_utils.py

Tree utility functions shared across evaluation metrics. Handles conversion
between scipy ClusterNode dendrograms and anytree Node trees, leaf extraction,
LCA (lowest common ancestor) lookup, and APTED tree edit distance computation.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx
from scipy.cluster.hierarchy import ClusterNode
from anytree import Node
from apted import APTED, Config


class AnyTreeAPTEDConfig(Config):
    """APTED config for anytree.Node trees.

    Rename cost is always 0 — predicted tree internal node names are arbitrary,
    so only structural differences (insertions/deletions, cost 1 each) are counted.
    """
    def rename(self, node1, node2):
        return 0

    def children(self, node):
        return list(node.children)


def apted_distance(tree1, tree2):
    """Compute APTED tree edit distance between two anytree.Node trees."""
    return APTED(tree1, tree2, AnyTreeAPTEDConfig()).compute_edit_distance()


def clusternode_to_anytree(cluster_node):
    """Convert scipy ClusterNode (binary tree) to anytree Node (multi-way tree).

    Uses iterative approach to avoid recursion depth issues.
    """
    root = Node(name=cluster_node.id)
    stack = [(cluster_node, root)]

    while stack:
        cn, an_parent = stack.pop()

        if not cn.is_leaf():
            if cn.left is not None:
                left_node = Node(name=cn.left.id, parent=an_parent)
                stack.append((cn.left, left_node))
            if cn.right is not None:
                right_node = Node(name=cn.right.id, parent=an_parent)
                stack.append((cn.right, right_node))

    return root


def anytree_to_children_list(root):
    """Convert anytree to a 2D children-list using pre-order traversal.

    Returns a list indexed by node ID where each entry is the list of
    that node's children IDs. Indices between 0 and max node ID with no
    node will be empty lists.
    """
    max_id = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.name > max_id:
            max_id = node.name
        for child in node.children:
            stack.append(child)

    result = [[] for _ in range(max_id + 1)]

    stack = [root]
    while stack:
        node = stack.pop()
        result[node.name] = [child.name for child in node.children]
        for child in reversed(node.children):
            stack.append(child)

    return result


def get_leaves(node):
    """Get all leaf node ids under a node using iterative DFS."""
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
    """Build node map and parent map in a single traversal."""
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


def get_ancestors(node_id, parent_map):
    """Get list of ancestors from node to root (inclusive) using parent pointers."""
    ancestors = []
    current = node_id
    while current is not None:
        ancestors.append(current)
        current = parent_map[current]
    return ancestors


def find_lca(i, j, parent_map):
    """Find LCA by comparing ancestor sets."""
    ancestors_i = get_ancestors(i, parent_map)
    ancestors_j_set = set(get_ancestors(j, parent_map))

    for ancestor in ancestors_i:
        if ancestor in ancestors_j_set:
            return ancestor
    return None


def build_ground_truth_tree(df, depth):
    """Build a multi-way anytree from hierarchical ground-truth labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns category_0 (coarsest) through category_{depth-1} (finest).
    depth : int
        Number of hierarchy levels.

    Returns
    -------
    root : anytree.Node
    node_map : dict  — node id → anytree.Node; leaf ids are row indices 0..n-1
    """
    n_samples = len(df)
    category_columns = [f'category_{i}' for i in range(depth) if f'category_{i}' in df.columns]
    n_levels = len(category_columns)

    if n_levels == 0:
        raise ValueError("No category columns found in dataframe")

    next_internal_id = n_samples
    leaf_nodes = {i: Node(name=i) for i in range(n_samples)}
    node_map = dict(leaf_nodes)

    finest_groups = defaultdict(list)
    for idx in range(n_samples):
        cat_path = tuple(df[col].iloc[idx] for col in category_columns)
        finest_groups[cat_path].append(idx)

    current_level_nodes = {}
    for cat_path, leaf_indices in finest_groups.items():
        internal_node = Node(name=next_internal_id)
        next_internal_id += 1
        node_map[internal_node.name] = internal_node
        for leaf_idx in leaf_indices:
            leaf_nodes[leaf_idx].parent = internal_node
        current_level_nodes[cat_path] = internal_node

    for level in range(n_levels - 2, -1, -1):
        parent_groups = defaultdict(list)
        for cat_path, node in current_level_nodes.items():
            parent_groups[cat_path[:level + 1]].append(node)

        next_level_nodes = {}
        for parent_path, child_nodes in parent_groups.items():
            if len(child_nodes) == 1:
                next_level_nodes[parent_path] = child_nodes[0]
            else:
                parent_node = Node(name=next_internal_id)
                next_internal_id += 1
                node_map[parent_node.name] = parent_node
                for child in child_nodes:
                    child.parent = parent_node
                next_level_nodes[parent_path] = parent_node

        current_level_nodes = next_level_nodes

    if len(current_level_nodes) == 1:
        root = list(current_level_nodes.values())[0]
    else:
        root = Node(name=next_internal_id)
        node_map[root.name] = root
        for node in current_level_nodes.values():
            node.parent = root

    return root, node_map
