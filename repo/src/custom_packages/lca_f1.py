"""
lca_f1.py

Monte Carlo estimation of LCA-F1 for hierarchical clustering evaluation.
Compares the lowest common ancestor (LCA) subtrees of sampled point pairs
in the predicted vs. ground truth hierarchies.
"""

import numpy as np
from scipy.cluster.hierarchy import ClusterNode

from custom_packages.graph_utils import (
    clusternode_to_anytree, get_leaves, build_maps, find_lca
)


def lca_f1(pred_tree, gt_tree, true_labels, n_samples=1000):
    """
    Monte Carlo estimation of LCA-F1 for hierarchical clustering evaluation.

    For each sample: picks a ground-truth cluster proportional to its number of pairs,
    selects two random points from it, finds their LCA in both the predicted and ground
    truth trees, then computes per-sample F1 from the overlap of the two LCA subtrees.

    Parameters
    ----------
    pred_tree : ClusterNode or anytree.Node
        Root of the predicted hierarchy.
    gt_tree : ClusterNode or anytree.Node
        Root of the ground truth hierarchy.
    true_labels : array-like of shape (n_samples,)
        Ground-truth flat cluster labels for each data point (index = leaf node id).
    n_samples : int, optional
        Number of Monte Carlo samples (default 1000).

    Returns
    -------
    float
        Average LCA-F1 score over all sampled pairs (0–1).
    """
    if isinstance(pred_tree, ClusterNode):
        pred_tree = clusternode_to_anytree(pred_tree)
    if isinstance(gt_tree, ClusterNode):
        gt_tree = clusternode_to_anytree(gt_tree)

    true_labels = np.asarray(true_labels)

    pred_node_map, pred_parent_map = build_maps(pred_tree)
    gt_node_map, gt_parent_map = build_maps(gt_tree)

    true_clusters = np.unique(true_labels)
    cluster_counts = np.array([np.sum(true_labels == c) for c in true_clusters])

    sort_idx = np.argsort(cluster_counts)[::-1]
    true_clusters_sorted = true_clusters[sort_idx]
    cluster_counts_sorted = cluster_counts[sort_idx]

    weights = cluster_counts_sorted * (cluster_counts_sorted - 1) / 2
    weights = weights / weights.sum()

    f1_scores = []

    for _ in range(n_samples):
        c = np.random.choice(true_clusters_sorted, p=weights)
        cluster_idx = np.where(true_labels == c)[0]
        i, j = np.random.choice(cluster_idx, size=2, replace=False)

        if i not in pred_parent_map or j not in pred_parent_map:
            continue
        if i not in gt_parent_map or j not in gt_parent_map:
            continue

        pred_lca_id = find_lca(i, j, pred_parent_map)
        gt_lca_id = find_lca(i, j, gt_parent_map)

        pred_leaves = set(get_leaves(pred_node_map[pred_lca_id]))
        gt_leaves = set(get_leaves(gt_node_map[gt_lca_id]))

        intersection = len(pred_leaves & gt_leaves)
        precision = intersection / len(pred_leaves)
        recall = intersection / len(gt_leaves)

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    if not f1_scores:
        return float('nan')
    return float(np.mean(f1_scores))
