"""
dendrogram_purity.py

Monte Carlo estimation of Dendrogram Purity for hierarchical clustering
evaluation. Measures how well a predicted dendrogram groups same-class
points relative to ground truth labels.
"""

import numpy as np
from scipy.cluster.hierarchy import ClusterNode

from custom_packages.graph_utils import clusternode_to_anytree, get_leaves, build_maps, find_lca


def dendrogram_purity(tree, true_labels, n_samples=1000):
    # Convert ClusterNode to anytree if needed
    if isinstance(tree, ClusterNode):
        tree = clusternode_to_anytree(tree)

    # Build node id to node object mapping and parent map in single traversal
    node_map, parent_map = build_maps(tree)

    # get true cluster assignments
    true_clusters = np.unique(true_labels)

    cluster_counts = np.array([np.sum(true_labels == c) for c in true_clusters])

    # sort clusters by size descending and compute pair-weighted sampling probs
    # larger clusters are sampled more often since they contain more pairs
    sort_idx = np.argsort(cluster_counts)[::-1]
    true_clusters_sorted = true_clusters[sort_idx]
    cluster_counts_sorted = cluster_counts[sort_idx]

    weights = cluster_counts_sorted * (cluster_counts_sorted - 1) / 2  # C(k,2) pairs per cluster
    weights = weights / weights.sum()  # normalize to valid probability distribution

    scores = []

    for _ in range(n_samples):
        # sample a cluster proportional to its number of pairs, then pick 2 points from it
        c = np.random.choice(true_clusters_sorted, p=weights)
        cluster_idx = np.where(true_labels == c)[0]
        i, j = np.random.choice(cluster_idx, size=2, replace=False)

        # skip if either point is absent from the predicted tree (e.g. DC produced incomplete tree)
        if i not in parent_map or j not in parent_map:
            continue

        # find lowest common ancestor of i and j in the dendrogram
        lca_id = find_lca(i, j, parent_map)
        lca_node = node_map[lca_id]
        lca_leaves = get_leaves(lca_node)

        # purity = fraction of LCA's subtree that belongs to the same true cluster
        purity = np.sum(true_labels[lca_leaves] == c) / len(lca_leaves)
        scores.append(purity)

    if not scores:
        return float('nan')
    return float(np.mean(scores))