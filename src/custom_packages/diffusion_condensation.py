"""
diffusion_condensation.py

GPU-accelerated Diffusion Condensation clustering algorithm.
Iteratively applies a diffusion operator to compress the data geometry until
clusters condense into stable regions. Uses cuML for nearest-neighbor and
pairwise distance computations.
"""

from sklearn.preprocessing import normalize
import sklearn
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
import scipy
from cuml.neighbors import NearestNeighbors
from cuml.metrics import pairwise_distances
import cupy
import cupyx


class DiffusionCondensation:
    def __init__(self,
                 k=5,
                 alpha=2,
                 epsilon_scale=0.99,
                 merge_threshold=1e-3,
                 merge_threshold_end=None,  # NEW
                 min_clusters=5,
                 max_iterations=1000,
                 t=1,
                 data_dependent_epsilon=True,
                 symmetric_kernel=False,
                 bandwidth_norm="max",
                 k_end=None,
                 t_end=None,
                 alpha_end=None):

        self.k = k
        self.k_end = k_end
        self.alpha = alpha
        self.alpha_end = alpha_end
        self.epsilon_scale = epsilon_scale
        self.merge_threshold = merge_threshold
        self.merge_threshold_end = merge_threshold_end  # NEW
        self.min_clusters = min_clusters
        self.max_iterations = max_iterations
        self.t = t
        self.t_end = t_end
        self.data_dependent_epsilon = data_dependent_epsilon
        self.symmetric_kernel = symmetric_kernel
        self.bandwidth_norm = bandwidth_norm
        self.cluster_function = None
        self.epsilon = None
        self.labels_ = None
        self.cluster_tree = None
        self.tree_ = None
        self.node_list_ = None
        self.n_samples_ = None  # Store original number of data points


    def diffusion_operator(self, data):
        """Create diffusion operator"""
        # Generate the k-NN graph adjacency matrix using cuML
        knn = NearestNeighbors(n_neighbors = self.k)
        knn.fit(data)
        distances, indices = knn.kneighbors(data)

        distances = cupy.asnumpy(distances).reshape(data.shape[0] * self.k)
        indices = cupy.asnumpy(indices).reshape(data.shape[0] * self.k)
        indptr = np.arange(0, (self.k * data.shape[0]) + 1, self.k)

        knn_graph = scipy.sparse.csr_matrix(
            (distances, indices, indptr),
            shape=(data.shape[0], data.shape[0])
        )
        knn_graph_adjacency = knn_graph.maximum(knn_graph.T)

        knn_graph_adjacency = normalize(knn_graph_adjacency, norm=self.bandwidth_norm, axis=1, copy=False)

        knn_graph_adjacency = -knn_graph_adjacency.power(self.alpha)
        np.exp(knn_graph_adjacency.data, out=knn_graph_adjacency.data)

        if self.symmetric_kernel:
            knn_graph_adjacency = 0.5*(knn_graph_adjacency + knn_graph_adjacency.T)
        knn_graph_adjacency = normalize(knn_graph_adjacency, norm='l1', axis=1, copy=False)

        return knn_graph_adjacency

    def diffusion_condensation(self, data):
        """Run Diffusion Condensation"""
        if self.k >= data.shape[0]:
            self.k = data.shape[0] - 1

        P = self.diffusion_operator(data)
        for _ in np.arange(self.t):
            data = P @ data

        return data

    def chunked_pairwise_distances(self, data_gpu, chunk_size=10000):
        n = data_gpu.shape[0]
        distances = cupy.zeros((n, n), dtype=cupy.float32)
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                distances[i:end_i, j:end_j] = pairwise_distances(
                    data_gpu[i:end_i], data_gpu[j:end_j], metric='euclidean'
                )
        return cupy.asnumpy(distances)

    def merge_data_points(self, data, row_to_cluster, cluster_sizes, next_cluster_id):
        """
        Merges data points that are within the merge_threshold distance of each other.
        Creates new cluster indices for merged clusters instead of relabeling.

        Args:
            data: current data array
            row_to_cluster: list mapping each row index to its cluster ID
            cluster_sizes: dict mapping cluster ID to number of original points
            next_cluster_id: next available cluster ID for merged clusters

        Returns:
            new_data: condensed data array
            new_row_to_cluster: updated mapping from row index to cluster ID
            cluster_sizes: updated cluster sizes dict
            next_cluster_id: updated next cluster ID
            merges: list of (cluster_a, cluster_b, new_cluster_id, distance, size) tuples
        """
        data_gpu = cupy.asarray(data)
        distance_matrix = self.chunked_pairwise_distances(data_gpu)
        numpoints = distance_matrix.shape[0]

        merged = set()  # indices of rows that have been merged into another
        merges = []  # merge events: (cluster_a, cluster_b, new_cluster_id, distance, size)

        # Process from highest index to lowest
        for j in range(numpoints - 1, 0, -1):
            if j in merged:
                continue

            # Find closest point with index < j that hasn't been merged
            min_dist = float('inf')
            target = -1
            for i in range(j):
                if i not in merged and distance_matrix[j, i] < min_dist:
                    min_dist = distance_matrix[j, i]
                    target = i

            if target >= 0 and min_dist < self.merge_threshold:
                # Merge row j into row target
                cluster_j = row_to_cluster[j]
                cluster_target = row_to_cluster[target]

                new_size = cluster_sizes[cluster_j] + cluster_sizes[cluster_target]

                # Record merge: clusters cluster_j and cluster_target merge into next_cluster_id
                merges.append((cluster_j, cluster_target, next_cluster_id, min_dist, new_size))

                # Update: target row now represents the new merged cluster
                row_to_cluster[target] = next_cluster_id
                cluster_sizes[next_cluster_id] = new_size

                # Mark row j as merged
                merged.add(j)

                next_cluster_id += 1

        # Build new data array and row_to_cluster for kept rows
        indices_to_keep = [i for i in range(numpoints) if i not in merged]
        new_data = data[indices_to_keep]
        new_row_to_cluster = [row_to_cluster[i] for i in indices_to_keep]

        if merges:
            return new_data, new_row_to_cluster, cluster_sizes, next_cluster_id, merges
        else:
            return new_data, new_row_to_cluster, cluster_sizes, next_cluster_id, None


    def fit(self, data, prev_cluster_tree=None, prev_data=None):
        n = data.shape[0]
        self.n_samples_ = n  # Store original number of data points

        if prev_cluster_tree is not None and prev_data is not None:
            self.cluster_tree = prev_cluster_tree
            data = prev_data
            num_clusters = data.shape[0]
            iterations = len(prev_cluster_tree)
            # Reconstruct state from previous cluster_tree
            next_cluster_id = n + len(prev_cluster_tree)
            # Rebuild cluster_sizes from merges
            cluster_sizes = {i: 1 for i in range(n)}
            for cluster_a, cluster_b, new_id, dist, size in prev_cluster_tree:
                cluster_sizes[new_id] = size
            # row_to_cluster needs to be reconstructed - for simplicity, derive from remaining clusters
            # This is an approximation; full reconstruction would require tracking active clusters
            row_to_cluster = list(range(num_clusters))
        else:
            num_clusters = data.shape[0]
            # row_to_cluster[i] = cluster ID that data row i represents
            # Initially, each row represents its own cluster (leaf node)
            row_to_cluster = list(range(n))
            # cluster_sizes tracks number of original points in each cluster
            cluster_sizes = {i: 1 for i in range(n)}
            # next_cluster_id for merged clusters (starts at n, leaf nodes are 0 to n-1)
            next_cluster_id = n
            # cluster_tree stores merge events: (cluster_a, cluster_b, new_cluster_id, distance, size)
            self.cluster_tree = []
            iterations = 0

        while iterations < self.max_iterations and num_clusters > self.min_clusters:
            self.k = int(self.interpolate_param(self.k, self.k_end, iterations, self.max_iterations))
            self.t = int(self.interpolate_param(self.t, self.t_end, iterations, self.max_iterations))
            self.alpha = self.interpolate_param(self.alpha, self.alpha_end, iterations, self.max_iterations)
            self.merge_threshold = self.interpolate_param(self.merge_threshold, self.merge_threshold_end, iterations, self.max_iterations)

            data = self.diffusion_condensation(data)
            data, row_to_cluster, cluster_sizes, next_cluster_id, merges = self.merge_data_points(
                data, row_to_cluster, cluster_sizes, next_cluster_id
            )

            if merges is not None:
                self.cluster_tree.extend(merges)

            num_clusters = data.shape[0]
            iterations += 1

        # Build ClusterNode tree by iterating backwards through cluster_tree
        self._build_cluster_tree(n)
            
    def interpolate_param(self, start, end, iteration, max_iterations):
        if end is None:
            return start
        # Ease-in function: fast early movement, then slow convergence
        progress = iteration / max_iterations
        eased_progress = 1 - np.exp(-5 * progress)  # The constant (5) controls steepness
        return start + (end - start) * eased_progress

    def _build_cluster_tree(self, n):
        """
        Build scipy ClusterNode tree directly from cluster_tree by iterating backwards.

        The cluster_tree stores merge events as tuples:
            (cluster_a, cluster_b, new_cluster_id, distance, size)

        Leaf nodes are indices 0 to n-1 (preserved original indices).
        Merged clusters have unique indices n, n+1, n+2, ...

        Args:
            n: number of original data points (leaf nodes)
        """
        from scipy.cluster.hierarchy import ClusterNode

        if not self.cluster_tree:
            self.tree_ = None
            self.node_list_ = None
            return

        # Build lookup from new_id to merge info for backwards traversal
        merge_lookup = {}
        for cluster_a, cluster_b, new_id, dist, size in self.cluster_tree:
            merge_lookup[new_id] = (cluster_a, cluster_b, dist, size)

        # Storage for all nodes (for node_list_ output)
        nodes = {}

        def build_node(cluster_id):
            """Recursively build ClusterNode tree by iterating backwards from root."""
            if cluster_id in nodes:
                return nodes[cluster_id]

            if cluster_id < n:
                # Leaf node - original data point (indices 0 to n-1 preserved)
                node = ClusterNode(id=cluster_id, left=None, right=None, dist=0, count=1)
            else:
                # Internal node - merged cluster with unique index >= n
                cluster_a, cluster_b, dist, size = merge_lookup[cluster_id]
                # Recursively build children (iterating backwards through tree structure)
                left = build_node(cluster_a)
                right = build_node(cluster_b)
                node = ClusterNode(id=cluster_id, left=left, right=right, dist=dist, count=size)

            nodes[cluster_id] = node
            return node

        # Start from the root (last merge in cluster_tree) and iterate backwards
        root_id = self.cluster_tree[-1][2]
        self.tree_ = build_node(root_id)

        # Build node_list sorted by id (compatible with scipy's to_tree rd=True output)
        self.node_list_ = [nodes[i] for i in sorted(nodes.keys())]

    def get_labels(self, n_clusters=None):
        """
        Retrieves cluster labels for all original data points at a given granularity.

        Args:
            n_clusters: number of clusters desired (if None, returns finest level)

        Returns:
            Sets self.labels_ to array of cluster assignments for each original point
        """
        if self.tree_ is None:
            # No tree built, each point is its own cluster
            n = self.n_samples_ if self.n_samples_ else 0
            self.labels_ = np.arange(n)
            return

        # Use stored number of original data points
        n = self.n_samples_

        if n_clusters is None:
            # Return finest level - each point is its own cluster
            self.labels_ = np.arange(n)
            return

        # Cut tree at desired number of clusters
        labels = np.zeros(n, dtype=int)

        def assign_leaf_labels(node, cluster_id):
            """Assign all leaves under this node to the same cluster."""
            if node.is_leaf():
                labels[node.id] = cluster_id
            else:
                assign_leaf_labels(node.left, cluster_id)
                assign_leaf_labels(node.right, cluster_id)

        # Start with root as single cluster, expand until we have n_clusters
        clusters = [self.tree_]
        while len(clusters) < n_clusters:
            # Find the cluster with largest count to split
            clusters.sort(key=lambda x: -x.count)
            to_split = None
            for c in clusters:
                if not c.is_leaf():
                    to_split = c
                    break
            if to_split is None:
                break  # All clusters are leaves, can't split further
            clusters.remove(to_split)
            clusters.extend([to_split.left, to_split.right])

        # Assign labels based on final clusters
        for cluster_id, cluster_node in enumerate(clusters):
            assign_leaf_labels(cluster_node, cluster_id)

        self.labels_ = labels

    def predict(self):
        return None