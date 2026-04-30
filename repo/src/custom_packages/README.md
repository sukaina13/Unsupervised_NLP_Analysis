# custom_packages

Custom implementations of clustering algorithms and hierarchical evaluation metrics.

## Clustering Algorithms

| Module | Description |
|--------|-------------|
| `diffusion_condensation.py` | GPU-accelerated Diffusion Condensation: iteratively applies a diffusion operator until clusters condense into stable regions |

## Evaluation Metrics

| Module | Description |
|--------|-------------|
| `fowlkes_mallows.py` | Fowlkes-Mallows index with expected value and variance for flat clustering comparison |
| `dendrogram_purity.py` | Monte Carlo Dendrogram Purity: measures how well a predicted hierarchy groups same-class points |
| `lca_f1.py` | Monte Carlo LCA-F1: compares lowest common ancestor subtrees between predicted and ground truth hierarchies |
| `graph_utils.py` | Shared tree utilities: scipy ClusterNode to anytree conversion, LCA lookup, APTED tree edit distance |
