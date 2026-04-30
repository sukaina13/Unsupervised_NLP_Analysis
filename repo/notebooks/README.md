# Notebooks

## milestones/

Class deliverables for CMSE 495. Useful for understanding the project's progression and overall pipeline.

| Notebook | Description |
|----------|-------------|
| `demo.ipynb` | Full end-to-end pipeline demo: loading data, generating embeddings, dimensionality reduction, clustering, and visualization |
| `MVP_demo.ipynb` | Minimum viable product demo from early in the project |
| `NCEAS_Reproducibility.ipynb` | Shepard diagram reproducibility notebook submitted as a milestone |

## evaluations/

Notebooks for analyzing and visualizing benchmark results.

| Notebook | Description |
|----------|-------------|
| `metric_tables.ipynb` | Primary evaluation notebook, generates summary tables of FM, ARI, AMI, Rand Index, Dendrogram Purity, and LCA-F1 scores across all DR + clustering combinations |
| `clustering_summary_tables.ipynb` | Summary statistics (median, mean +/- std) across all clustering score CSVs |
| `visualization_metrics_benchmark.ipynb` | Computes Trustworthiness, Continuity, Spearman, and DEMaP for benchmark datasets; produces Shepard diagrams |
| `visualization_metrics_synthetic.ipynb` | Same visualization metrics for synthetic dataset configurations |
| `clustering_results.ipynb` | Aggregates viz metrics CSVs across all synthetic configs and benchmark datasets to produce mean ± std summary tables (Tables 4 and 5 in the paper) |
| `dataset_statistics.ipynb` | Summary statistics for each benchmark dataset (size, label distribution, hierarchy depth) |
| `results.ipynb` | Combined results analysis notebook for the final paper |
