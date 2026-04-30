# Unsupervised NLP Analysis

**Note:** This project requires a Linux environment with a CUDA-enabled GPU (tested on an HPC cluster with CUDA support).

---

## Project Summary

The goal of this project is to benchmark dimensionality reduction methods combined with clustering algorithms on textual embeddings to support unsupervised NLP analysis for US fisheries discourse and narratives. We evaluate how well methods like PHATE, PCA, UMAP, t-SNE, and PaCMAP preserve hierarchical structure in text data when paired with clustering algorithms including Agglomerative, HDBSCAN, Diffusion Condensation, and Hercules. Results are measured using FM index, ARI, AMI, and Rand Index across five benchmark datasets.

---

## Quickstart (HPC Cluster)

Run these at the start of every new terminal session:

```bash
module purge
module load Miniforge3/25.11.0-1
conda activate phate-env
```

Then download data (once) and run the smallest dataset end-to-end:

```bash
PYTHONPATH="" python data/download_data.py --datasets rcv1
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
```

All scripts must be run from the **repo root** - not from inside `src/`. See [INSTALL.md](INSTALL.md) for full setup.

---

## Installation

For full installation instructions and environment setup, see:

[INSTALL.md](INSTALL.md)

---

## Key Components

### Text Embeddings
Documents are embedded using two models: `Qwen3-Embedding-0.6B` and `all-MiniLM-L6-v2` via the `sentence-transformers` library.

### Dimensionality Reduction
We compare six reduction methods: PHATE, PCA, UMAP, t-SNE, PaCMAP, and TriMAP. GPU-accelerated implementations (cuML) are used where available.

### Clustering
Four clustering methods are applied at multiple hierarchy levels: Agglomerative Clustering, HDBSCAN, Diffusion Condensation, and Hercules.

### Evaluation
Clustering quality is measured against ground truth labels using FM index, Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), Rand Index, Dendrogram Purity, and LCA-F1.

---

## Reproducibility

For step-by-step instructions to reproduce all figures and results, see:

[REPRODUCIBILITY.md](REPRODUCIBILITY.md)

---

## Running Benchmark Experiments

Once datasets are in place (see INSTALL.md), run:

```bash
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset amazon
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset dbpedia
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset wos
```

---

## Expected Output

Running the benchmark pipeline will generate:
- Clustering metric CSVs saved in `results/clustering/benchmark/`
- Scatter grid figures saved in `results/summary_figures/`
- Visualization quality metric CSVs saved in `results/viz_metrics/`
- Cached embeddings and reductions in `src/cache/` (not tracked, gitignored)

---

## Project Structure

```
Unsupervised_NLP_Analysis/
│
├── data/                                   # All datasets and computed outputs
│   ├── arxiv/                              # arXiv abstracts
│   ├── amazon/                             # Amazon reviews
│   ├── dbpedia/                            # DBpedia articles
│   ├── rcv1/                               # RCV1 news articles
│   ├── WebOfScience/                       # Web of Science papers
│   ├── synthetic/                          # LLM-generated synthetic datasets
│   │   ├── generate.py
│   │   └── generated_data/
│
├── notebooks/
│   └── evaluations/                        # Analysis and evaluation notebooks
│       ├── metric_tables.ipynb
│       ├── clustering_summary_tables.ipynb
│       ├── visualization_metrics_benchmark.ipynb
│       └── visualization_metrics_synthetic.ipynb
├── results/                                # All pipeline outputs (see results/README.md)
│   ├── clustering/
│   │   ├── benchmark/                      # Benchmark clustering + herc scores
│   │   └── synthetic/                      # Synthetic clustering + herc scores
│   ├── summary_figures/                    # Scatter grid PNGs (gitignored)
│   └── viz_metrics/                        # Visualization quality metric CSVs
│
├── src/
│   ├── cache/                              # Pipeline cache: embeddings, reductions (gitignored)
│   ├── custom_packages/                    # Custom algorithm implementations
│   │   ├── diffusion_condensation.py
│   │   ├── fowlkes_mallows.py
│   │   ├── hercules.py
│   │   ├── dendrogram_purity.py
│   │   ├── graph_utils.py
│   │   └── lca_f1.py
│   │
│   └── run_models/
│       ├── benchmark_datasets/             # Benchmark evaluation pipeline
│       │   ├── eval_pipeline.py
│       │   ├── herc_pipeline.py
│       │   ├── viz_metrics_script.py
│       │   └── scatter_grid_benchmark.py
│       └── synthetic_data/                 # Synthetic data evaluation pipeline
│           ├── scatter_grid_synthetic.py
│           ├── synth_herc_pipeline.py
│           ├── viz_metrics_script.py
│           └── run_all.sh
│
├── environment.yml                         # Conda environment (Linux/CUDA)
├── INSTALL.md                              # Installation and data setup instructions
├── REPRODUCIBILITY.md                      # Steps to reproduce all figures and results
├── LICENSE
└── README.md
```

---

## Authors

Anonymous
