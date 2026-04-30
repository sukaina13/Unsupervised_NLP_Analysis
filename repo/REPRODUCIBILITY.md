# NCEAS NLP Reproducibility Guide


For installation and environment setup, see [INSTALL.md](https://github.com/harshil0217/NCEAS_Unsupervised_NLP/blob/main/INSTALL.md).

---

## Table of Contents

1. [Setup](#1-setup)
2. [Figure: Synthetic Scatter Grid](#2-figure-synthetic-scatter-grid)
3. [Figure: Benchmark Scatter Grid](#3-figure-benchmark-scatter-grid)
4. [Table: Average Clustering Performance by Dimensionality Reduction Method - Benchmark](#4-table-average-clustering-performance-by-dimensionality-reduction-method---benchmark)
5. [Table: Average Clustering Performance by Dimensionality Reduction Method - Synthetic](#5-table-average-clustering-performance-by-dimensionality-reduction-method---synthetic)
6. [Table: Top 8 Configurations - Benchmark](#6-table-top-8-configurations---benchmark)
7. [Table: Top 8 Configurations - Synthetic](#7-table-top-8-configurations---synthetic)
8. [Figure: Shepard Diagrams - Synthetic Data](#8-figure-shepard-diagrams---synthetic-data)
9. [Figure: Shepard Diagrams - Benchmark Datasets](#9-figure-shepard-diagrams---benchmark-datasets)
10. [Tables: Aggregated DR and Clustering Performance](#10-tables-aggregated-dr-and-clustering-performance)

---

## 1. Setup

Follow [INSTALL.md](https://github.com/harshil0217/NCEAS_Unsupervised_NLP/blob/main/INSTALL.md) to clone the repo, set up the conda environment, and download all datasets.

On MSU HPCC, run at the start of every terminal session:

```bash
module purge
module load Miniforge3/25.11.0-1
conda activate phate-env
export $(grep -v '^#' .env | xargs)
```

All commands below must be run from the **repo root**.

> **HPCC dev nodes:** If you see import errors, prefix every `python` command with `PYTHONPATH=""` (e.g. `PYTHONPATH="" python src/...`).

---

## 2. Figure: Synthetic Scatter Grid

**Used in:** Final report (Figure 2), final presentation.

**Description:** A grid of scatter plots showing 2D dimensionality reductions of synthetic text embeddings. Rows = 4 synthetic dataset configurations (2 themes x 2 hierarchy shapes). Columns = 6 DR methods (PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP). Points are colored by category label. Two versions per embedding model: top-level category and subcategory.

**Output files** (saved to `results/summary_figures/`, gitignored):

* `fig2_scatter_grid_minilm.png`
* `fig2_scatter_grid_qwen.png`
* `fig2_scatter_grid_minilm_cat1.png`
* `fig2_scatter_grid_qwen_cat1.png`
* `fig2_scatter_grid_minilm_cat1_legend.png`
* `fig2_scatter_grid_qwen_cat1_legend.png`

---

### Step 1: Generate Synthetic Data

Skip if already done - the script checks for existing files before regenerating.

```
python data/synthetic/generate.py
```

This runs all 12 configs automatically (2 themes × 3 noise levels × 2 hierarchy shapes) and saves CSVs to `data/synthetic/generated_data/`. The configs cover both `Energy_Ecosystems_and_Humans` and `Offshore_energy_impacts_on_fisheries` themes with `add_noise` values of `0.0`, `0.25`, and `0.5`.

> **Note:** This step calls the Groq API and may take a while for all 12 configs. Existing output files are skipped on rerun.

---

### Step 2: Generate Embeddings and 2D Reductions

Run the viz metrics script for each synthetic config (requires GPU on HPCC):

```
python src/run_models/synthetic_data/viz_metrics_script.py
```

This embeds each config with both `all-MiniLM-L6-v2` and `Qwen3-Embedding-0.6B` and reduces to 2D using all six DR methods. Results are cached in `src/cache/{model}_reduced_2d/`.

---

### Step 3: Generate the Figure

```
python src/run_models/synthetic_data/scatter_grid_synthetic.py
```

This reads 2D reductions from `src/cache/` and labels from `data/synthetic/generated_data/`, then saves all PNGs to `results/summary_figures/`. Uses a fixed random seed (42) for reproducibility.

---

## 3. Figure: Benchmark Scatter Grid

**Used in:** Final report, final presentation.

**Description:** A 5x6 scatter grid for real-world benchmark datasets. Rows = RCV1, arXiv, Amazon, WoS, DBpedia. Columns = 6 DR methods. Points colored by top-level category.

> **Note:** This figure requires all 5 benchmark datasets. Run `python data/download_data.py` (not `--datasets rcv1`) to download all datasets before proceeding.

**Output files** (saved to `results/summary_figures/`, gitignored):

* `fig_scatter_grid_benchmark_minilm.png`
* `fig_scatter_grid_benchmark_qwen.png`
* `fig_scatter_grid_benchmark_minilm_legend.png`
* `fig_scatter_grid_benchmark_qwen_legend.png`

---

### Step 1: Download Datasets

```
python data/download_data.py
```

See [INSTALL.md](https://github.com/harshil0217/NCEAS_Unsupervised_NLP/blob/main/INSTALL.md) for Kaggle credentials required for arXiv, Amazon, and DBpedia.

---

### Step 2: Generate Embeddings and 2D Reductions

Run the viz metrics script for each dataset (requires GPU on HPCC):

```
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset rcv1
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset arxiv
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset amazon
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset wos
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset dbpedia
```

2D reductions are cached in `src/cache/{model}_reduced_2d/` with filenames like `PHATE_2d_{dataset}_full{N}.npy`. The script is safe to rerun and resumes from the last completed method.

---

### Step 3: Generate the Figure

```
python src/run_models/benchmark_datasets/scatter_grid_benchmark.py
```

This reads 2D reductions from `src/cache/` and saves the scatter grid PNGs and separate legend PNGs to `results/summary_figures/`. All points are plotted using a 1-99 percentile axis clip with fixed random seed (42).

**Expected dataset sizes:**

| Dataset | N |
| --- | --- |
| RCV1 | 1,566 |
| arXiv | ~29,500-30,000 |
| Amazon | 14,824 |
| WoS | 46,985 |
| DBpedia | 60,794 |

---

## 4. Table: Average Clustering Performance by Dimensionality Reduction Method - Benchmark

**Used in:** Final report (Table 5).

**Description:** Summary table showing average performance of all dimensionality reduction methods across all benchmark datasets and metrics (AMI, ARI, Dendrogram Purity, FM, LCA-F1, Rand).

**Output files** (saved to `results/clustering/benchmark/`)

* The results are stored in multiple csv files that contain all the clustering scores for each inputted dataset

---

### Step 1: Download datasets

Follow Step 1 from [Section 3](#3-figure-benchmark-scatter-grid) above, or skip if already done.

---

### Step 2: Run Pipeline Evals

Generate the raw clustering metrics by running the evaluation pipeline for each dataset:

```
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset amazon
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset dbpedia
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset wos
```

---

### Step 3: Run Metric Tables Notebook

Open and run all cells in the aggregation notebook to produce the final summary tables:

```
notebooks/evaluations/metric_tables.ipynb
```

This notebook will import the precomputed CSVs from `results/clustering/benchmark/`, group results by embedding model and reduction method, and sort by Dendrogram Purity to identify the top 8 combinations for the final report.

---

## 5. Table: Average Clustering Performance by Dimensionality Reduction Method - Synthetic

**Used in:** Final report (Table 6).

**Description:** Summary table showing average performance of all dimensionality reduction methods across all synthetic datasets and metrics (AMI, ARI, Dendrogram Purity, FM, LCA-F1, Rand, TED).

**Output files** (saved to `results/clustering/synthetic/`)

* The results are stored in multiple csv files that contain all the clustering scores for each inputted dataset

---

### Step 1: Generate Synthetic

Run data generation and embeddings if not already done:

```
python data/synthetic/generate.py
```

See [Section 2](#2-figure-synthetic-scatter-grid) for full details on what these produce.

---

### Step 2: Run Clustering Evals

Generate the raw clustering metrics by running:

```
python src/run_models/synthetic_data/synthetic_eval_pipeline.py
```

---

### Step 3: Run Metric Tables Notebook

Open and run all cells in the aggregation notebook to produce the final summary tables:

```
notebooks/evaluations/metric_tables.ipynb
```

---

## 6. Table: Top 8 Configurations - Benchmark

**Used in:** Final report (Table 7).

**Description:** Top 8 best performing embedding model + dimensionality reduction + hierarchical clustering algorithm configurations, ordered by Dendrogram Purity, evaluated on benchmark data.

**Output files** (saved to `results/clustering/benchmark/`)

* The results are stored in multiple csv files that contain all the clustering scores for each inputted dataset

---

### Step 1: Download datasets

Follow Step 1 from [Section 3](#3-figure-benchmark-scatter-grid) above, or skip if already done.

---

### Step 2: Run Pipeline Evals

```
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset amazon
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset arxiv
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset dbpedia
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset wos
```

---

### Step 3: Run Metric Tables Notebook

```
notebooks/evaluations/metric_tables.ipynb
```

---

## 7. Table: Top 8 Configurations - Synthetic

**Used in:** Final report (Table 8).

**Description:** Top 8 best performing embedding model + dimensionality reduction + hierarchical clustering algorithm configurations, ordered by Dendrogram Purity, evaluated on synthetic data.

**Output files** (saved to `results/clustering/synthetic/`)

* The results are stored in multiple csv files that contain all the clustering scores for each inputted dataset

---

### Step 1: Generate Synthetic

Run data generation and embeddings if not already done:

```
python data/synthetic/generate.py
```

See [Section 2](#2-figure-synthetic-scatter-grid) for full details on what these produce.

---

### Step 2: Run Clustering Evals

```
python src/run_models/synthetic_data/synthetic_eval_pipeline.py
```

---

### Step 3: Run Metric Tables Notebook

```
notebooks/evaluations/metric_tables.ipynb
```

---

## 8. Figure: Shepard Diagrams - Synthetic Data

**Used in:** Final report, final presentation.

**Description:** Shepard diagrams compare pairwise distances in the original high-dimensional embedding space vs. distances in 2D. Points near the diagonal indicate better global distance preservation. One diagram per DR method per synthetic config.

**Output files** (saved to `results/shepard_diagrams/{provider}/`, gitignored):

* `shepard_{config}_{method}.png`

**Intermediate results (tracked in git):** Visualization quality metric CSVs in `results/viz_metrics/{provider}/` - these can be loaded directly without rerunning the pipeline.

---

### Step 1: Generate Synthetic Data and Embeddings

Run data generation and embeddings if not already done:

```
python data/synthetic/generate.py
python src/run_models/synthetic_data/viz_metrics_script.py
```

See [Section 2](#2-figure-synthetic-scatter-grid) for full details on what these produce.

---

### Step 2: Run the Visualization Metrics Notebook

Open and run all cells in:

```
notebooks/evaluations/visualization_metrics_synthetic.ipynb
```

Set the config variables at the top:

```
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # or Qwen/Qwen3-Embedding-0.6B
theme           = "Energy_Ecosystems_and_Humans"
max_sub         = 3
depth           = 5
add_noise       = 0
```

The notebook computes Trustworthiness, Continuity, Spearman Correlation, and DEMaP for each DR method, then saves a Shepard diagram for each combination to `results/shepard_diagrams/`. Results are cached so the notebook can be safely re-run.

---

## 9. Figure: Shepard Diagrams - Benchmark Datasets

**Used in:** Final report, final presentation.

**Description:** Same as above but for the five real-world benchmark datasets. For large datasets (>10,000 points), metrics are computed over 30 random subsamples of 10,000 points and reported as mean +/- std.

**Output files** (saved to `results/shepard_diagrams/{provider}/`, gitignored):

* `shepard_{dataset}_{method}.png`

**Intermediate results (tracked in git):** Metric CSVs already in `results/viz_metrics/{provider}/viz_metrics_{dataset}.csv` - skip Step 1 if you only want to view results.

| Dataset | Points | Subsampling |
| --- | --- |  --- |
| RCV1 | 1,566 | No - full dataset used |
| arXiv | ~29,500-30,000 | Yes - 30x subsamples of 10,000 |
| Amazon | 14,824 | Yes - 30x subsamples of 10,000 |
| DBpedia | 60,794 | Yes - 30x subsamples of 10,000 |
| WoS | 46,985 | Yes - 30x subsamples of 10,000 |

---

### Step 1: Run Metrics on HPCC (optional - CSVs already tracked)

Benchmark datasets require a GPU. Run from repo root:

```
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset rcv1 --model sentence-transformers/all-MiniLM-L6-v2
python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset rcv1 --model Qwen/Qwen3-Embedding-0.6B
```

Replace `rcv1` with `arxiv`, `amazon`, `dbpedia`, or `wos` for other datasets. The script handles 2D reductions, subsampling, and incremental CSV output - it resumes from the last completed method if interrupted.

---

### Step 2: Run the Benchmark Visualization Metrics Notebook

Open and run all cells in:

```
notebooks/evaluations/visualization_metrics_benchmark.ipynb
```

Set the variables at the top:

```
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # or Qwen/Qwen3-Embedding-0.6B
dataset         = "rcv1"  # options: rcv1, arxiv, amazon, dbpedia, wos
```

The notebook loads precomputed CSVs from `results/viz_metrics/`, plots the metrics, and generates Shepard diagrams saved to `results/shepard_diagrams/`.

---

## 10. Tables: Aggregated DR and Clustering Performance

**Used in:** Final report (Tables 3, 4, 10, 11, 12, 13, 14).

**Description:** `clustering_results.ipynb` produces all aggregated performance tables:
- **Table 3**: Mean DR performance (Trust, Cont, Spearman, DEMaP) across all benchmark datasets
- **Table 4**: Same across all synthetic configs
- **Tables 10-13**: Per-dataset clustering performance (arXiv, RCV1, WoS, DBpedia)
- **Table 14**: Benchmark DR evaluation per dataset and embedding model

**Prerequisites:** Viz metrics CSVs in `results/viz_metrics/` (tracked in git) and clustering CSVs in `results/clustering/` (tracked in git). Skip Step 1 to use precomputed results.

---

### Step 1: Generate CSVs (if not already done)

For viz metrics: run Section 8 Step 1 (synthetic) and Section 9 Step 1 (benchmark).

For clustering scores: run Section 4 Step 2 (benchmark) and Section 5 Step 2 (synthetic).

---

### Step 2: Run the Aggregation Notebook

Open and run all cells in:

```
notebooks/evaluations/clustering_results.ipynb
```

This notebook loads all viz metrics and clustering CSVs, aggregates across configs/datasets, and produces all summary tables.
