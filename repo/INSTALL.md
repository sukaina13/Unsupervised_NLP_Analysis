# Installation Instructions
NCEAS Unsupervised NLP 

These instructions reproduce the project environment and run the benchmark pipeline on a Linux system with CUDA support.


## A few Important Notes
1. This project requires a Linux system with a CUDA-compatible GPU. All experiments are designed to run on an HPC cluster (e.g., MSU HPCC). The pipeline uses GPU-accelerated libraries (cuML, cuPCA, cuUMAP) that are not available on macOS or Windows. For reference, the code in this library was developed and executed with a configuration of 8 CPU cores, 64 GB of RAM, and 2 V100 GPUs.

2. Running the full pipeline for all real and synthetic data sources will take hours with most hardware configurations. If you simply wish to run our pipeline end to end, to ensure our code is reproducible, we recommend only running our pipelines for the **RCV1** dataset, which is the smallest of our data sources.

3. A Developer Groq API key with a payment method configured is required to use GPT-OSS-120B, the model used for synthetic data generation. It shouldn't cost you more than a few cents, however. Specific instructions for creating an account on Groq, enabling payment authorization, and creating an API key can be found below.

---

## 1. Clone the Repository

```bash
git clone https://github.com/harshil0217/NCEAS_Unsupervised_NLP.git
cd NCEAS_Unsupervised_NLP
```

---

## 2. Create the Project Environment with Conda

**On MSU HPCC**, load the module first:

```bash
module purge
module load Miniforge3/25.11.0-1
```

Then create the environment from the repo root:

```bash
conda env create -f environment.yml
conda activate phate-env
```

Skip `conda env create` if `phate-env` already exists from a previous session - just run `conda activate phate-env`.

> **Note:** On dev nodes, prefix all python commands with `PYTHONPATH=""` to avoid system Python conflicts (e.g. `PYTHONPATH="" python src/...`).

---

## 3. Subsequent HPCC Sessions

Run these at the start of every new terminal session before anything else:

```bash
module purge
module load Miniforge3/25.11.0-1
conda activate phate-env
```

---

## 4. Create a Groq API key

If you do not have existing Groq credentials or a Groq developer account, follow the steps listed below as needed.

1. Visit https://console.groq.com/home and create an account
2. Navigate to the billing tab on the settings page to upgrade to a developer account
3. Create an API key in the Groq console https://console.groq.com/keys

---

## 5. Set Up API Keys

Create a `.env` file in the project root:

```bash
nano .env
```

Add the following keys and save:

```bash
# Kaggle (required for arXiv, Amazon, DBpedia downloads)
# Get your token at https://www.kaggle.com/settings > API > Create New Token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Groq (required for synthetic data generation)
# Get your key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key
```

Then load the keys into your session:

```bash
export $(grep -v '^#' .env | xargs)
```

---

## 6. Data Setup

This repository does **not include benchmark datasets**. Use the provided download script to fetch all datasets automatically.

### Download All Datasets

```bash
PYTHONPATH="" python data/download_data.py
```

This downloads and preprocesses all five benchmark datasets automatically:

| Dataset | Source | Notes |
|---------|--------|-------|
| arXiv | Kaggle | ~4 GB download, sampled to 30k papers |
| Amazon | Kaggle | `train_40k.csv` + `val_10k.csv` |
| DBpedia | Kaggle | `DBPEDIA_test.csv` |
| RCV1 | sklearn | No Kaggle needed |
| Web of Science | Mendeley | Downloaded automatically |

To download a specific dataset only:
```bash
PYTHONPATH="" python data/download_data.py --datasets rcv1
```

### Required Output Structure

```bash
data/
├── arxiv/
│   └── arxiv_clean.csv
├── amazon/
│   ├── train_40k.csv
│   └── val_10k.csv
├── dbpedia/
│   └── DBPEDIA_test.csv
├── rcv1/
│   └── rcv1.csv
└── WebOfScience/
    └── Data.xlsx
```

---

**Path convention note:** All scripts must be run from the **repo root** (not from inside `src/`). The scripts internally `cd` into `src/` at startup and use `../` to reference data and results. Running from the wrong directory will cause file-not-found errors.

Results are saved to `results/clustering/benchmark/`.

---

## 7. Quickstart (end-to-end on RCV1)

RCV1 is the smallest dataset and recommended for a quick end-to-end test. It completes within the 2-hour dev node limit (tested on `dev-amd24`, H200 GPU).

> **Note:** To reproduce the scatter grid figures, all 5 datasets are required. Use `PYTHONPATH="" python data/download_data.py` (without `--datasets rcv1`) to download everything.

```bash
# Load environment (run at the start of every session on HPCC)
module purge
module load Miniforge3/25.11.0-1
conda activate phate-env

# Load API keys
export $(grep -v '^#' .env | xargs)

# Download RCV1 only (for quick test)
PYTHONPATH="" python data/download_data.py --datasets rcv1

# Run clustering pipeline
PYTHONPATH="" python src/run_models/benchmark_datasets/eval_pipeline.py --dataset rcv1

# Run visualization metrics
PYTHONPATH="" python src/run_models/benchmark_datasets/viz_metrics_script.py --dataset rcv1

# Generate scatter grid figures
PYTHONPATH="" python src/run_models/benchmark_datasets/scatter_grid_benchmark.py
```

Results will be saved to `results/clustering/benchmark/rcv1_clustering_scores.csv`.

---

## 8. Synthetic Data

To generate synthetic data run:

```bash
PYTHONPATH="" python data/synthetic/generate.py
```

**Note**: Ensure that your Groq API key is configured properly in your `.env` file.

The synthetic datasets will be saved to `data/synthetic/generated_data/`
