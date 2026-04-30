# Synthetic Data

LLM-generated synthetic ecological text datasets with known hierarchical structure. Used to evaluate dimensionality reduction and clustering methods under controlled conditions where ground truth labels are available.

## Overview

Datasets are generated across two ecological themes, two hierarchy shapes, and three noise levels:

| Config | Theme | max_sub | depth |
|--------|-------|---------|-------|
| Ecosystems (shallow) | Energy Ecosystems and Humans | 5 | 3 |
| Ecosystems (deep) | Energy Ecosystems and Humans | 3 | 5 |
| Fisheries (shallow) | Offshore Energy Impacts on Fisheries | 5 | 3 |
| Fisheries (deep) | Offshore Energy Impacts on Fisheries | 3 | 5 |

Each config is run at noise levels 0%, 25%, and 50% (off-topic sentences added), yielding 24 datasets total.

## Requirements

A Groq API key is required:

```bash
export GROQ_API_KEY=your_key_here
```

## Generate all datasets

```bash
cd src/run_models/synthetic_data
bash run_all.sh
```

## Generate a single dataset

```bash
python data/synthetic/generate.py \
  --theme Energy_Ecosystems_and_Humans \
  --t 1.0 --max_sub 3 --depth 5 \
  --synonyms 0 --branching random --add_noise 0
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `--theme` | `Energy_Ecosystems_and_Humans` or `Offshore_energy_impacts_on_fisheries` |
| `--t` | LLM temperature (default: 1.0) |
| `--max_sub` | Max subcategories per node (hierarchy width) |
| `--depth` | Hierarchy depth |
| `--synonyms` | Synonym variations per document |
| `--branching` | `random` or `constant` |
| `--add_noise` | Fraction of off-topic noise documents (0, 0.25, or 0.5) |

## Output

Generated CSV files are saved to `data/synthetic/generated_data/`. Each file contains document text and hierarchical topic labels at every level.
