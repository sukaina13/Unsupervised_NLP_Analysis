# Data

This folder contains the five benchmark datasets and LLM-generated synthetic data. Datasets are **not included in the repository** and must be downloaded separately.

## Automated Download

Run from the repo root:

```bash
python data/download_data.py
```

Requires Kaggle credentials for arXiv, Amazon, and DBpedia. See [INSTALL.md](../INSTALL.md) for setup instructions.

## Datasets

| Dataset | Folder | Size | Source |
|---------|--------|------|--------|
| arXiv | `arxiv/` | ~30k abstracts | Kaggle / Cornell |
| Amazon | `amazon/` | 50k reviews | Kaggle |
| DBpedia | `dbpedia/` | ~342k articles | Kaggle |
| RCV1 | `rcv1/` | 1,566 news articles | scikit-learn |
| Web of Science | `WebOfScience/` | ~46k papers | Mendeley |
| Synthetic | `synthetic/` | 12 configs | LLM-generated (Groq API) |

Synthetic data is generated separately, see [INSTALL.md](../INSTALL.md) Section 5.
