# RCV1 Dataset

Reuters Corpus Volume 1: a benchmark news categorization dataset with hierarchical topic labels.

## Download

Run from the repo root to download automatically (recommended):

```bash
python data/download_data.py --datasets rcv1
```

Or run the import script directly:

```bash
python data/rcv1/import_rcv1.py
```

This uses `sklearn.datasets.fetch_rcv1`, preprocesses the data, and saves it to `data/rcv1/rcv1.csv`.

## Expected output
```
data/rcv1/rcv1.csv
```

> **Note:** Recommended starting point for reproducing results, as it is the smallest of the five benchmark datasets.
