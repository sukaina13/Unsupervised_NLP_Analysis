# Amazon Dataset

Hierarchical product review dataset with multi-level category labels.

## Download

Run from the repo root to download automatically (recommended):

```bash
python data/download_data.py --datasets amazon
```

Or manually:

1. Download `train_40k.csv` and `val_10k.csv` from Kaggle:
   https://www.kaggle.com/datasets/kashnitsky/hierarchical-text-classification

2. Place both files at:
   ```
   data/amazon/train_40k.csv
   data/amazon/val_10k.csv
   ```
