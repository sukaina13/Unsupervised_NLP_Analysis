# Web of Science Dataset

Research article dataset with hierarchical subject category labels across 7 domains and 134 subcategories.

## Download

Run from the repo root to download automatically (recommended):

```bash
python data/download_data.py --datasets wos
```

Or manually:

1. Download `Data.xlsx` from Mendeley:
   https://data.mendeley.com/datasets/9rw3vkcfy4/6

2. Place it at:
   ```
   data/WebOfScience/Data.xlsx
   ```

> **Note:** The file must be named `Data.xlsx` directly in `data/WebOfScience/`, not inside a `Meta-data/` subfolder.
