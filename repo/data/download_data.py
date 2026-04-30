"""
download_data.py

Automates downloading and preprocessing all benchmark datasets.

Usage (from repo root):
    python data/download_data.py                      # all automated datasets
    python data/download_data.py --datasets arxiv amazon dbpedia rcv1 wos

Requirements:
    - Kaggle API credentials at ~/.kaggle/kaggle.json
      (get yours at https://www.kaggle.com/settings > API > Create New Token)

Datasets:
    arxiv   -- downloaded via Kaggle CLI, then cleaned with data/arxiv/clean_arxiv.py
    amazon  -- downloaded via Kaggle CLI
    dbpedia -- downloaded via Kaggle CLI
    rcv1    -- fetched automatically via sklearn (no Kaggle needed)
    wos     -- downloaded automatically from Mendeley Data

Not handled here:
    synthetic -- requires a Groq API key, run separately:
                 cd src/run_models/synthetic_data && bash run_all.sh
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from dotenv import load_dotenv
load_dotenv()

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ========================
# Repo root resolution
# ========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def path(*parts):
    return os.path.join(REPO_ROOT, *parts)

# ========================
# Helpers
# ========================

def run(cmd, cwd=None):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f"  ERROR: command failed with code {result.returncode}")
        sys.exit(1)

def check_kaggle():
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in your .env file.")
        print("Get your API token at https://www.kaggle.com/settings > API > Create New Token")
        sys.exit(1)
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    if shutil.which("kaggle") is None:
        print("Kaggle CLI not installed. Run: pip install kaggle")
        sys.exit(1)

def extract_zip(zip_path, dest_dir):
    print(f"  Extracting {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    os.remove(zip_path)

def already_exists(label, filepath):
    if os.path.exists(filepath):
        print(f"  [{label}] Already exists, skipping: {filepath}")
        return True
    return False

# ========================
# Dataset downloaders
# ========================

def download_arxiv():
    print("\n[arXiv]")
    out_csv = path("data", "arxiv", "arxiv_clean.csv")
    raw_json = path("data", "arxiv", "arxiv-metadata-oai-snapshot.json")

    if already_exists("arXiv", out_csv):
        return

    if not os.path.exists(raw_json):
        print("  WARNING: arXiv raw snapshot is ~4 GB. This may take a while.")
        check_kaggle()
        zip_path = path("data", "arxiv", "arxiv.zip")
        run(["kaggle", "datasets", "download", "-d", "Cornell-University/arxiv",
             "-p", path("data", "arxiv")])
        # kaggle saves as <dataset-name>.zip
        downloaded = path("data", "arxiv", "arxiv.zip")
        if not os.path.exists(downloaded):
            # kaggle may name it differently
            zips = [f for f in os.listdir(path("data", "arxiv")) if f.endswith(".zip")]
            if zips:
                downloaded = path("data", "arxiv", zips[0])
            else:
                print("  Could not find downloaded zip. Check data/arxiv/")
                sys.exit(1)
        extract_zip(downloaded, path("data", "arxiv"))

    print("  Running clean_arxiv.py ...")
    run([sys.executable, path("data", "arxiv", "clean_arxiv.py")])
    print(f"  Done: {out_csv}")


def download_amazon():
    print("\n[Amazon]")
    train = path("data", "amazon", "train_40k.csv")
    val   = path("data", "amazon", "val_10k.csv")

    if already_exists("Amazon", train) and already_exists("Amazon", val):
        return

    check_kaggle()
    run(["kaggle", "datasets", "download", "-d", "kashnitsky/hierarchical-text-classification",
         "-p", path("data", "amazon")])

    zips = [f for f in os.listdir(path("data", "amazon")) if f.endswith(".zip")]
    if zips:
        extract_zip(path("data", "amazon", zips[0]), path("data", "amazon"))

    print(f"  Done: {train}, {val}")


def download_dbpedia():
    print("\n[DBpedia]")
    out = path("data", "dbpedia", "DBPEDIA_test.csv")

    if already_exists("DBpedia", out):
        return

    check_kaggle()
    run(["kaggle", "datasets", "download", "-d", "danofer/dbpedia-classes",
         "-p", path("data", "dbpedia")])

    dbpedia_dir = path("data", "dbpedia")
    zips = [f for f in os.listdir(dbpedia_dir) if f.endswith(".zip")]
    if zips:
        extract_zip(path("data", "dbpedia", zips[0]), dbpedia_dir)

    # Delete any extra CSVs that aren't DBPEDIA_test.csv
    for f in os.listdir(dbpedia_dir):
        if f.endswith(".csv") and f != "DBPEDIA_test.csv":
            os.remove(os.path.join(dbpedia_dir, f))
            print(f"  Deleted: {f}")

    if not os.path.exists(out):
        print("  Could not find DBPEDIA_test.csv after extraction.")
    else:
        print(f"  Done: {out}")


def download_rcv1():
    print("\n[RCV1]")
    out = path("data", "rcv1", "rcv1.csv")

    if already_exists("RCV1", out):
        return

    print("  Fetching via sklearn (no Kaggle needed) ...")
    run([sys.executable, path("data", "rcv1", "import_rcv1.py")])
    print(f"  Done: {out}")


def download_wos():
    print("\n[Web of Science]")
    out = path("data", "WebOfScience", "Data.xlsx")

    if already_exists("WoS", out):
        return

    wos_dir = path("data", "WebOfScience")

    # Check for an existing zip (manually downloaded or previously saved)
    existing_zips = [f for f in os.listdir(wos_dir) if f.endswith(".zip")]
    if existing_zips:
        zip_path = os.path.join(wos_dir, existing_zips[0])
        print(f"  Found existing zip: {zip_path}")
    else:
        zip_path = os.path.join(wos_dir, "wos.zip")
        url = "https://data.mendeley.com/public-api/zip/9rw3vkcfy4/download/6"
        print("  Downloading from Mendeley ...")
        if not HAS_REQUESTS:
            print("  requests not installed. Run: pip install requests")
            sys.exit(1)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = _requests.get(url, headers=headers, stream=True)
        if response.status_code != 200:
            print(f"  Download failed: HTTP {response.status_code}")
            sys.exit(1)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract Data.xlsx — handle zip-within-a-zip from Mendeley
    print(f"  Extracting Data.xlsx ...")
    with zipfile.ZipFile(zip_path, "r") as outer:
        candidates = [n for n in outer.namelist() if os.path.basename(n) == "Data.xlsx"]
        if candidates:
            with outer.open(candidates[0]) as src, open(out, "wb") as dst:
                dst.write(src.read())
        else:
            # Look for inner zip
            inner_zips = [n for n in outer.namelist() if n.endswith(".zip")]
            if not inner_zips:
                print("  Could not find Data.xlsx in zip. Check data/WebOfScience/ manually.")
                sys.exit(1)
            import io
            with outer.open(inner_zips[0]) as inner_file:
                with zipfile.ZipFile(io.BytesIO(inner_file.read())) as inner:
                    inner_candidates = [n for n in inner.namelist() if os.path.basename(n) == "Data.xlsx"]
                    if not inner_candidates:
                        print("  Could not find Data.xlsx in inner zip. Check data/WebOfScience/ manually.")
                        sys.exit(1)
                    with inner.open(inner_candidates[0]) as src, open(out, "wb") as dst:
                        dst.write(src.read())

    if os.path.exists(out):
        print(f"  Done: {out}")
    else:
        print("  Could not find Data.xlsx after extraction. Check data/WebOfScience/ manually.")

# ========================
# Main
# ========================

DATASETS = {
    "arxiv":   download_arxiv,
    "amazon":  download_amazon,
    "dbpedia": download_dbpedia,
    "rcv1":    download_rcv1,
    "wos":     download_wos,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess benchmark datasets.")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Datasets to download (default: all)"
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if "all" in args.datasets else args.datasets

    print("=" * 50)
    print("NCEAS NLP -- Data Setup")
    print("=" * 50)

    for name in targets:
        DATASETS[name]()

    print("\n" + "=" * 50)
    print("Done. Run the benchmark pipeline once data is in place.")
    print("See REPRODUCIBILITY.md for next steps.")
    print("=" * 50)
