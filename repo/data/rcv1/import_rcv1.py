import os
import sys
import pandas as pd
import nltk
from nltk.corpus import reuters

os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(current_dir, "src")):
    parent = os.path.abspath(os.path.join(current_dir, ".."))
    if parent == current_dir:
        raise FileNotFoundError("Could not find repo root.")
    current_dir = parent

OUT_DIR = os.path.join(current_dir, "data", "rcv1")
os.makedirs(OUT_DIR, exist_ok=True)

filename = os.path.join(OUT_DIR, "rcv1.csv")

def main():
    print("Fetching NLTK Reuters...")
    nltk.download('reuters')
    
    file_ids = reuters.fileids()
    data_list = []
    for f in file_ids:
        text = reuters.raw(f).replace('\n', ' ').strip()
        cats = reuters.categories(f)
        if len(cats) < 2: continue

        data_list.append({
            "item_id": f,
            "topic": text,
            "category_0": cats[0],
            "category_1": cats[1],
        })

    df = pd.DataFrame(data_list)
    df = df[['item_id', 'topic', 'category_0', 'category_1']]
    df.to_csv(filename, index=False)
    print(f"Success! Data saved to {filename}")

if __name__ == "__main__":
    main()
