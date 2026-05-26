# index/retrieve_airbnb.py

import faiss
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "data/processed/airbnb_faiss.index"
IDMAP_PATH = PROJECT_ROOT / "data/processed/airbnb_index_ids.pkl"


def load_index():
    index = faiss.read_index(str(INDEX_PATH))
    with open(IDMAP_PATH, "rb") as f:
        ids = pickle.load(f)
    return index, ids


def retrieve_airbnb(query_emb, k=20):
    index, ids = load_index()

    query_emb = query_emb.reshape(1, -1).astype("float32")
    scores, idxs = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        results.append((ids[idx], float(score)))

    return results
