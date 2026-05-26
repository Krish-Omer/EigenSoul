import faiss
import numpy as np
import pickle

INDEX_PATH = "data/processed/faiss.index"
IDMAP_PATH = "data/processed/index_ids.pkl"

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(IDMAP_PATH, "rb") as f:
        ids = pickle.load(f)
    return index, ids


def retrieve(query_emb, k=100):
    index, ids = load_index()

    # 🔑 FIX: normalize query
    query_emb = query_emb.astype("float32")
    query_emb = query_emb / np.linalg.norm(query_emb)

    D, I = index.search(query_emb.reshape(1, -1), k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append((ids[idx], float(score)))

    return results

