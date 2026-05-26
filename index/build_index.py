import faiss
import numpy as np
import pickle
from pathlib import Path

INDEX_PATH = "data/processed/faiss.index"
IDMAP_PATH = "data/processed/index_ids.pkl"

def build_index(resnet_embeddings):
    dim = next(iter(resnet_embeddings.values())).shape[0]

    index = faiss.IndexFlatIP(dim)  # cosine via inner product
    ids = []

    for path, emb in resnet_embeddings.items():
        if "copydays/strong" in path:
            continue  # skip query images
        index.add(emb.reshape(1, -1).astype("float32"))
        ids.append(path)

    return index, ids


if __name__ == "__main__":
    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        resnet = pickle.load(f)

    index, ids = build_index(resnet)

    faiss.write_index(index, INDEX_PATH)

    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(ids, f)

    print(f"Indexed {len(ids)} images")
