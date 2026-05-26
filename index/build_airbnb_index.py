# index/build_airbnb_index.py

import faiss
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
INDEX_PATH = PROJECT_ROOT / "data/processed/airbnb_faiss.index"
IDMAP_PATH = PROJECT_ROOT / "data/processed/airbnb_index_ids.pkl"

IMAGE_ROOT = "data/raw/airbnb/images"


if __name__ == "__main__":
    with open(EMB_PATH, "rb") as f:
        embeddings = pickle.load(f)

    paths = []
    vectors = []

    for p, emb in embeddings.items():
        if IMAGE_ROOT in p:
            paths.append(p)
            vectors.append(emb)

    vectors = np.stack(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (L2-normalized)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))

    with open(IDMAP_PATH, "wb") as f:
        pickle.dump(paths, f)

    print(f"[DONE] Airbnb index built with {len(paths)} images")
