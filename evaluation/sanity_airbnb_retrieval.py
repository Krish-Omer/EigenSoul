# evaluation/sanity_airbnb_retrieval.py

import pickle
from pathlib import Path

from index.retrieve_airbnb import retrieve_airbnb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
QUERY_DIR = PROJECT_ROOT / "data/raw/airbnb/queries"


def norm(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def base_id(p: str) -> str:
    # bathroom_berlin_37836_2.jpg -> bathroom_berlin_37836
    return "_".join(Path(p).stem.split("_")[:-1])


if __name__ == "__main__":
    with open(EMB_PATH, "rb") as f:
        embeddings = pickle.load(f)

    queries = list(QUERY_DIR.glob("*.jpg"))

    K = 20
    hits = 0

    for q in queries:
        q_key = norm(q)

        if q_key not in embeddings:
            print("[WARN] Missing embedding for:", q_key)
            continue

        q_emb = embeddings[q_key]

        results = retrieve_airbnb(q_emb, k=K)
        retrieved_ids = [base_id(p) for p, _ in results]

        if base_id(q_key) in retrieved_ids:
            hits += 1

    recall = hits / len(queries)

    print(f"Queries: {len(queries)}")
    print(f"Recall@{K}: {recall:.4f}")
