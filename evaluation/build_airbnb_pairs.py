# evaluation/build_airbnb_pairs.py

import pickle
from pathlib import Path

from index.retrieve_airbnb import retrieve_airbnb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMB_PATH = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
OUT_PATH = PROJECT_ROOT / "data/processed/airbnb_pairs.pkl"
QUERY_DIR = PROJECT_ROOT / "data/raw/airbnb/queries"


def norm(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def base_id(p: str) -> str:
    # berlin_364497299_2.jpg -> berlin_364497299
    return "_".join(Path(p).stem.split("_")[:-1])


if __name__ == "__main__":
    with open(EMB_PATH, "rb") as f:
        embeddings = pickle.load(f)

    pairs = []
    K = 20

    for q in QUERY_DIR.glob("*.jpg"):
        qk = norm(q)
        if qk not in embeddings:
            continue

        q_id = base_id(qk)
        q_emb = embeddings[qk]

        results = retrieve_airbnb(q_emb, k=K)

        positive_added = False

        for path, _ in results:
            if base_id(path) == q_id and not positive_added:
                pairs.append((qk, path, 1))   # positive
                positive_added = True
            elif base_id(path) != q_id:
                pairs.append((qk, path, 0))   # negative

        # safety: only keep queries with a positive
        if not positive_added:
            pairs = pairs[:-K]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] Saved {len(pairs)} Airbnb pairs")
