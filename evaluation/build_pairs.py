import pickle
import numpy as np
from pathlib import Path

K_NEG = 10  # negatives per query

def build_pairs():
    with open("data/processed/phashes.pkl", "rb") as f:
        ph = pickle.load(f)

    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        rs = pickle.load(f)

    queries = list(Path("data/raw/copydays/strong").glob("*.jpg"))
    pairs = []

    for q in queries:
        q = str(q)
        stem = Path(q).stem
        block_id = stem[:4]
        true_match = f"data/raw/copydays/original/{block_id}00.jpg"

        # positive pair
        pairs.append((q, true_match, 1))

        # negative pairs = top-K ResNet retrieved (excluding true)
        q_emb = rs[q]
        q_emb = q_emb / np.linalg.norm(q_emb)

        sims = []
        for p, emb in rs.items():
            if "copydays/original" not in p:
                continue
            sim = float(np.dot(q_emb, emb))
            sims.append((p, sim))

        sims.sort(key=lambda x: -x[1])

        for p, _ in sims:
            if p != true_match:
                pairs.append((q, p, 0))
            if len(pairs) >= K_NEG + 1:
                break

    out = Path("data/processed/pairs.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(pairs, f)

    print("Saved pairs:", len(pairs))


if __name__ == "__main__":
    build_pairs()
