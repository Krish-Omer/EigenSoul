import pickle
import numpy as np
from pathlib import Path

from scripts.build_pairs import build_positive_pairs, build_negative_pairs
from scripts.assign_split import assign_split
from evaluation.gating import gate
from embedding.clip_embedder import CLIPEmbedder

def cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    # load cached features
    with open("data/processed/phashes.pkl", "rb") as f:
        phashes = pickle.load(f)

    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        resnet = pickle.load(f)

    clipper = CLIPEmbedder()

    pos = build_positive_pairs()
    neg = build_negative_pairs(len(pos))
    pairs = pos + neg

    records = []

    for A, B, label in pairs:
        split = assign_split(A)
        if split is None:
            continue

        ph_d = phashes[str(A)] - phashes[str(B)]
        rs_s = cosine(resnet[str(A)], resnet[str(B)])

        clip_s = None
        if gate(ph_d, rs_s) == "AMBIGUOUS":
            c1 = clipper.embed(A).numpy()
            c2 = clipper.embed(B).numpy()
            clip_s = cosine(c1, c2)

        records.append({
            "phash": ph_d,
            "resnet": rs_s,
            "clip": clip_s,
            "label": label,
            "split": split,
        })

    out = Path("data/processed/pair_signals.pkl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "wb") as f:
        pickle.dump(records, f)

    print("Saved records:", len(records))

# Used for exploratory signal analysis only.