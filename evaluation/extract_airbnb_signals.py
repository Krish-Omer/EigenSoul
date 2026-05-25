import pickle
import numpy as np
import torch
import pathlib
from evaluation.gating import gate
from embedding.clip_embedder import CLIPEmbedder

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

PAIR_PATH  = PROJECT_ROOT / "data/processed/airbnb_pairs.pkl"
PHASH_PATH = PROJECT_ROOT / "data/processed/phashes.pkl"
EMB_PATH   = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
OUT_PATH   = PROJECT_ROOT / "data/processed/airbnb_pair_signals.pkl"


def cosine(a, b):
    return float(np.dot(a, b))  # already L2 normalized


if __name__ == "__main__":
    with open(PAIR_PATH, "rb") as f:
        pairs = pickle.load(f)

    with open(PHASH_PATH, "rb") as f:
        ph = pickle.load(f)

    with open(EMB_PATH, "rb") as f:
        rs = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipper = CLIPEmbedder(device=device)

    records = []
    skipped = 0

    for q, p, label in pairs:
        if q not in ph or p not in ph:
            skipped += 1
            continue
        if q not in rs or p not in rs:
            skipped += 1
            continue

        ph_d    = abs(ph[q] - ph[p])
        res_sim = cosine(rs[q], rs[p])

        # Wide pre-filter only — no ACCEPT shortcut
        if gate(ph_d) == "REJECT":
            skipped += 1
            continue

        # Always compute CLIP — logistic regression needs all 3 signals
        clip_sim = cosine(
            clipper.embed(q).numpy(),
            clipper.embed(p).numpy()
        )

        records.append({
            "phash_dist": ph_d,
            "resnet_sim": res_sim,
            "clip_sim":   clip_sim,
            "label":      label
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] Saved {len(records)} records  (skipped {skipped})")