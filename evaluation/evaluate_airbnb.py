# evaluation/evaluate_airbnb.py

import pickle
import numpy as np
from pathlib import Path
import torch

from index.retrieve_airbnb import retrieve_airbnb
from evaluation.gating import gate
from evaluation.decision import decide
from embedding.clip_embedder import CLIPEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHASH = PROJECT_ROOT / "data/processed/phashes.pkl"
EMB = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
QUERY_DIR = PROJECT_ROOT / "data/raw/airbnb/queries"


def norm(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def listing_id(p: str) -> str:
    return "_".join(Path(p).stem.split("_")[:-1])


def cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    with open(PHASH, "rb") as f:
        ph = pickle.load(f)
    with open(EMB, "rb") as f:
        rs = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipper = CLIPEmbedder(device=device)

    TP = FP = FN = 0
    K = 20

    for q in QUERY_DIR.glob("*.jpg"):
        qk = norm(q)
        if qk not in rs or qk not in ph:
            continue

        q_emb = rs[qk]
        q_emb /= np.linalg.norm(q_emb)
        q_id = listing_id(qk)

        candidates = retrieve_airbnb(q_emb, k=K)

        accepted = []
        for path, sim in candidates:
            if path not in ph or path not in rs:
                continue

            ph_d = abs(ph[qk] - ph[path])
            g = gate(ph_d, sim)

            if g == "AMBIGUOUS":
                clip_sim = cosine(
                    clipper.embed(qk).numpy(),
                    clipper.embed(path).numpy()
                )
            else:
                clip_sim = -1.0

            if decide(ph_d, sim, clip_sim) == 1:
                accepted.append(path)

        if any(listing_id(p) == q_id for p in accepted):
            TP += 1
        elif len(accepted) > 0:
            FP += 1
        else:
            FN += 1

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("Queries:", TP + FP + FN)
    print("TP:", TP, "FP:", FP, "FN:", FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
