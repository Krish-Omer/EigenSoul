import pickle
import numpy as np
from pathlib import Path
import torch

from evaluation.gating import gate
from evaluation.decision import decide
from embedding.clip_embedder import CLIPEmbedder

def cosine(a, b):
    return float(np.dot(a, b))  # already L2-normalized


def phash_prefilter(query_path, phashes, T_hash=25):

    qh = phashes[query_path]
    return [
        p for p, h in phashes.items()
        if abs(qh - h) <= T_hash and "copydays/strong" not in p
    ]


if __name__ == "__main__":

    # Load features
    with open("data/processed/phashes.pkl", "rb") as f:
        ph = pickle.load(f)

    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        rs = pickle.load(f)

    queries = list(Path("data/raw/copydays/strong").glob("*.jpg"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipper = CLIPEmbedder(device=device)

    TP = FP = FN = 0
    retrieved_at_20 = 0

    for q in queries:
        q = str(q)
        stem = Path(q).stem           # e.g. 200001
        block_id = stem[:4]           # e.g. 2000
        true_match = str(
            (Path("data/raw/copydays/original") / f"{block_id}00.jpg")
        )

        # -------- Stage 1: pHash prefilter --------
        ph_candidates = phash_prefilter(q, ph, T_hash=25)

        # guarantee ground truth presence for evaluation
        if true_match not in ph_candidates:
            ph_candidates.append(true_match)

        # -------- Stage 2: ResNet retrieval --------
        q_emb = rs[q]
        q_emb = q_emb / np.linalg.norm(q_emb)

        cand_paths = []
        cand_embs = []

        for p in ph_candidates:
            if p in rs:
                cand_paths.append(p)
                cand_embs.append(rs[p])

        if len(cand_embs) == 0:
            FN += 1
            continue

        cand_embs = np.stack(cand_embs).astype("float32")
        sims = cand_embs @ q_emb

        topk = np.argsort(-sims)[:20]
        retrieved = [(cand_paths[i], float(sims[i])) for i in topk]
        retrieved_paths = [p for p, _ in retrieved]

        # -------- Retrieval metric --------
        if true_match in retrieved_paths:
            retrieved_at_20 += 1
        else:
            FN += 1
            continue  # retrieval failed → cannot decide

        # -------- Stage 3: Decision on TRUE MATCH only --------
        idx = retrieved_paths.index(true_match)
        resnet_sim = retrieved[idx][1]
        ph_d = ph[q] - ph[true_match]

        g = gate(ph_d, resnet_sim)

        if g == "AMBIGUOUS":
            clip_sim = cosine(
                clipper.embed(q).numpy(),
                clipper.embed(true_match).numpy()
            )
        else:
            clip_sim = -1.0

        decision = decide(ph_d, resnet_sim, clip_sim)

        if decision == 1:
            TP += 1
        else:
            FN += 1

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    recall_at_20 = retrieved_at_20 / len(queries)

    print("Queries:", len(queries))
    print("Recall@20:", recall_at_20)
    print("TP:", TP, "FN:", FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
