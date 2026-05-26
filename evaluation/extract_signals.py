import pickle
import numpy as np
import torch
from embedding.clip_embedder import CLIPEmbedder

def cosine(a, b):
    return float(np.dot(a, b))

if __name__ == "__main__":

    with open("data/processed/pairs.pkl", "rb") as f:
        pairs = pickle.load(f)

    with open("data/processed/phashes.pkl", "rb") as f:
        ph = pickle.load(f)

    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        rs = pickle.load(f)

    clipper = CLIPEmbedder(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    X, y = [], []

    for a, b, label in pairs:
        ph_d = ph[a] - ph[b]
        resnet_sim = float(np.dot(rs[a], rs[b]))

        # compute CLIP only if ambiguous range
        if 0.5 < resnet_sim < 0.8:
            clip_sim = cosine(
                clipper.embed(a).numpy(),
                clipper.embed(b).numpy()
            )
        else:
            clip_sim = -1.0

        X.append([ph_d, resnet_sim, clip_sim])
        y.append(label)

    with open("data/processed/decision_dataset.pkl", "wb") as f:
        pickle.dump((X, y), f)

    print("Decision dataset size:", len(X))
