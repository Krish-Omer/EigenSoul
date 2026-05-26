import pickle
import numpy as np
import torch

from embedding.clip_embedder import CLIPEmbedder
from evaluation.decision import decide
from evaluation.gating import gate

def cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipper = CLIPEmbedder(device=device)

    # choose known ambiguous-ish case
    A = "data/raw/copydays/original/200000.jpg"
    B = "data/raw/copydays/strong/200001.jpg"

    with open("data/processed/phashes.pkl", "rb") as f:
        ph = pickle.load(f)

    with open("data/processed/resnet_embeddings.pkl", "rb") as f:
        rs = pickle.load(f)

    ph_d = ph[A] - ph[B]
    rs_s = cosine(rs[A], rs[B])

    print("Gate:", gate(ph_d, rs_s))

    if gate(ph_d, rs_s) == "AMBIGUOUS":
        c1 = clipper.embed(A).numpy()
        c2 = clipper.embed(B).numpy()
        clip_sim = cosine(c1, c2)

        print("CLIP sim:", clip_sim)
        print("Final decision:", decide(ph_d, rs_s, clip_sim))
