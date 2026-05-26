import pickle
import numpy as np

from index.retrieve import retrieve
from evaluation.decision import decide

def cosine(a, b):
    return float(np.dot(a, b))


# load caches
with open("data/processed/phashes.pkl", "rb") as f:
    ph = pickle.load(f)

with open("data/processed/resnet_embeddings.pkl", "rb") as f:
    rs = pickle.load(f)

QUERY = "data/raw/copydays/strong/200001.jpg"

q_emb = rs[QUERY]

candidates = retrieve(q_emb, k=10)

for path, resnet_sim in candidates:
    ph_d = ph[QUERY] - ph[path]

    # CLIP sim unknown yet → pass dummy for now
    decision = decide(ph_d, resnet_sim, clip_sim=0.0)

    print(path, resnet_sim, ph_d, decision)
