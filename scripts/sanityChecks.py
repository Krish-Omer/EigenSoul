import numpy as np
import pickle

def cosine(a, b):
    return float(np.dot(a, b))

with open("data/processed/resnet_embeddings.pkl", "rb") as f:
    emb = pickle.load(f)

orig = "data/raw/copydays/original/200000.jpg"
strong = "data/raw/copydays/strong/200001.jpg"
landmark = "data/raw/landmarks/3/003a0cc8aa8d08fa.jpg"

sim_orig_strong = cosine(emb[orig], emb[strong])
sim_orig_landmark = cosine(emb[orig], emb[landmark])

print("orig ↔ strong:", sim_orig_strong)
print("orig ↔ landmark:", sim_orig_landmark)
