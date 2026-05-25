import pickle
import numpy as np
from pathlib import Path
 
_bundle = None
 
def _load():
    global _bundle
    if _bundle is None:
        with open(Path("data/processed/decider.pkl"), "rb") as f:
            _bundle = pickle.load(f)
    return _bundle
 
def decide(phash_dist, resnet_sim, clip_sim):
    clf = _load()["clf"]
    x   = np.array([[phash_dist, resnet_sim, clip_sim]])
    return int(clf.predict(x)[0])
