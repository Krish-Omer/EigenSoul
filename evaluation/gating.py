# evaluation/gating.py
# Architecture unchanged — pHash pre-filter + ResNet gates + AMBIGUOUS zone.
# Thresholds are now loaded from the learned bundle, not hardcoded.

import pickle
from pathlib import Path

_bundle = None

def _load():
    global _bundle
    if _bundle is None:
        p = Path("data/processed/decider.pkl")
        with open(p, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle

def gate(phash_dist, resnet_sim):
    """
    Returns one of:
    - 'ACCEPT'     — confidently a duplicate
    - 'REJECT'     — confidently not a duplicate
    - 'AMBIGUOUS'  — needs CLIP + logistic regression
    """
    b = _load()
    T_hash = b["T_hash"]
    T_high = b["T_high"]
    T_low  = b["T_low"]

    if phash_dist > T_hash:
        return "REJECT"

    if resnet_sim >= T_high:
        return "ACCEPT"

    if resnet_sim <= T_low:
        return "REJECT"

    return "AMBIGUOUS"