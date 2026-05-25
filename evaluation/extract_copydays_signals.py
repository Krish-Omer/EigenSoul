# evaluation/extract_copydays_signals.py
#
# Uses WIDE gates during training signal extraction only.
# This ensures hard pairs (print-scan, heavy JPEG) reach CLIP
# so the learned thresholds are informed by difficult cases.
#
# At inference time, the learned tight gates still apply —
# CLIP is still only called for ambiguous cases (README unchanged).

import pickle
import numpy as np
import torch
import pathlib
from embedding.clip_embedder import CLIPEmbedder

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

PAIR_PATH  = PROJECT_ROOT / "data/processed/copydays_pairs.pkl"
PHASH_PATH = PROJECT_ROOT / "data/processed/phashes.pkl"
EMB_PATH   = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
OUT_PATH   = PROJECT_ROOT / "data/processed/copydays_pair_signals.pkl"

# Wide gates for training extraction only
# Catches hard print-scan pairs (pHash up to 28, ResNet as low as 0.35)
T_HASH_WIDE = 30
T_HIGH_WIDE = 0.95   # only skip CLIP for very confident accepts
T_LOW_WIDE  = 0.35   # only skip CLIP for very confident rejects


def cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    with open(PAIR_PATH, "rb") as f:
        pairs = pickle.load(f)
    with open(PHASH_PATH, "rb") as f:
        ph = pickle.load(f)
    with open(EMB_PATH, "rb") as f:
        rs = pickle.load(f)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    clipper = CLIPEmbedder(device=device)

    records  = []
    skipped  = 0
    clip_called = 0
    gated_out   = 0

    for q, p, label in pairs:
        if q not in ph or p not in ph or q not in rs or p not in rs:
            skipped += 1
            continue

        ph_d    = abs(ph[q] - ph[p])
        res_sim = cosine(rs[q], rs[p])

        # Wide pre-filter — only skip truly hopeless pairs
        if ph_d > T_HASH_WIDE:
            gated_out += 1
            continue

        # Compute CLIP for ambiguous zone (wide definition)
        if res_sim >= T_HIGH_WIDE:
            clip_sim = -1.0   # confident accept — skip CLIP
        elif res_sim <= T_LOW_WIDE:
            clip_sim = -1.0   # confident reject — skip CLIP
        else:
            clip_sim = cosine(
                clipper.embed(q).numpy(),
                clipper.embed(p).numpy()
            )
            clip_called += 1

        records.append({
            "phash_dist": ph_d,
            "resnet_sim": res_sim,
            "clip_sim":   clip_sim,
            "label":      label
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    pos = sum(1 for r in records if r["label"] == 1)
    neg = sum(1 for r in records if r["label"] == 0)

    print(f"[DONE] Saved {len(records)} records  (skipped:{skipped}  gated_out:{gated_out})")
    print(f"       positives:{pos}  negatives:{neg}")
    print(f"       CLIP called for {clip_called} pairs  ({100*clip_called/max(1,len(records)):.1f}%)")
    print(f"\nWide gates used for extraction:")
    print(f"  T_hash_wide : {T_HASH_WIDE}  (learned T_hash was 12)")
    print(f"  T_high_wide : {T_HIGH_WIDE}  (learned T_high was 0.8912)")
    print(f"  T_low_wide  : {T_LOW_WIDE}   (learned T_low  was 0.7235)")
    print(f"\nAt inference, learned tight gates still apply — README unchanged.")