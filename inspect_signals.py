import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def inspect(name, path):
    if not path.exists():
        print(f"{name}: file not found at {path}")
        return
    with open(path, "rb") as f:
        records = pickle.load(f)

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]

    print(f"\n{'='*50}")
    print(f"{name}: {len(records)} records  |  positives: {len(pos)}  negatives: {len(neg)}")

    ph_pos = [r["phash_dist"] for r in pos]
    ph_neg = [r["phash_dist"] for r in neg]
    rs_pos = [r["resnet_sim"] for r in pos]
    rs_neg = [r["resnet_sim"] for r in neg]

    print(f"\npHash distance:")
    print(f"  positives → min:{min(ph_pos):.1f}  max:{max(ph_pos):.1f}  mean:{np.mean(ph_pos):.1f}")
    print(f"  negatives → min:{min(ph_neg):.1f}  max:{max(ph_neg):.1f}  mean:{np.mean(ph_neg):.1f}")

    print(f"\nResNet similarity:")
    print(f"  positives → min:{min(rs_pos):.4f}  max:{max(rs_pos):.4f}  mean:{np.mean(rs_pos):.4f}")
    print(f"  negatives → min:{min(rs_neg):.4f}  max:{max(rs_neg):.4f}  mean:{np.mean(rs_neg):.4f}")

    # Distribution of positives by ResNet score
    print(f"\nPositive ResNet distribution:")
    buckets = [(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    for lo, hi in buckets:
        count = sum(lo <= r < hi for r in rs_pos)
        print(f"  {lo:.1f}-{hi:.1f}: {count} pairs")

    print(f"\nNegative ResNet distribution:")
    for lo, hi in buckets:
        count = sum(lo <= r < hi for r in rs_neg)
        print(f"  {lo:.1f}-{hi:.1f}: {count} pairs")

inspect("Airbnb",   PROJECT_ROOT / "data/processed/airbnb_pair_signals.pkl")
inspect("Copydays", PROJECT_ROOT / "data/processed/copydays_pair_signals.pkl")