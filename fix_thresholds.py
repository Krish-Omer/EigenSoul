# Run this to check what the raw percentiles are
import pickle
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

airbnb   = pickle.load(open(PROJECT_ROOT / "data/processed/airbnb_pair_signals.pkl", "rb"))
copydays = pickle.load(open(PROJECT_ROOT / "data/processed/copydays_pair_signals.pkl", "rb"))
combined = airbnb + copydays

pos = [r for r in combined if r["label"] == 1]
neg = [r for r in combined if r["label"] == 0]

ph_pos = np.array([r["phash_dist"] for r in pos])
rs_pos = np.array([r["resnet_sim"] for r in pos])
rs_neg = np.array([r["resnet_sim"] for r in neg])

print("pHash positives percentiles:")
for p in [50, 75, 90, 95, 99, 100]:
    print(f"  {p}th: {np.percentile(ph_pos, p):.1f}")

print("\nResNet positives percentiles:")
for p in [1, 2, 5, 10]:
    print(f"  {p}th: {np.percentile(rs_pos, p):.4f}")

print("\nResNet negatives percentiles:")
for p in [90, 95, 97, 99]:
    print(f"  {p}th: {np.percentile(rs_neg, p):.4f}")