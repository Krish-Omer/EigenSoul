import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_signals(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    airbnb   = load_signals(PROJECT_ROOT / "data/processed/airbnb_pair_signals.pkl")
    copydays = load_signals(PROJECT_ROOT / "data/processed/copydays_pair_signals.pkl")
    combined = airbnb + copydays

    print(f"Airbnb:   {len(airbnb)} records")
    print(f"Copydays: {len(copydays)} records")
    print(f"Combined: {len(combined)} records")

    X, y = [], []
    for r in combined:
        clip = r["clip_sim"] if r["clip_sim"] != -1.0 else 0.0
        X.append([r["phash_dist"], r["resnet_sim"], clip])
        y.append(r["label"])

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset: {len(X)} samples  |  positives: {int(y.sum())}  negatives: {int((1-y).sum())}")

    # Shuffle before split
    idx = np.random.RandomState(42).permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    ph_pos  = X_train[y_train == 1, 0]
    rs_pos  = X_train[y_train == 1, 1]
    rs_neg  = X_train[y_train == 0, 1]

    # ── Distribution-based threshold learning ─────────────────
    # T_hash: cover 99th percentile of positive pHash distances
    # (don't use max to avoid outliers)

    #T_hash = float(np.percentile(ph_pos, 99)) + 2
    T_hash = float(np.max(ph_pos)) + 2

    # T_high: ResNet above which negatives are essentially zero
    # = 95th percentile of negative ResNet scores (safe accept zone)
    T_high = float(np.percentile(rs_neg, 95))

    # T_low: ResNet below which positives are essentially zero
    # = 5th percentile of positive ResNet scores (safe reject zone)
    T_low  = float(np.percentile(rs_pos, 5))

    # Sanity checks
    T_hash = max(T_hash, 20.0)   # never go below original learned value (12)
    T_high = min(T_high, 0.97)   # never be too permissive
    T_low  = max(T_low,  0.35)   # never be too aggressive

    if T_low >= T_high:
        T_low = T_high - 0.15

    print(f"\nLearned thresholds (distribution-based):")
    print(f"  T_hash : {T_hash:.1f}  (covers 99th pctile of positive pHash)")
    print(f"  T_high : {T_high:.4f}  (95th pctile of negative ResNet → safe ACCEPT)")
    print(f"  T_low  : {T_low:.4f}  (5th pctile of positive ResNet → safe REJECT)")
    print(f"  Ambiguous zone: {T_low:.4f} — {T_high:.4f}")

    # ── Train LR for ambiguous zone ────────────────────────────
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    f1    = f1_score(y_val, preds, pos_label=1)
    print(f"\nValidation F1: {f1:.4f}")
    print(classification_report(y_val, preds, target_names=["unique", "duplicate"]))

    bundle = {"clf": clf, "T_hash": T_hash, "T_high": T_high, "T_low": T_low}
    with open(PROJECT_ROOT / "data/processed/decider.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("[DONE] Decider + learned thresholds saved")