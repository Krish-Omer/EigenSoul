import pickle
from sklearn.metrics import precision_recall_fscore_support
from evaluation.decision import decide

def evaluate(records, T_hash, T_low, T_high, T_clip):
    y_true, y_pred = [], []

    for r in records:
        pred = decide(
            r["phash"],
            r["resnet"],
            r["clip"] if r["clip"] is not None else -1.0,
            T_hash=T_hash,
            T_low=T_low,
            T_high=T_high,
            T_clip=T_clip,
        )
        y_true.append(r["label"])
        y_pred.append(pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return p, r, f1


if __name__ == "__main__":
    with open("data/processed/pair_signals.pkl", "rb") as f:
        data = pickle.load(f)

    val = [r for r in data if r["split"] == "val"]

    best = (0, None)

    for T_hash in [8, 10, 12, 14]:
        for T_low in [0.50, 0.55, 0.60]:
            for T_high in [0.75, 0.80, 0.85]:
                for T_clip in [0.80, 0.85, 0.90]:
                    p, r, f1 = evaluate(
                        val, T_hash, T_low, T_high, T_clip
                    )
                    if f1 > best[0]:
                        best = (f1, (T_hash, T_low, T_high, T_clip))

    print("BEST F1:", best[0])
    print("BEST THRESHOLDS:", best[1])
    print("Evaluation complete.")