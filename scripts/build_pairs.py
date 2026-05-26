from pathlib import Path
import random

COPY_ORIG = Path("data/raw/copydays/original")
COPY_STRONG = Path("data/raw/copydays/strong")
LANDMARKS = Path("data/raw/landmarks")

random.seed(42)

def image_id(p: Path):
    # XXXX00.jpg → XXXX
    return p.stem[:4]


def build_positive_pairs():
    """
    (original, strong, label=1)
    """
    orig_map = {image_id(p): p for p in COPY_ORIG.glob("*.jpg")}
    pairs = []

    for s in COPY_STRONG.glob("*.jpg"):
        iid = image_id(s)
        if iid in orig_map:
            pairs.append((orig_map[iid], s, 1))

    return pairs


def build_negative_pairs(n):
    """
    (original, landmark, label=0)
    """
    origs = list(COPY_ORIG.glob("*.jpg"))
    landmarks = list(LANDMARKS.rglob("*.jpg"))

    pairs = []
    while len(pairs) < n:
        o = random.choice(origs)
        l = random.choice(landmarks)
        pairs.append((o, l, 0))

    return pairs


if __name__ == "__main__":
    pos = build_positive_pairs()
    neg = build_negative_pairs(len(pos))

    print("Positive pairs:", len(pos))
    print("Negative pairs:", len(neg))

# NOTE:
# This script was used for pairwise signal analysis.
# Final evaluation uses retrieval-based protocol (see evaluate_retrieval.py).
