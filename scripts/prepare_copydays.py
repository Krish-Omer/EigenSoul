# scripts/prepare_copydays.py
# Builds Copydays duplicate pairs for training.
# Pairs each attacked image with its original counterpart.

from pathlib import Path
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COPYDAYS_ROOT = PROJECT_ROOT / "data/raw/copydays"
OUT_PATH = PROJECT_ROOT / "data/processed/copydays_pairs.pkl"


def stem(p: Path) -> str:
    """Base name without extension, e.g. 'image0001'"""
    return p.stem


def normalize(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    orig_dir        = COPYDAYS_ROOT / "original"
    strong_dir      = COPYDAYS_ROOT / "strong"
    jpeg_root       = COPYDAYS_ROOT / "scale_and_jpeg" / "jpegqual"

    # Build lookup: stem -> original path
    originals = {stem(p): p for p in sorted(orig_dir.glob("*.jpg"))}
    print(f"Originals found: {len(originals)}")

    pairs = []

    # ── JPEG quality attacks ──────────────────────────────────
    # Each quality subfolder (3,5,8,...,75) contains attacked copies
    # of the originals — filename stem matches the original
    jpeg_pos = 0
    for qfolder in sorted(jpeg_root.iterdir()):
        if not qfolder.is_dir():
            continue
        for attacked in sorted(qfolder.glob("*.jpg")):
            s = stem(attacked)
            if s in originals:
                pairs.append((normalize(originals[s]), normalize(attacked), 1))
                jpeg_pos += 1

    print(f"JPEG attack pairs: {jpeg_pos}")

    # ── Strong attacks ────────────────────────────────────────
    # strong/ filenames: image0001_1.jpg, image0001_2.jpg etc.
    # stem before last '_' matches original stem
    strong_pos = 0
    for attacked in sorted(strong_dir.glob("*.jpg")):
        base = "_".join(stem(attacked).split("_")[:-1])  # strip _1, _2 suffix
        if base in originals:
            pairs.append((normalize(originals[base]), normalize(attacked), 1))
            strong_pos += 1
        else:
            # some strong images may be standalone — pair with themselves as negatives skipped
            pass

    print(f"Strong attack pairs: {strong_pos}")

    # ── Hard negatives: cross-image pairs from originals ─────
    # Pick pairs of different originals as negatives
    orig_list = sorted(originals.values())
    neg_count = 0
    # Sample every 5th pair to keep negatives balanced (~1:2 pos:neg)
    target_negs = len(pairs) * 2
    step = max(1, len(orig_list) * (len(orig_list) - 1) // 2 // target_negs)
    i = 0
    for a_idx in range(len(orig_list)):
        for b_idx in range(a_idx + 1, len(orig_list)):
            if i % step == 0 and neg_count < target_negs:
                pairs.append((normalize(orig_list[a_idx]), normalize(orig_list[b_idx]), 0))
                neg_count += 1
            i += 1

    print(f"Hard negatives added: {neg_count}")

    pos = sum(1 for _, _, l in pairs if l == 1)
    neg = sum(1 for _, _, l in pairs if l == 0)
    print(f"\nTotal pairs: {len(pairs)}  |  positives: {pos}  negatives: {neg}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("wb") as f:
        pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] Saved to {OUT_PATH}")