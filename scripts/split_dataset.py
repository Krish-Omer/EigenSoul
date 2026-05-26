from pathlib import Path
import random

def extract_copydays_ids(copydays_root):
    original_dir = Path(copydays_root) / "original"
    ids = []

    for img in original_dir.glob("*.jpg"):
        image_id = img.stem[:4]  # XXXX from XXXX00
        ids.append(image_id)

    return sorted(list(set(ids)))


def split_ids(ids, seed=42):
    random.seed(seed)
    random.shuffle(ids)

    n = len(ids)
    train = ids[: int(0.7 * n)]
    val   = ids[int(0.7 * n) : int(0.85 * n)]
    test  = ids[int(0.85 * n) :]

    return train, val, test


def save(ids, path):
    path.write_text("\n".join(ids))


if __name__ == "__main__":
    copydays_root = "data/raw/copydays"

    ids = extract_copydays_ids(copydays_root)
    train, val, test = split_ids(ids)

    split_dir = Path("data/splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    save(train, split_dir / "train_ids.txt")
    save(val,   split_dir / "val_ids.txt")
    save(test,  split_dir / "test_ids.txt")

    print("Splits created:")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
