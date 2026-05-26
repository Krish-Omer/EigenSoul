from pathlib import Path

def load_ids(path):
    return set(line.strip() for line in Path(path).read_text().splitlines())

TRAIN_IDS = load_ids("data/splits/train_ids.txt")
VAL_IDS   = load_ids("data/splits/val_ids.txt")
TEST_IDS  = load_ids("data/splits/test_ids.txt")

def image_id(p: Path):
    return p.stem[:4]

def assign_split(orig_path: Path):
    iid = image_id(orig_path)

    if iid in TRAIN_IDS:
        return "train"
    if iid in VAL_IDS:
        return "val"
    if iid in TEST_IDS:
        return "test"

    return None
