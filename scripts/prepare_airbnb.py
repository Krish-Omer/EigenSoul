from pathlib import Path
from collections import defaultdict
import random
import shutil

IMAGES_DIR = Path("data/raw/airbnb/images")
QUERIES_DIR = Path("data/raw/airbnb/queries")

def listing_id(path: Path) -> str:
    # bathroom_berlin_37836_2.jpg → bathroom_berlin_37836
    return path.stem.rsplit("_", 1)[0]

def main():
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)

    for img in IMAGES_DIR.glob("*.jpg"):
        groups[listing_id(img)].append(img)

    moved = 0

    for lid, imgs in groups.items():
        if len(imgs) < 2:
            continue  # cannot form duplicate query

        q = random.choice(imgs)
        shutil.move(str(q), QUERIES_DIR / q.name)
        moved += 1

    print(f"[OK] Created {moved} Airbnb queries")

if __name__ == "__main__":
    main()
