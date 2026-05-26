import pickle
from pathlib import Path
from features.phash import hamming_distance

PHASH_PATH = Path("data/processed/phashes.pkl")

def image_id_from_path(p):
    # XXXXYY.jpg → XXXX
    return Path(p).stem[:4]


if __name__ == "__main__":
    with open(PHASH_PATH, "rb") as f:
        phashes = pickle.load(f)

    paths = list(phashes.keys())

    same_id_dists = []
    diff_id_dists = []

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            id1 = image_id_from_path(paths[i])
            id2 = image_id_from_path(paths[j])

            d = hamming_distance(phashes[paths[i]], phashes[paths[j]])

            if id1 == id2:
                same_id_dists.append(d)
            else:
                diff_id_dists.append(d)

    print("Same-image pairs:")
    print(f"  mean: {sum(same_id_dists)/len(same_id_dists):.2f}")
    print(f"  min/max: {min(same_id_dists)} / {max(same_id_dists)}")

    print("Different-image pairs:")
    print(f"  mean: {sum(diff_id_dists)/len(diff_id_dists):.2f}")
    print(f"  min/max: {min(diff_id_dists)} / {max(diff_id_dists)}")
