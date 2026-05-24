# scripts/compute_resnet_embeddings.py — updated to include scale_and_jpeg
from pathlib import Path
import pickle
import torch

from embedding.resnet_embedder import ResNetEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROOTS = [
    PROJECT_ROOT / "data/raw/copydays/original",
    PROJECT_ROOT / "data/raw/copydays/strong",
    PROJECT_ROOT / "data/raw/copydays/scale_and_jpeg/jpegqual",  # added
    PROJECT_ROOT / "data/raw/landmarks",
    PROJECT_ROOT / "data/raw/airbnb/images",
    PROJECT_ROOT / "data/raw/airbnb/queries",
]


def normalize_path(p: Path) -> str:
    return p.resolve().relative_to(PROJECT_ROOT).as_posix()


def collect_images():
    imgs = []
    for r in ROOTS:
        if r.exists():
            imgs.extend(sorted(r.rglob("*.jpg")))
    return imgs


if __name__ == "__main__":
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ResNetEmbedder(device=device)

    embeddings = {}
    images     = collect_images()
    print(f"Embedding {len(images)} images")

    for img in images:
        embeddings[normalize_path(img)] = embedder.embed(img).numpy()

    out = PROJECT_ROOT / "data/processed/resnet_embeddings.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[DONE] ResNet embeddings saved")