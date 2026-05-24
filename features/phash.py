from PIL import Image
import imagehash

def compute_phash(image_path):
    """
    Compute 64-bit perceptual hash for an image.
    Returns an imagehash object.
    """
    img = Image.open(image_path).convert("RGB")
    return imagehash.phash(img)


def hamming_distance(hash1, hash2):
    """
    Compute Hamming distance between two pHashes.
    """
    return hash1 - hash2
