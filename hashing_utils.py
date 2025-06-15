from imagededup.methods import PHash  # type: ignore
import numpy as np


def compute_phash(image: np.ndarray) -> str:
    phasher = PHash()
    return phasher.encode_image(image_array=image)


def compute_phashes(images: list[np.ndarray]) -> list[str]:
    phasher: PHash = PHash()
    images_hashes: list[str] = [phasher.encode_image(image_array=img) for img in images]
    return images_hashes


def compute_hamming_distance(hash1: str, hash2: str) -> float:
    """
    Computes the Hamming distance between two hash strings.

    Args:
        hash1: The first hash string.
        hash2: The second hash string.

    Returns:
        The Hamming distance between the two hashes.
    """
    phasher = PHash()
    return phasher.hamming_distance(hash1, hash2)
