from imagededup.methods import PHash  # type: ignore
import numpy as np


def _compute_phash(image: np.ndarray) -> str:
    phasher = PHash()
    return phasher.encode_image(image_array=image)


def compute_phashes(images: list[np.ndarray]) -> list[str]:
    phasher: PHash = PHash()
    images_hashes: list[str] = [phasher.encode_image(image_array=img) for img in images]
    return images_hashes


def compute_hamming_distances_to_reference(reference_hash: str, comparison_hashes: list[str]) -> list[float]:
    """
    Computes Hamming distances between a reference hash and a list of comparison hashes.

    Args:
        reference_hash: The hash to compare all others against.
        comparison_hashes: A list of hash strings to be compared to the reference hash.

    Returns:
        A list of Hamming distances between the reference hash and each comparison hash.
    """
    phasher = PHash()
    return [
        phasher.hamming_distance(reference_hash, comparison_hash) for comparison_hash in comparison_hashes
    ]
