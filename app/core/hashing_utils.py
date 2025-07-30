import numpy as np
from PIL import Image
import imagehash
import cv2


def compute_phash(image: np.ndarray) -> str:
    """
    Computes the perceptual hash of a single image.

    Args:
        image: The input image as a NumPy array (OpenCV format, BGR).

    Returns:
        A perceptual hash as a hexadecimal string.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    return str(imagehash.phash(pil_image))


def compute_phashes(images: list[np.ndarray]) -> list[str]:
    """
    Computes perceptual hashes for a list of images.

    Args:
        images: A list of OpenCV images (NumPy arrays).

    Returns:
        A list of perceptual hash strings.
    """
    return [compute_phash(img) for img in images]


def compute_hamming_distance(hash1: str, hash2: str) -> float:
    """
    Computes the Hamming distance between two hash strings.

    Args:
        hash1: The first hash string.
        hash2: The second hash string.

    Returns:
        The Hamming distance between the two hashes.
    """
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)
