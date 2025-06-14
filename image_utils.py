import cv2
import numpy as np


def apply_roi_crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    roi_top_left_x, roi_top_left_y, roi_bottom_right_x, roi_bottom_right_y = roi
    return frame[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x]


def add_black_border(image: np.ndarray, padding_size: int = 50):
    """
    Add a black border to an image with given padding size.

    Args:
        image: Input image (BGR format).
        padding_size: Width of the border to add on all sides (in pixels). Defaults to 50.

    Returns:
        Image with added black borders on all sides.
    """
    return cv2.copyMakeBorder(
        image,
        top=padding_size,
        bottom=padding_size,
        left=padding_size,
        right=padding_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
