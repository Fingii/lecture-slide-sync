import cv2
import numpy as np

from config import DEBUG_MODE
from debug_utils import show_image_resized
from image_utils import add_black_border


def extract_slide_roi_coordinates(
    image: np.ndarray, min_width: int = 500, min_height: int = 500, min_area: int = 5000
) -> tuple[int, int, int, int] | None:
    """
    Loads an image, preprocesses it (grayscale, edge detection, contour extraction),
    and finds the largest rectangular Region of Interest (RoI) which represents the entire slide.

    Args:
        image: Input image as an OpenCV array.
        min_width: Minimum width for a valid RoI.
        min_height: Minimum height for a valid RoI.
        min_area: Minimum area for a valid RoI.
    Returns:
        Extracted RoI coordinates [top-left-x, top-left-y, bottom_right_x, bottom_right_y] or None if no valid region is found.
    """

    # Add a black border to ensure edge detection works, when the image only contains the slide.
    padding: int = 5
    padded_image: np.ndarray = add_black_border(image, padding)

    gray_scale_image: np.ndarray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    canny_edges: np.ndarray = cv2.Canny(gray_scale_image, 50, 150)

    # Using cv2.RETR_EXTERNAL to detect only the outer contours,
    # inner contours are not relevant for finding the biggest rectangle (the slide).
    contours_sequence, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: tuple[np.ndarray, ...] = tuple(contours_sequence)

    if not contours:
        print("No contours found.")
        return None

    # Find the largest contour and extract coordinates
    largest_contour: np.ndarray = max(contours, key=cv2.contourArea)

    largest_contour_top_left_x: int
    largest_contour_top_left_y: int
    largest_contour_width: int
    largest_contour_height: int

    (
        largest_contour_top_left_x,
        largest_contour_top_left_y,
        largest_contour_width,
        largest_contour_height,
    ) = cv2.boundingRect(largest_contour)

    largest_contour_bottom_right_x: int = largest_contour_top_left_x + largest_contour_width
    largest_contour_bottom_right_y: int = largest_contour_top_left_y + largest_contour_height

    if DEBUG_MODE:
        visualization_image: np.ndarray = padded_image.copy()

        # Extract RoI if a valid region is found
        roi: np.ndarray = padded_image[
            largest_contour_top_left_y:largest_contour_bottom_right_y,
            largest_contour_top_left_x:largest_contour_bottom_right_x,
        ]

        for cnt in contours:
            top_left_x_debug, top_left_y_debug, width_debug, height_debug = cv2.boundingRect(cnt)
            cv2.rectangle(
                visualization_image,
                (top_left_x_debug, top_left_y_debug),
                (top_left_x_debug + width_debug, top_left_y_debug + height_debug),
                (0, 0, 255),
                2,
            )

        contour_visualization: np.ndarray = padded_image.copy()
        cv2.drawContours(contour_visualization, contours, -1, (0, 0, 255), 2)

        show_image_resized(image, "Orignal Image")
        show_image_resized(padded_image, "Padded Image")
        show_image_resized(gray_scale_image, "Gray Scale Image")
        show_image_resized(canny_edges, "Canny Edge Image")
        show_image_resized(contour_visualization, "Contour Image")
        show_image_resized(visualization_image, "Contour Bounding Boxes Image")
        show_image_resized(roi, "Extracted RoI Image")

    largest_contour_area: int = largest_contour_width * largest_contour_height
    if (
        largest_contour_width >= min_width
        and largest_contour_height >= min_height
        and largest_contour_area >= min_area
    ):

        image_width = image.shape[1]
        image_height = image.shape[0]

        # Adjust RoI coordinates by removing padding
        return (
            max(0, largest_contour_top_left_x - padding),
            max(0, largest_contour_top_left_y - padding),
            min(image_width, largest_contour_bottom_right_x - padding),
            min(image_height, largest_contour_bottom_right_y - padding),
        )
    else:
        print("Detected RoI is too small to be valid.")
        return None
