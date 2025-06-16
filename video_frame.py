from dataclasses import dataclass, field
from functools import cached_property

import cv2
import numpy as np
from pytesseract import pytesseract  # type: ignore

from config import DEBUG_MODE
from debug_utils import show_image_resized
from hashing_utils import compute_phash
from image_utils import add_black_border


@dataclass
class VideoFrame:
    """
    Represents a single frame from a video along with its metadata and derived information.

    Each VideoFrame includes the original frame (`full_frame`), its index (`frame_number`),
    and optional precomputed Region of Interest (RoI) coordinates. It lazily computes
    derived values like the RoI image, its perceptual hash, and OCR data for keyword detection.
    """

    full_frame: np.ndarray
    frame_number: int
    roi_coordinates: tuple[int, int, int, int] | None = None

    def compute_roi_coordinates(self) -> tuple[int, int, int, int]:
        """
        Computes the bounding box of the largest outer contour in the frame, assumed to be the slide.

        A black border is added before edge detection, if the image only contains the slide.

        Returns: A tuple containing the (largest_contour_top_left_x, largest_contour_top_left_y,
        largest_contour_bottom_right_x, largest_contour_bottom_right_y) coordinates of the Region of Interest.

        Raises:
            ValueError: If no valid contour is found or the region is too small to be a slide.
        """

        # Add a black border to ensure edge detection works, when the image only contains the slide.
        padding: int = 5
        padded_image: np.ndarray = add_black_border(self.full_frame, padding)

        gray_scale_image: np.ndarray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        canny_edges: np.ndarray = cv2.Canny(gray_scale_image, 50, 150)

        # Using cv2.RETR_EXTERNAL to detect only the outer contours,
        # inner contours are not relevant for finding the biggest rectangle (the slide).
        contours_sequence, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours: tuple[np.ndarray, ...] = tuple(contours_sequence)

        if not contours:
            raise ValueError("No contours found for ROI detection.")

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

            show_image_resized(self.full_frame, "Orignal Image")
            show_image_resized(padded_image, "Padded Image")
            show_image_resized(gray_scale_image, "Gray Scale Image")
            show_image_resized(canny_edges, "Canny Edge Image")
            show_image_resized(contour_visualization, "Contour Image")
            show_image_resized(visualization_image, "Contour Bounding Boxes Image")
            show_image_resized(roi, "Extracted RoI Image")

        largest_contour_area: int = largest_contour_width * largest_contour_height

        if largest_contour_width < 500 or largest_contour_height < 500 or largest_contour_area < 5000:
            raise ValueError("Detected ROI is too small to be valid.")

        img_height, img_width = self.full_frame.shape[:2]
        return (
            max(0, largest_contour_top_left_x - padding),
            max(0, largest_contour_top_left_y - padding),
            min(img_width, largest_contour_bottom_right_x - padding),
            min(img_height, largest_contour_bottom_right_y - padding),
        )

    @cached_property
    def roi_frame(self) -> np.ndarray:
        """
        Extracts and caches the Region of Interest (RoI) from the full frame.

        If RoI coordinates are not already set, they are computed from the frame content.

        Returns:
            A NumPy array representing the cropped RoI image.
        """
        if self.roi_coordinates is None:
            self.roi_coordinates = self.compute_roi_coordinates()

        roi_top_left_x, roi_top_left_y, roi_bottom_right_x, roi_bottom_right_y = self.roi_coordinates
        return self.full_frame[roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x]

    @cached_property
    def roi_hash(self) -> str:
        """
        Computes and caches the perceptual hash (pHash) of the RoI image.

        Returns:
            A string representing the image hash, used for slide matching.
        """
        return compute_phash(self.roi_frame)

    @cached_property
    def ocr_data(self) -> dict[str, list[str | int]]:
        """
        Applies OCR to the full frame and extracts word-level metadata.

        Returns:
            A dictionary containing extracted text and bounding box data from Tesseract OCR.
            Keys include 'text', 'conf', 'left', 'top', 'width', 'height'.
        """
        frame_rgb = cv2.cvtColor(self.full_frame, cv2.COLOR_BGR2RGB)
        return pytesseract.image_to_data(
            frame_rgb,
            output_type=pytesseract.Output.DICT,
            config="--psm 11 --oem 3",
        )
