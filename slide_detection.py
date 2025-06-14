import cv2
import numpy as np
from PIL import Image

import pymupdf  # type: ignore

from ocr_keyword_detector import are_all_keywords_present  # type: ignore
from debug_utils import show_image_resized
from config import DEBUG_MODE


def convert_pdf_slides_to_images(pdf_path: str) -> list[np.ndarray]:
    """
    Extracts each single slide from a PDF and converts them into OpenCV images.

    This function loads each slide (page) of the given PDF, gets the pixel map of it,
    converts it into a Pillow image, and transforms it into an OpenCV-compatible format (BGR `np.ndarray`).
    The extracted slides are returned as a list of OpenCV images for further processing.

    Args:
        pdf_path: Path to the PDF file containing slides.

    Returns:
        A list of OpenCV images, each representing a slide from the PDF, in order.
    """
    pdf_document: pymupdf.Document = pymupdf.open(pdf_path)
    pdf_slide_images_cv2: list[np.ndarray] = []

    for i in range(pdf_document.page_count):
        current_page_pixmap: pymupdf.Pixmap = pdf_document.get_page_pixmap(i, dpi=200)
        pillow_image: Image.Image = current_page_pixmap.pil_image()
        open_cv_image: np.ndarray = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
        pdf_slide_images_cv2.append(open_cv_image)

    pdf_document.close()
    return pdf_slide_images_cv2


def find_first_slide(video_path: str, max_seconds: int = 30) -> tuple[np.ndarray, int] | None:
    """
    Searches the video for a slide containing specific keywords by checking frames
    until either the slide is found or the maximum duration is reached.

    Args:
        video_path: Path to the video file.
        max_seconds: Maximum duration (in seconds) to scan the video for a slide.

    Returns:
        A tuple containing:
        - The frame containing the text if found
        - The frame count where it was found
        If no slide is found, returns None.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    fps: float = float(cap.get(cv2.CAP_PROP_FPS))
    max_attempts: int = int(max_seconds * fps)
    keywords_to_be_matched: set[str] = {"UNIVERSITY", "FH", "AACHEN", "OF", "APPLIED", "SCIENCES"}

    for frame_count in range(max_attempts):
        success: bool
        frame: np.ndarray
        success, frame = cap.read()

        if not success:
            break

        if are_all_keywords_present(frame, keywords_to_be_matched):
            cap.release()
            return frame, frame_count

    cap.release()
    return None


def add_black_border(image, padding_size=50):
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


def extract_slide_roi_coordinates_from_image(
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


def detect_slide_transitions(video_file_path: str) -> None:
    """
    Analyzes a video file to detect slide transitions based on structural similarity.

    The function processes the video frame by frame, comparing each frame to the previous one.
    A slide change is identified when the similarity score between consecutive frames falls
    below a predefined threshold.

    Args:
        video_file_path: Path to the video file that will be analyzed for slide transitions.
    Returns:
        None (prints detected slide change timestamps to the console).
    """

    first_slide_result: tuple[np.ndarray, int] | None = find_first_slide(video_file_path)
    if first_slide_result is None:
        print("Error: No slide was found in the video.")
        return

    first_slide_frame: np.ndarray = first_slide_result[0]
    first_slide_frame_index: int = first_slide_result[1]

    roi_coordinates: tuple[int, int, int, int] | None = extract_slide_roi_coordinates_from_image(
        first_slide_frame
    )
    if roi_coordinates is None:
        print("Error: Unable to extract ROI coordinates from the first slide.")
        return

    roi_top_left_x, roi_top_left_y, roi_bottom_right_x, roi_bottom_right_y = roi_coordinates

    cap: cv2.VideoCapture = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    fps: float = cap.get(cv2.CAP_PROP_FPS)
    current_frame_index: float = first_slide_frame_index

    print("Starting video analysis")
    while cap.isOpened():

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        success: bool
        video_frame: np.ndarray
        success, video_frame = cap.read()

        if not success:
            break

        video_frame_adjusted_roi: np.ndarray = video_frame[
            roi_top_left_y:roi_bottom_right_y, roi_top_left_x:roi_bottom_right_x
        ]

        if DEBUG_MODE:
            # Show video
            cv2.imshow("ROI Video", video_frame_adjusted_roi)

            # quit with 'q'
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        current_frame_index += fps
    cap.release()


if __name__ == "__main__":
    detect_slide_transitions("tests/test_data/videos/dbwt1/dbwt1_02.mp4")
