import cv2
import numpy as np
import pyautogui
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pytesseract  # type: ignore
import re
from typing import Union
import pymupdf  # type: ignore

DEBUG_MODE = False

# Threshold for detecting slide changes
STRUCTURAL_SIMILARITY_THRESHOLD = 0.85


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


def show_image_resized(image: np.ndarray, window_name: str = "default", scale_factor: float = 0.8) -> None:
    """
    DEBUG ONLY: DON'T USE IN PRODUCTIVE CODE.
    Displays an image in an OpenCV window, resizes it to fit the screen while keeping the title visible.

    Args:
        image: Image to be displayed.
        window_name: Name of the OpenCV window.
        scale_factor: Scaling factor to reduce the image size (default: 0.9 for 90% of the screen).
    Returns:
        None, displays the given image centered and resized
    """
    screen_height: int = pyautogui.size().height
    screen_width: int = pyautogui.size().width
    image_height: int = image.shape[0]
    image_width: int = image.shape[1]

    scale_factor = min(
        (screen_width * scale_factor) / image_width,
        (screen_height * scale_factor) / image_height,
    )  # Calculate optimal scale factor to fit screen

    new_width: int = max(1, int(image_width * scale_factor))
    new_height: int = max(1, int(image_height * scale_factor))

    resized_image: np.ndarray = cv2.resize(image, (new_width, new_height))  # Resize image

    # Create and resize window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)

    # Center window on screen
    x_pos: int = (screen_width - new_width) // 2
    y_pos: int = (screen_height - new_height) // 2
    cv2.moveWindow(window_name, x_pos, y_pos)

    # Display image
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_all_keywords_in_image(
    frame: np.ndarray,
    keywords_to_be_matched: set[str] | None = None,
    confidence_threshold: int = 80,
) -> bool:
    """
    Checks whether all specified keywords appear in the image using OCR.

    Args:
        frame: OpenCV image (BGR).
        keywords_to_be_matched: Set of expected words (case-insensitive). Set is used because the order doesn't matter. If None, uses default keywords.
        confidence_threshold: Minimum OCR confidence level (0-100).

    Returns:
        True if all keywords are found, False otherwise.
    """
    if keywords_to_be_matched is None:
        keywords_to_be_matched = {
            "UNIVERSITY",
            "FH",
            "AACHEN",
            "OF",
            "APPLIED",
            "SCIENCES",
        }

    # Convert image for OCR processing
    frame_RGB: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data: dict[str, list[Union[str, int]]] = pytesseract.image_to_data(
        frame_RGB,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3",  # psm11 because we don't care about the order of the text
    )

    if DEBUG_MODE:
        bounding_boxes: dict[str, tuple[int, int, int, int]] = {}  # for visualization

    found_keywords: set[str] = set()

    for i in range(len(data["text"])):
        text: str = str(data["text"][i]).strip()
        conf: int = int(data["conf"][i])

        if conf < confidence_threshold:
            continue

        # Check if detected text matches any expected keywords
        for keyword in keywords_to_be_matched:
            # Has to be a standalone word and case-insensitive
            if keyword not in found_keywords and re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                found_keywords.add(keyword)

                if DEBUG_MODE:
                    textbox_leftmost_x_position: int = int(data["left"][i])
                    textbox_topmost_y_position: int = int(data["top"][i])
                    text_box_width: int = int(data["width"][i])
                    text_box_height: int = int(data["height"][i])

                    # Extracting position of textbox (x1, y1, x2, y2)
                    bounding_boxes[keyword] = (
                        textbox_leftmost_x_position,
                        textbox_topmost_y_position,
                        textbox_leftmost_x_position + text_box_width,
                        textbox_topmost_y_position + text_box_height,
                    )

    # Early exit if not all keywords were found
    if not keywords_to_be_matched.issubset(found_keywords):
        return False

    if DEBUG_MODE:
        for keyword in found_keywords:
            if keyword in bounding_boxes:
                x1, y1, x2, y2 = bounding_boxes[keyword]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(
                    frame,
                    keyword,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        show_image_resized(frame)

    return True


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

    fps: float = float(cap.get(cv2.CAP_PROP_FPS))  # Frames per second (FPS) from video
    max_attempts: int = int(max_seconds * fps)  # Calculate max frames to check

    for frame_count in range(max_attempts):
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()

        if not ret:
            break

        if check_all_keywords_in_image(frame):
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
) -> list[int] | None:
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
        return [
            max(0, largest_contour_top_left_x - padding),
            max(0, largest_contour_top_left_y - padding),
            min(image_width, largest_contour_bottom_right_x - padding),
            min(image_height, largest_contour_bottom_right_y - padding),
        ]
    else:
        print("Detected RoI is too small to be valid.")
        return None


def compute_image_similarity(
    image1: np.ndarray,
    image2: np.ndarray,
) -> float:
    """
    Computes the Structural Similarity Index (SSIM) between two images.

    The function resizes both images, converts them to grayscale, and calculates SSIM
    to determine the similarity between them.

    Args:
        image1: First image.
        image2: Second image.

    Returns:
        Similarity score (range: -1 to 1), where 1 means identical images.

    """
    # Convert images to grayscale for similarity calculation
    grayscale_image1: np.ndarray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2: np.ndarray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the structural similarity index (SSIM)
    similarity_score: float
    similarity_score, _ = ssim(grayscale_image1, grayscale_image2, full=True)

    return similarity_score


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
    first_slide_start_frame_index: int = first_slide_result[1]

    roi_values: list[int] | None = extract_slide_roi_coordinates_from_image(first_slide_frame)

    if roi_values is None:
        print("Error: Unable to extract ROI coordinates from the first slide.")
        return

    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int

    top_left_x, top_left_y, bottom_right_x, bottom_right_y = roi_values

    video_capture: cv2.VideoCapture = cv2.VideoCapture(video_file_path)
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    fps: float = video_capture.get(cv2.CAP_PROP_FPS)
    current_frame_index: float = first_slide_start_frame_index
    previous_video_frame: np.ndarray | None = None

    print("Starting video analysis")
    while video_capture.isOpened():

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        frame_read_successful: bool
        current_video_frame: np.ndarray
        frame_read_successful, current_video_frame = video_capture.read()
        current_video_frame_with_adjusted_roi: np.ndarray = current_video_frame[
            top_left_y:bottom_right_y, top_left_x:bottom_right_x
        ]

        if DEBUG_MODE:
            # Show video
            cv2.imshow("ROI Video", current_video_frame_with_adjusted_roi)

            # quit with 'q'
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        if not frame_read_successful:
            break

        # For the first frame of the video
        if previous_video_frame is None:
            previous_video_frame = current_video_frame
            current_frame_index += int(fps)
            continue

        similarity_value: float = compute_image_similarity(previous_video_frame, current_video_frame)

        if similarity_value < STRUCTURAL_SIMILARITY_THRESHOLD:
            timestamp_seconds: float = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            minutes: int = int(timestamp_seconds // 60)
            seconds: int = int(timestamp_seconds % 60)
            print(f"Slide changed at {minutes:02d}:{seconds:02d}")

        previous_video_frame = current_video_frame

        current_frame_index += fps
    video_capture.release()


if __name__ == "__main__":
    detect_slide_transitions("tests/test_data/videos/dbwt1/dbwt1_02.mp4")
    # print("Hi")
