import cv2
import numpy as np
import pyautogui
from skimage.metrics import structural_similarity as ssim
import pytesseract
import re

DEBUG_MODE = False

# Threshold for detecting slide changes
STRUCTURAL_SIMILARITY_THRESHOLD = 0.85


def show_image_resized(image, window_name="default", scale_factor=0.8):
    """
    DEBUG ONLY: DON'T USE IN PRODUCTIVE CODE.
    Displays an image in an OpenCV window, resizes it to fit the screen while keeping the title visible.

    Args:
        image (np.ndarray): Image to be displayed.
        window_name (str): Name of the OpenCV window.
        scale_factor (float): Scaling factor to reduce the image size (default: 0.9 for 90% of the screen).
    """
    # Get the screen resolution
    screen_width, screen_height = pyautogui.size()
    image_height, image_width = image.shape[:2]

    # Calculate the scaling factor to ensure the window fits within the screen
    scale_factor = min((screen_width * scale_factor) / image_width, (screen_height * scale_factor) / image_height)
    resized_image = cv2.resize(image, (int(image_width * scale_factor), int(image_height * scale_factor)))

    # Create the window and set its size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(image_width * scale_factor), int(image_height * scale_factor))

    # Center the window on the screen
    x_pos = (screen_width - int(image_width * scale_factor)) // 2
    y_pos = (screen_height - int(image_height * scale_factor)) // 2
    cv2.moveWindow(window_name, x_pos, y_pos)

    # Display the image
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_all_keywords_in_image(frame, keywords_to_be_matched=None, confidence_threshold=80):
    """
    Checks whether all specified keywords appear in the image using OCR.

    Args:
        frame: OpenCV image (BGR).
        keywords_to_be_matched: List of expected words (case-insensitive).
        confidence_threshold: Minimum OCR confidence level (0-100).

    Returns:
        bool: True if all keywords are found, False otherwise.
    """
    if keywords_to_be_matched is None:
        keywords_to_be_matched = ["UNIVERSITY", "FH", "AACHEN", "OF", "APPLIED", "SCIENCES"]

    # Convert image for OCR processing
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        frame_RGB,
        output_type=pytesseract.Output.DICT,
        config="--psm 11 --oem 3"  # psm11 because we don't care about the order of the text
    )

    found_keywords = {}  # Stores detected words

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if conf < confidence_threshold:
            continue

        # Check if detected text matches any expected keywords
        for keyword in keywords_to_be_matched:
            # Has to be a standalone word and case-insensitive
            if keyword not in found_keywords and re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                found_keywords[keyword] = (x, y, x + w, y + h)  # (x1, y1, x2, y2)

    # Ensure all keywords are found before returning True
    if set(keywords_to_be_matched).issubset(set(found_keywords.keys())):

        if DEBUG_MODE:

            # Draw bounding boxes for all found keywords
            for keyword, (x1, y1, x2, y2) in found_keywords.items():
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(frame, keyword, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            show_image_resized(frame)

        return True

    return False


def find_first_slide(video_path, max_seconds=30):
    """
    Searches the video for a slide shown in the video by utilizing check_all_keywords_in_image() for every frame
    until the slide which contains the keywords shows up

    Args:
        video_path: Path to the video file.
        max_seconds: Maximum duration (in seconds) to scan the video for a slide.

    Returns:
        Optional[np.ndarray]: The frame containing the text if found, otherwise None.
    """
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second (FPS) from video
    max_attempts = max_seconds * fps  # Calculate max frames to check

    for _ in range(max_attempts):

        ret, frame = cap.read()
        if not ret:
            break

        found = check_all_keywords_in_image(frame)
        if found:
            cap.release()
            return frame  # Return the frame where keywords were detected

    cap.release()
    return None


def add_black_border(image, padding_size=50):
    """
        Add a black border to an image with given padding size.

        Args:
            image (np.ndarray): Input image (BGR format).
            padding_size (int): Border size in pixels (default: 50).

        Returns:
            np.ndarray: Image with black borders.
    """
    return cv2.copyMakeBorder(
        image,
        top=padding_size,
        bottom=padding_size,
        left=padding_size,
        right=padding_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )


def extract_roi_from_image_for_slide(image_path: str, min_width: int = 100, min_height: int = 100,
                                     aspect_ratio_range: tuple = (1.3, 2.0)):
    """
    Loads an image, preprocesses it (grayscale, edge detection, contour extraction),
    and finds the largest rectangular Region of Interest (RoI).

    Args:
        image_path (str): Path to the input image file.
        min_width (int): Minimum width of a valid RoI.
        min_height (int): Minimum height of a valid RoI.
        aspect_ratio_range (tuple): Acceptable aspect ratio range (width/height).

    Returns:
        np.ndarray: Extracted RoI image or None if no valid region is found.
    """

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if frame is None:
        print(f" Error: Image '{image_path}' couldn't be loaded.")

    # Add a black border to ensure edge detection works, when the image only contains the slide.
    padded_image = add_black_border(frame)

    gray_scale_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
    canny_edges = cv2.Canny(gray_scale_image, 50, 150)

    # Using cv2.RETR_EXTERNAL to detect only the outer contours,
    # inner contours are not relevant for finding the biggest rectangle (the slide).
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if DEBUG_MODE:
        visualization_image = padded_image.copy()

    # Find the largest rectangle (the slide)
    biggest_rectangle = None
    max_area = 0

    for cnt in contours:
        top_left_x, top_left_y, width, height = cv2.boundingRect(cnt)
        area = width * height

        if DEBUG_MODE:
            cv2.rectangle(visualization_image, (top_left_x, top_left_y),
                          (top_left_x + width, top_left_y + height), (0, 0, 255), 2)

        if area > max_area and width > min_width and height > min_height and aspect_ratio_range[0] < width / height < \
                aspect_ratio_range[1]:
            biggest_rectangle = (top_left_x, top_left_y, width, height)
            max_area = area

    # Extract RoI if a valid region is found
    if biggest_rectangle:
        top_left_x, top_left_y, width, height = biggest_rectangle
        roi = padded_image[top_left_y:top_left_y + height, top_left_x:top_left_x + width]

        if DEBUG_MODE:
            # Draw contours on a copy of the original image for visualization
            contour_visualization = padded_image.copy()
            cv2.drawContours(contour_visualization, contours, -1, (0, 0, 255), 2)

            show_image_resized(frame)
            show_image_resized(padded_image)
            show_image_resized(gray_scale_image)
            show_image_resized(canny_edges)
            show_image_resized(contour_visualization)
            show_image_resized(visualization_image)
            show_image_resized(roi)

        return roi
    else:
        print("No valid slide region found.")
        return None


def compute_image_similarity(image1, image2):
    """
        Computes the Structural Similarity Index (SSIM) between two images.

        The function resizes both images, converts them to grayscale, and calculates SSIM
        to determine the similarity between them.

        Args:
            image1 (np.ndarray): First image.
            image2 (np.ndarray): Second image.

        Returns:
            float: Similarity score (range: -1 to 1), where 1 means identical images.

        """
    # Resize images to a smaller size for faster computation
    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))

    # Convert images to grayscale for similarity calculation
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the structural similarity index (SSIM)
    similarity_score, _ = ssim(grayscale_image1, grayscale_image2, full=True)

    return similarity_score


def detect_slide_transitions(video_file_path):
    """
    Analyzes a video file to detect slide transitions based on structural similarity.

    The function processes the video frame by frame, comparing each frame to the previous one.
    A slide change is identified when the similarity score between consecutive frames falls
    below a predefined threshold.

    Args:
        video_file_path (str): Path to the video file that will be analyzed for slide transitions.
    Returns:
        None (prints detected slide change timestamps to the console).
    """
    video_capture = cv2.VideoCapture(video_file_path)
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    print(frames_per_second)

    current_frame_index = 0
    previous_video_frame = None

    print("Starting video analysis")
    while video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        frame_read_successful, current_video_frame = video_capture.read()
        if not frame_read_successful:
            break

        # For the first frame of the video
        if previous_video_frame is None:
            previous_video_frame = current_video_frame

        if previous_video_frame is not None:
            similarity_value = compute_image_similarity(previous_video_frame, current_video_frame)

            if similarity_value < STRUCTURAL_SIMILARITY_THRESHOLD:
                timestamp_seconds = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                print(f"Slide changed at {minutes:02d}:{seconds:02d}")

        previous_video_frame = current_video_frame

        current_frame_index += frames_per_second

    video_capture.release()


if __name__ == "__main__":
    detect_slide_transitions("tests/test_data/videos/01_DBWT2.mp4")