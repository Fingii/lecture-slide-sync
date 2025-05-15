import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

VIDEO_FILE_PATH = "testVideos/01_DBWT2.mp4"

# Threshold for detecting slide changes
STRUCTURAL_SIMILARITY_THRESHOLD = 0.85


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

    # Find the largest rectangle (the slide)
    biggest_rectangle = None
    max_area = 0

    for cnt in contours:
        top_left_x, top_left_y, width, height = cv2.boundingRect(cnt)
        area = width * height

        if area > max_area and width > min_width and height > min_height and aspect_ratio_range[0] < width / height < \
                aspect_ratio_range[1]:
            biggest_rectangle = (top_left_x, top_left_y, width, height)
            max_area = area

    # Extract RoI if a valid region is found
    if biggest_rectangle:
        top_left_x, top_left_y, width, height = biggest_rectangle
        roi = frame[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
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


def detect_slide_transitions():
    """
    Analyzes a video file to detect slide transitions based on structural similarity.

    The function processes the video frame by frame, comparing each frame to the previous one.
    A slide change is identified when the similarity score between consecutive frames falls
    below a predefined threshold.

    Args:
        None (uses global VIDEO_FILE_PATH to access the video).

    Returns:
        None (prints detected slide change timestamps to the console).
    """
    video_capture = cv2.VideoCapture(VIDEO_FILE_PATH)
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
    extract_roi_from_image_for_slide("test_Images/only_slide_dbwt2.png")
