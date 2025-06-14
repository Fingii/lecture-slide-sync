import cv2
import numpy as np

from config import DEBUG_MODE
from ocr_keyword_detector import are_all_keywords_present
from roi_utils import extract_slide_roi_coordinates


def detect_first_slide(video_path: str, max_seconds: int = 30) -> tuple[np.ndarray, int] | None:
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

    first_slide_result: tuple[np.ndarray, int] | None = detect_first_slide(video_file_path)
    if first_slide_result is None:
        print("Error: No slide was found in the video.")
        return

    first_slide_frame: np.ndarray = first_slide_result[0]
    first_slide_frame_index: int = first_slide_result[1]

    roi_coordinates: tuple[int, int, int, int] | None = extract_slide_roi_coordinates(first_slide_frame)
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
