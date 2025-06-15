import cv2
import numpy as np

from image_utils import apply_roi_crop
from video_utils import open_video_capture, generate_video_frames_with_number
from ocr_keyword_detector import are_all_keywords_present
from roi_utils import extract_slide_roi_coordinates
from video_utils import open_video_capture, generate_indexed_video_frames


def detect_first_slide(video_file_path: str, max_seconds: int = 30) -> tuple[np.ndarray, int] | None:
    """
    Searches the video for a slide containing specific keywords by checking frames
    until either the slide is found or the maximum duration is reached.

    Args:
        video_file_path: Path to the video file.
        max_seconds: Maximum duration (in seconds) to scan the video for a slide.

    Returns:
        A tuple containing:
        - The frame containing the text if found
        - The frame count where it was found
        If no slide is found, returns None.
    """
    cap: cv2.VideoCapture = open_video_capture(video_file_path)

    fps: float = float(cap.get(cv2.CAP_PROP_FPS))
    max_attempts: int = int(max_seconds * fps)
    keywords_to_be_matched: set[str] = {"UNIVERSITY", "FH", "AACHEN", "OF", "APPLIED", "SCIENCES"}

    for frame, frame_number in generate_video_frames_with_number(cap, frames_step=1):
        if frame_number >= max_attempts:
            break
        if are_all_keywords_present(frame, keywords_to_be_matched):
            cap.release()
            return frame, frame_number

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
    first_slide_frame_number: int = first_slide_result[1]

    roi_coordinates: tuple[int, int, int, int] | None = extract_slide_roi_coordinates(first_slide_frame)
    if roi_coordinates is None:
        print("Error: Unable to extract ROI coordinates from the first slide.")
        return

    current_pdf_slide_page = 0
    cap: cv2.VideoCapture = open_video_capture(video_file_path)

    print("Starting video analysis")
    for video_frame, _ in generate_video_frames_with_number(cap, start_frame_number=first_slide_frame_number):
        video_frame_roi = apply_roi_crop(video_frame, roi_coordinates)

    cap.release()
