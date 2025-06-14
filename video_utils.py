import cv2
import numpy as np


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap


def read_frame_at_index(video_path: str, frame_index: int) -> np.ndarray:
    """
    Returns the frame at the given index from a video without changing the state of any external capture object.

    Args:
        video_path: Path to the video file.
        frame_index: Index of the desired frame.

    Returns:
        The requested frame as a NumPy array, or None if reading fails.
    """
    video_capture: cv2.VideoCapture = open_video_capture(video_path)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success: bool
    frame: np.ndarray
    success, frame = video_capture.read()
    video_capture.release()
    return frame
