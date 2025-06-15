from typing import Generator

import cv2
import numpy as np


def generate_video_frames_with_number(
    video_capture: cv2.VideoCapture, frames_step: int = 1, start_frame_number: int = 0
) -> Generator[tuple[np.ndarray, int], None, None]:
    """
    Generates video frames along with their frame numbers, starting from a given frame
    and optionally skipping frames at a fixed interval.

    Args:
        video_capture: An open cv2.VideoCapture object.
        frames_step: Number of frames to skip between each yield.
        start_frame_number: The initial frame number to begin reading from.

    Yields:
        A tuple containing:
        - frame: The actual video frame as a NumPy array.
        - frame_number: The number where the frame appeared in the video.
    """
    video_frame_number = start_frame_number

    while video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, video_frame_number)
        success, video_frame = video_capture.read()
        if not success:
            break

        yield video_frame, video_frame_number
        video_frame_number += frames_step


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap


def read_video_frame(video_path: str, frame_number: int) -> np.ndarray:
    """
    Returns the frame at the given index from a video without changing the state of any external capture object.

    Args:
        video_path: Path to the video file.
        frame_number: Index of the desired frame.

    Returns:
        The requested frame as a NumPy array, or None if reading fails.
    """
    video_capture: cv2.VideoCapture = open_video_capture(video_path)

    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success: bool
    video_frame: np.ndarray
    success, video_frame = video_capture.read()
    video_capture.release()
    return video_frame
