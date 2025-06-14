from typing import Generator

import cv2
import numpy as np


def generate_indexed_video_frames(
    video_capture: cv2.VideoCapture, frames_step: int = 1, start_index: int = 0
) -> Generator[tuple[int, np.ndarray], None, None]:
    """
    Generator that yields frames from a video with customizable starting point and skipping.

    Args:
        video_capture: An open cv2.VideoCapture object.
        frames_step: Number of frames to skip between reads.
        start_index: Frame index to start reading from.

    Yields:
        A tuple (frame_index, frame) for each read frame.
    """
    frame_index = start_index

    while video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video_capture.read()
        if not success:
            break

        yield frame_index, frame
        frame_index += frames_step


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
