from typing import Generator
from video_frame import VideoFrame

import cv2


def generate_video_frame(
    video_capture: cv2.VideoCapture,
    frames_step: int = 1,
    start_frame_number: int = 0,
    roi_coordinates: tuple[int, int, int, int] | None = None,
) -> Generator[VideoFrame, None, None]:
    """
    Generates video frames along with their frame numbers, starting from a given frame
    and optionally skipping frames at a fixed interval.

    Args:
        video_capture: An open cv2.VideoCapture object.
        frames_step: Number of frames to skip between each yield.
        start_frame_number: The initial frame number to begin reading from.
        roi_coordinates: Precomputed ROI coordinates to use for all frames.
                         If provided, the VideoFrame will use this instead of computing its own.

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

        # Optional shared RoI to avoid re-running compute_roi_coordinates in VideoFrame
        yield VideoFrame(video_frame, video_frame_number, roi_coordinates=roi_coordinates)
        video_frame_number += frames_step


def open_video_capture(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    return cap
