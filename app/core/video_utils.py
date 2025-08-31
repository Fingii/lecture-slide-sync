from typing import Generator
from app.models.video_frame import VideoFrame
from logs.logging_config import logger

from fractions import Fraction
from pathlib import Path

import numpy as np
import av


def generate_video_frame(
    video_file_path: Path,
    frames_step: int = 1,
    start_frame_number: int = 0,
    roi_coordinates: tuple[int, int, int, int] | None = None,
) -> Generator[VideoFrame, None, None]:
    """
    Generates video frames using PyAV with accurate frame indices.

    This function seeks to the nearest keyframe before the desired frame,
    then decodes forward and yields every Nth frame starting from the
    specified frame index.

    Args:
        video_file_path: Path to the input video file.
        frames_step: Number of frames to skip between each yield.
        start_frame_number: The initial frame number to begin reading from.
        roi_coordinates: Precomputed ROI coordinates to use for all frames.
                         If provided, the VideoFrame will use this instead of computing its own.

    Yields:
        VideoFrame objects containing the frame image and its metadata.
    """
    logger.debug(
        "Generating frames from %s (start: %d, step: %d)",
        video_file_path.name,
        start_frame_number,
        frames_step,
    )
    with av.open(video_file_path) as container:
        video_stream: av.video.stream.VideoStream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        if video_stream.average_rate is None:
            logger.error("Missing average_rate in video stream")
            raise ValueError("video_stream.average_rate is None")
        if video_stream.time_base is None:
            logger.error("Missing time_base in video stream")
            raise ValueError("video_stream.time_base is None")

        # Convert frame index to pts
        start_frame_second: float = start_frame_number / float(video_stream.average_rate)
        start_frame_pts: int = int(start_frame_second / float(video_stream.time_base))

        logger.debug(
            "Seeking to frame %d (%.2fs, pts=%d)", start_frame_number, start_frame_second, start_frame_pts
        )
        container.seek(
            offset=start_frame_pts,
            backward=True,
            any_frame=False,
            stream=video_stream,
        )

        frame_counter = 0

        for decoded_frame in container.decode(video_stream):
            if decoded_frame.pts is None:
                continue

            if frame_counter % frames_step != 0:
                frame_counter += 1
                continue

            current_frame_second = float(decoded_frame.pts * video_stream.time_base)
            current_frame_number = round(current_frame_second * float(video_stream.average_rate))

            if current_frame_number < start_frame_number:
                continue

            frame_counter += 1

            decoded_frame_ndarray: np.ndarray = decoded_frame.to_ndarray(format="bgr24")
            yield VideoFrame(
                full_frame=decoded_frame_ndarray,
                frame_number=current_frame_number,
                frame_timestamp_seconds=current_frame_second,
                roi_coordinates=roi_coordinates,
            )


def get_video_fps(video_file_path: Path) -> float:  # type: ignore
    """
    Retrieves the average frames per second (FPS) of a video file.

    Args:
        video_file_path: Path to the input video file.

    Returns:
        The average FPS as a float.

    Raises:
        ValueError: If the video stream does not contain average_rate metadata.
    """
    logger.debug("Getting FPS from video: %s", video_file_path.name)
    with av.open(video_file_path) as container:
        video_stream: av.video.stream.VideoStream = container.streams.video[0]
        average_rate: Fraction | None = video_stream.average_rate

        if average_rate is None:
            logger.error("Missing average_rate in video: %s", video_file_path.name)
            raise ValueError("Video stream has no average_rate metadata.")
        fps = float(average_rate)
        logger.debug("Detected FPS: %.2f", fps)
        return fps
