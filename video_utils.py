from typing import Generator
from video_frame import VideoFrame

import numpy as np
import av


def generate_video_frame(
    video_path: str,
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
        video_path: Path to the input video file.
        frames_step: Number of frames to skip between each yield.
        start_frame_number: The initial frame number to begin reading from.
        roi_coordinates: Precomputed ROI coordinates to use for all frames.
                         If provided, the VideoFrame will use this instead of computing its own.

    Yields:
        VideoFrame objects containing the frame image and its metadata.
    """
    with av.open(video_path) as container:
        video_stream: av.video.stream.VideoStream = container.streams.video[0]

        if video_stream.average_rate is None:
            raise ValueError("video_stream.average_rate is None")
        if video_stream.time_base is None:
            raise ValueError("video_stream.time_base is None")

        # Convert frame index to pts
        start_frame_second: float = start_frame_number / float(video_stream.average_rate)
        start_frame_pts: int = int(start_frame_second / float(video_stream.time_base))

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
                roi_coordinates=roi_coordinates,
            )
