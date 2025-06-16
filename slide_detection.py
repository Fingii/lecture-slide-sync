import cv2

from video_utils import open_video_capture, generate_video_frame
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame


def detect_first_slide(video_file_path: str, max_seconds: int = 30) -> VideoFrame:
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

    for video_frame in generate_video_frame(cap, frames_step=1):
        if video_frame.frame_number >= max_attempts:
            break
        if are_all_keywords_present(video_frame, keywords_to_be_matched):
            cap.release()
            return video_frame

    cap.release()
    raise RuntimeError(f"No slide detected within the first {max_seconds} seconds of the video.")


def detect_slide_transitions(video_file_path: str) -> None:
    """
    Analyzes a video file to detect slide transitions.

    Args:
        video_file_path: Path to the video file that will be analyzed for slide transitions.
    Returns:
        None (prints detected slide change timestamps to the console).
    """

    first_slide_video_frame: VideoFrame = detect_first_slide(video_file_path)
    cap: cv2.VideoCapture = open_video_capture(video_file_path)
    precomputed_roi = first_slide_video_frame.compute_roi_coordinates()

    print("Starting video analysis")
    for video_frame in generate_video_frame(
        cap,
        frames_step=100,
        start_frame_number=first_slide_video_frame.frame_number,
        roi_coordinates=precomputed_roi,
    ):
        cv2.imshow("RoI Frame", video_frame.roi_frame)
        print(video_frame.frame_number)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
