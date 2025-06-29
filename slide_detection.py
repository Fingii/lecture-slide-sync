import json
import cv2

from video_utils import open_video_capture, generate_video_frame
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame
from lecture_slides import LectureSlides
from slide_matcher import SlideTracker
from debug_utils import show_image_resized


def detect_first_slide(
    video_file_path: str, keywords_to_be_matched: set[str], max_seconds: int = 30
) -> VideoFrame:
    """
    Scans the beginning of a lecture video to find the first visible slide
    based on the presence of predefined OCR keywords.

    The function reads video frames sequentially and performs OCR on each one.
    It stops when all expected keywords are detected in a frame or when the
    maximum scan duration is exceeded.

    Args:
        video_file_path: Path to the input video file.
        keywords_to_be_matched: Set of required OCR keywords that must all appear
                                for a frame to be considered a valid slide.
        max_seconds: Maximum number of seconds to scan for the first slide (default is 30).

    Returns:
        The first VideoFrame containing all required keywords.

    Raises:
        RuntimeError: If no matching slide is found within the specified time window.
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


def is_slide_change_detected(video_frame: VideoFrame, slide_tracker: SlideTracker) -> bool:
    """
    Determines whether the current video frame represents a new forward slide transition.

    This function uses perceptual hashing to compare the current frame's slide content
    against known slide hashes. It only returns True if:
    - A valid match is found, and
    - The matched slide index is greater than the current index (ignores backward slides).

    Args:
        video_frame: The current VideoFrame to evaluate if it shows a new slide.
        slide_tracker: An instance of SlideTracker managing slide state and matching logic.

    Returns:
        True if a new forward slide is detected, False otherwise.
    """
    most_similar_slide_index: int | None = slide_tracker.find_most_similar_slide_index(video_frame)
    if most_similar_slide_index is None:
        return False

    if most_similar_slide_index > slide_tracker.current_slide_index:
        slide_tracker.update_slide_index(most_similar_slide_index)
        return True
    return False


def detect_slide_transitions(
    video_file_path: str,
    pdf_file_path: str,
    keywords_to_be_matched: set[str],
) -> None:
    """
    Detects slide transitions in a lecture video by matching video frame content to slides from a given PDF.

    The function begins by identifying the first frame in the video that contains a valid slide
    using OCR keyword matching. It then precomputes the region of interest (RoI) from that frame
    and uses it consistently for the rest of the video to improve performance. Each subsequent frame
    is hashed and compared to the PDF slide hashes. A new slide is only counted as a transition
    if it is different and comes after the last matched slide (to avoid detecting backward navigation).

    Args:
        video_file_path: Path to the lecture video file.
        pdf_file_path: Path to the PDF file containing lecture slides.
        keywords_to_be_matched: A set of required OCR keywords used to detect the first valid slide.

    Returns:
        None. Saves a dictionary of detected slide transitions as JSON.
    """

    first_slide_video_frame = detect_first_slide(video_file_path, keywords_to_be_matched)
    cap: cv2.VideoCapture = open_video_capture(video_file_path)

    # RoI precomputed since it remains constant from now on
    precomputed_roi = first_slide_video_frame.compute_roi_coordinates()

    lecture_slides: LectureSlides = LectureSlides(pdf_file_path)
    slide_tracker: SlideTracker = SlideTracker(lecture_slides)

    slide_changes: dict[int, int] = {}  # slide_index: frame_number
    print("Starting video analysis")
    for video_frame in generate_video_frame(
        cap,
        frames_step=100,
        start_frame_number=first_slide_video_frame.frame_number,
        roi_coordinates=precomputed_roi,
    ):
        if is_slide_change_detected(video_frame, slide_tracker):
            slide_changes[slide_tracker.current_slide_index] = video_frame.frame_number

    cap.release()

    with open("slide_changes.json", "w", encoding="utf-8") as f:
        json.dump(slide_changes, f, indent=4)
