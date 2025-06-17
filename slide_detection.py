import cv2

from video_utils import open_video_capture, generate_video_frame
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame
from lecture_slides import LectureSlides
from slide_matcher import SlideTracker
from debug_utils import show_image_resized


def detect_first_slide(video_file_path: str, max_seconds: int = 30) -> VideoFrame:
    """
    Scans the beginning of a lecture video to find the first visible slide
    based on the presence of predefined OCR keywords.

    The function reads frames sequentially and performs OCR on each one.
    It stops when all expected keywords are detected or when the maximum scan duration is exceeded.

     Args:
        video_file_path: Path to the input video file.
        max_seconds: Maximum number of seconds to scan for the first slide (default is 30).

    Returns:
        The video frame of the first slide

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


def detect_slide_transitions(video_file_path: str, pdf_file_path: str) -> list[int]:
    """
    Detects slide transitions in a lecture video by matching video frame content to slides from a given PDF.

    The function begins by identifying the first frame in the video that contains a valid slide.
    It then precomputes the region of interest (RoI) from that frame and uses it consistently
    for the rest of the video to improve performance. Each subsequent frame is hashed and compared
    to the PDF slide hashes. A new slide is only counted as a transition if it is different and comes
    after the last matched slide (e.g., to avoid backtracking to old slides).

    Args:
        video_file_path: Path to the lecture video file.
        pdf_file_path: Path to the PDF file containing lecture slides.

    Returns:
        A list of frame numbers where valid slide transitions were detected.
    """

    first_slide_video_frame: VideoFrame = detect_first_slide(video_file_path)
    cap: cv2.VideoCapture = open_video_capture(video_file_path)
    precomputed_roi = first_slide_video_frame.compute_roi_coordinates()

    lecture_slides: LectureSlides = LectureSlides(pdf_file_path)
    slide_tracker: SlideTracker = SlideTracker(lecture_slides)

    slide_change_frame_number: list[int] = []

    print("Starting video analysis")
    for video_frame in generate_video_frame(
        cap,
        frames_step=100,
        start_frame_number=first_slide_video_frame.frame_number,
        roi_coordinates=precomputed_roi,
    ):
        if is_slide_change_detected(video_frame, slide_tracker):
            show_image_resized(video_frame.roi_frame)
            slide_change_frame_number.append(video_frame.frame_number)

    cap.release()
    return slide_change_frame_number
