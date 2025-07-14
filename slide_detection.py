import re
import av

from video_utils import generate_video_frame
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame
from lecture_slides import LectureSlides
from slide_tracker import SlideTracker

from fractions import Fraction
from thefuzz import fuzz  # type: ignore


def normalize_text(
    raw_text: str,
    keywords_to_ignore: set[str] | None = None,
    lowercase: bool = True,
    min_length: int = 2,
) -> str:
    """
    Normalizes raw text by optionally removing short words and ignored keywords.

    Args:
        raw_text: The input string to normalize.
        keywords_to_ignore: Optional set of keywords to exclude (case-insensitive).
        lowercase: Whether to convert text to lowercase.
        min_length: Minimum word length to retain.

    Returns:
        A cleaned string ready for text similarity comparison.
    """
    if lowercase:
        raw_text = raw_text.lower()
        keywords_to_ignore = {kw.lower() for kw in (keywords_to_ignore or set())}
    else:
        keywords_to_ignore = keywords_to_ignore or set()

    words: list[str] = re.findall(r"\S+", raw_text)
    cleaned_words: list[str] = [
        word for word in words if word not in keywords_to_ignore and len(word) >= min_length
    ]
    return " ".join(cleaned_words)


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
    with av.open(video_file_path) as container:
        video_stream = container.streams.video[0]
        avg_rate: Fraction | None = video_stream.average_rate

        if avg_rate is None:
            raise ValueError("Video stream has no average_rate metadata.")

        fps: float = float(avg_rate)
        max_attempts: int = int(max_seconds * fps)

    for video_frame in generate_video_frame(
        video_path=video_file_path,
        frames_step=1,
    ):
        if video_frame.frame_number >= max_attempts:
            break
        if are_all_keywords_present(video_frame, keywords_to_be_matched):
            return video_frame

    raise RuntimeError(f"No slide detected within the first {max_seconds} seconds of the video.")


def is_slide_change_detected(
    video_frame: VideoFrame, slide_tracker: SlideTracker, keywords_to_ignore: set[str]
) -> bool:
    """
    Determines whether the current video frame represents a new forward slide transition.

    The function first attempts to match the video frame to one of the PDF slides using perceptual hashing.
    If the match is highly confident (Hamming distance < 2), the slide is accepted immediately.
    Otherwise, OCR is used to extract high-confidence text from the video frame, and the slide's PDF text is also extracted.
    Both texts are normalized (e.g., filtered, lowercased, keywords removed) and compared using fuzzy token set similarity.

    Args:
        video_frame: The current VideoFrame to evaluate.
        slide_tracker: An instance of SlideTracker managing slide state and hash matching.
        keywords_to_ignore: A set of recurring words to exclude during text comparison (e.g., university names, headers).

    Returns:
        True if a valid and unseen new slide is detected and confirmed; False otherwise.
    """
    match: tuple[int, float] | None = slide_tracker.find_most_similar_slide_index(video_frame)
    if match is None:
        return False
    most_similar_slide_index, most_similar_slide_index_hamming_distance = match

    if slide_tracker.has_seen_slide(most_similar_slide_index):
        return False

    # Definite match — no OCR needed, helpful for image slides, where PDF text is not extractable from images
    if most_similar_slide_index_hamming_distance < 2:
        slide_tracker.mark_slide_as_seen(most_similar_slide_index)
        return True

    pdf_page_text: str = slide_tracker.lecture_slides.plain_texts[most_similar_slide_index]
    ocr_text: str = video_frame.ocr_confident_text

    normalized_pdf_page_text: str = normalize_text(pdf_page_text, keywords_to_ignore)
    normalized_ocr_text: str = normalize_text(ocr_text, keywords_to_ignore)

    similarity = fuzz.token_set_ratio(normalized_pdf_page_text, normalized_ocr_text)
    if similarity >= 75:
        slide_tracker.mark_slide_as_seen(most_similar_slide_index)
        return True

    return False


def detect_slide_transitions(
    video_file_path: str,
    pdf_file_path: str,
    keywords_to_be_matched: set[str],
) -> dict[int, int]:
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
        A dictionary mapping slide indices to their frame number.
    """

    first_slide_video_frame: VideoFrame = detect_first_slide(video_file_path, keywords_to_be_matched)

    # RoI precomputed since it remains constant from now on
    precomputed_roi: tuple[int, int, int, int] = first_slide_video_frame.compute_roi_coordinates()

    lecture_slides: LectureSlides = LectureSlides(pdf_file_path)
    slide_tracker: SlideTracker = SlideTracker(lecture_slides)

    slide_changes: dict[int, int] = {}  # slide_index: frame_number
    for video_frame in generate_video_frame(
        video_path=video_file_path,
        frames_step=100,
        start_frame_number=first_slide_video_frame.frame_number,
        roi_coordinates=precomputed_roi,
    ):
        if is_slide_change_detected(video_frame, slide_tracker, keywords_to_be_matched):
            slide_changes[slide_tracker.current_slide_index + 1] = video_frame.frame_number

    return slide_changes
