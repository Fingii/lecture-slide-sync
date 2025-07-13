import av

from video_utils import generate_video_frame
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame
from lecture_slides import LectureSlides
from slide_tracker import SlideTracker

from fractions import Fraction


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


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """
    Calculates the Jaccard similarity between two sets of strings.

    Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the two sets.

    Args:
        set1: First set of strings.
        set2: Second set of strings.

    Returns:
        A float value between 0 and 1 representing the Jaccard similarity.
        Returns 0.0 if both sets are empty.
    """
    if not set1 and not set2:
        return 0.0

    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union)


def normalize_token_set(
    original_token_set: set[str],
    keywords_to_ignore: set[str] | None = None,
    remove_short_tokens: bool = True,
    lowercase: bool = True,
    min_length: int = 2,
) -> set[str]:
    """
    Normalizes a set of tokens for text similarity comparisons.

    Args:
        original_token_set: The original set of tokens.
        keywords_to_ignore: Set of keywords to remove from the token set.
        remove_short_tokens: If True, removes tokens below min_length.
        lowercase: If True, converts all tokens to lowercase.
        min_length: Minimum token length (inclusive) to keep if remove_short_tokens is enabled.

    Returns:
        A new set of normalized tokens.
    """

    if keywords_to_ignore is None:
        keywords_to_ignore = set()

    normalized_set: set[str] = set()

    for token in original_token_set:
        if lowercase:
            token = token.lower()

        if token in keywords_to_ignore:
            continue

        if remove_short_tokens and len(token) < min_length:
            continue

        normalized_set.add(token)

    return normalized_set


def is_slide_change_detected(
    video_frame: VideoFrame, slide_tracker: SlideTracker, keywords_to_ignore: set[str]
) -> bool:
    """
    Determines whether the current video frame represents a new forward slide transition.

    This function first matches the frame to a known slide using perceptual hashing.
    If the hash match is very strong (distance < 2), the match is accepted directly.
    Otherwise, the function uses OCR-based Jaccard similarity to confirm the match.
    Tokens such as predefined recurring keywords can be excluded from this comparison.

    Args:
        video_frame: The current VideoFrame to evaluate.
        slide_tracker: An instance of SlideTracker managing slide state and hash matching.
        keywords_to_ignore: A set of common tokens (e.g. "FH", "AACHEN") to exclude from similarity checks.

    Returns:
        True if a valid new slide is detected and confirmed, False otherwise.
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

    pdf_tokens: set[str] = slide_tracker.lecture_slides.word_tokens[most_similar_slide_index]
    normalized_pdf_tokens: set[str] = normalize_token_set(pdf_tokens, keywords_to_ignore)

    normalized_video_frame_ocr_tokens: set[str] = normalize_token_set(
        video_frame.ocr_word_tokens, keywords_to_ignore
    )

    token_similarity: float = jaccard_similarity(normalized_pdf_tokens, normalized_video_frame_ocr_tokens)

    if token_similarity >= 0.65:
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
