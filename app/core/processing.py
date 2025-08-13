from pathlib import Path

from app.core.slide_detection import detect_slide_transitions
from app.core.srt_utils import transcribe_video_to_srt, merge_srt_by_slide_ranges


def generate_merged_srt(
    video_file_path: Path,
    pdf_file_path: Path,
    keywords_to_be_matched: set[str],
    sampling_interval_seconds: float,
) -> tuple[str, dict[int, float]]:
    """
    Generate merged subtitles by analyzing video and PDF slides.

    Args:
        video_file_path: Path to the video file
        pdf_file_path: Path to the PDF slides file
        keywords_to_be_matched: Set of keywords to detect in slides
        sampling_interval_seconds: Time interval for slide analysis (in seconds)

    Returns:
        str: Merged SRT content with slide-aligned subtitles
    """
    srt_content: str = transcribe_video_to_srt(video_file_path)
    slide_changes: dict[int, float] = detect_slide_transitions(
        video_file_path=video_file_path,
        pdf_file_path=pdf_file_path,
        keywords_to_be_matched=keywords_to_be_matched,
        sampling_interval_seconds=sampling_interval_seconds,
    )
    return merge_srt_by_slide_ranges(srt_content, slide_changes), slide_changes
