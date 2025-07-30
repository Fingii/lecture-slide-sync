import re

from app.models.video_frame import VideoFrame
from logs.logging_config import logger


def _get_matching_keywords(words: set[str], keywords_to_match: set[str]) -> set[str]:
    """
    Identifies keywords that are present in a given set of words using case-insensitive matching with word boundaries.

    Searches for each keyword within the provided words, ensuring that matches occur as standalone words
    (not as substrings within other words). The search is case-insensitive.
    If a keyword is found within any word, it is added to the result set.

    Args:
        words: A set of words to search within.
        keywords_to_match: A set of keywords to look for in the given words. Matching is case-insensitive and respects word boundaries.

    Returns:
        A set of keywords that were found in the provided words. Only exact word matches (not substrings) are included.
    """
    found_keywords: set[str] = set()

    for keyword in keywords_to_match:
        for text in words:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                found_keywords.add(keyword)
                break

    return found_keywords


def filter_words_by_confidence(
    ocr_data: dict[str, list[str | int]], confidence_threshold: int = 80
) -> set[str]:
    """
    Filters words from OCR data based on a confidence threshold.

    Args:
        ocr_data: A dictionary containing OCR-extracted text and confidence scores.
        confidence_threshold: Minimum confidence score required for a word to be included (0-100).

    Returns:
        A set of words that meet the confidence threshold.
    """
    valid_words: set[str] = set()

    for i in range(len(ocr_data["text"])):
        text: str = str(ocr_data["text"][i]).strip()
        conf: int = int(ocr_data["conf"][i])

        if conf >= confidence_threshold:
            valid_words.add(text)

    return valid_words


def are_all_keywords_present(
    video_frame: VideoFrame,
    keywords: set[str],
    confidence_threshold: int = 80,
) -> bool:
    """
    Checks whether all specified keywords appear in the image using OCR.

    Args:
        video_frame: The video frame to check in if all keywords are present
        keywords: Set of expected words (case-insensitive). Set is used because the order doesn't matter.
        confidence_threshold: Minimum OCR confidence level (0-100).

    Returns:
        True if all keywords are found, False otherwise.
    """

    ocr_data: dict[str, list[str | int]] = video_frame.ocr_data_full_frame
    valid_words: set[str] = filter_words_by_confidence(ocr_data, confidence_threshold)
    valid_words_non_empty: set[str] = {word.strip() for word in valid_words if word.strip()}
    found_keywords: set[str] = _get_matching_keywords(valid_words_non_empty, keywords)

    logger.debug(
        "Frame %d: OCR match=%s | found=%s | required=%s",
        video_frame.frame_number,
        keywords.issubset(found_keywords),
        found_keywords,
        keywords,
    )
    return keywords.issubset(found_keywords)
