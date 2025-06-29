import numpy as np
import cv2
import re

from config import DEBUG_MODE
from debug_utils import show_image_resized
from video_frame import VideoFrame


def _visualize_keyword_matches(
    frame: np.ndarray, ocr_data: dict[str, list[str | int]], found_keywords: set[str]
) -> None:
    """
    DEBUG ONLY: DON'T USE IN PRODUCTIVE CODE.

    Displays an image with bounding boxes around words that match the given keywords,
    using OCR-extracted text data. The function searches for keywords in the text,
    draws green rectangles around matches, and labels them for visualization.

    Creates a deep copy of the given image (`frame_copy`) to ensure
    the original image remains unchanged.

    Args:
        frame: The original image (not modified).
        ocr_data: Dictionary containing OCR-extracted text and bounding box coordinates
                  (keys: "text", "left", "top", "width", "height").
        found_keywords: Set of keywords to search for in the OCR text. Matching words
                        will be highlighted in the copied image.

    Returns:
        None, modifies a deep copy of the given image by adding bounding boxes and labels,
        then displays it. The original image remains unchanged.
    """

    frame_copy: np.ndarray = frame.__deepcopy__({})
    bounding_boxes: dict[str, tuple[int, int, int, int]] = {}

    for i in range(len(ocr_data["text"])):
        text = str(ocr_data["text"][i]).strip()
        if not text:
            continue

        for keyword in found_keywords:
            if re.search(rf"\b{keyword}\b", text, re.IGNORECASE):
                textbox_leftmost_x_position = int(ocr_data["left"][i])
                textbox_topmost_y_position = int(ocr_data["top"][i])
                text_box_width = int(ocr_data["width"][i])
                text_box_height = int(ocr_data["height"][i])

                bounding_boxes[keyword] = (
                    textbox_leftmost_x_position,
                    textbox_topmost_y_position,
                    textbox_leftmost_x_position + text_box_width,
                    textbox_topmost_y_position + text_box_height,
                )
                break

    for keyword, (x1, y1, x2, y2) in bounding_boxes.items():
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_copy,
            keyword,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    show_image_resized(frame_copy)


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

    if DEBUG_MODE and keywords.issubset(found_keywords):
        _visualize_keyword_matches(video_frame.full_frame, ocr_data, found_keywords)

    return keywords.issubset(found_keywords)
