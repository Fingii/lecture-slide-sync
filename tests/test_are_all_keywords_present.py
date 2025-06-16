import cv2
from pathlib import Path
import numpy as np
from ocr_keyword_detector import are_all_keywords_present
from video_frame import VideoFrame

keywords_to_be_matched: set[str] = {"UNIVERSITY", "FH", "AACHEN", "OF", "APPLIED", "SCIENCES"}


def pytest_generate_tests(metafunc) -> None:

    TEST_DATA_PATH: Path = Path(__file__).parent / "test_data" / "test_check_all_keywords_in_image"

    if "valid_image" in metafunc.fixturenames:
        valid_images: list[Path] = list((TEST_DATA_PATH / "valid").iterdir())
        metafunc.parametrize("valid_image", valid_images, ids=[p.name for p in valid_images])

    if "partial_image" in metafunc.fixturenames:
        partial_images: list[Path] = list((TEST_DATA_PATH / "partial").iterdir())
        metafunc.parametrize("partial_image", partial_images, ids=[p.name for p in partial_images])


class TestKeywordDetection:
    def test_valid_images(self, valid_image) -> None:
        img: np.ndarray = cv2.imread(str(valid_image))
        assert img is not None, f"Could not load image {valid_image}"

        video_frame = VideoFrame(full_frame=img, frame_number=0)
        assert (
            are_all_keywords_present(video_frame, keywords_to_be_matched) is True
        ), f"Failed on {valid_image.name}"

    def test_partial_images(self, partial_image) -> None:
        img: np.ndarray = cv2.imread(str(partial_image))
        assert img is not None, f"Could not load image {partial_image}"

        video_frame = VideoFrame(full_frame=img, frame_number=0)
        assert (
            are_all_keywords_present(video_frame, keywords_to_be_matched) is False
        ), f"Failed on {partial_image.name}"
