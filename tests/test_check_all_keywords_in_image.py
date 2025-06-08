import cv2
from pathlib import Path
import numpy as np
from slideDetection import check_all_keywords_in_image


def pytest_generate_tests(metafunc) -> None:

    TEST_DATA_PATH: Path = (
        Path(__file__).parent / "test_data" / "test_check_all_keywords_in_image"
    )

    if "valid_image" in metafunc.fixturenames:
        valid_images: list[Path] = list((TEST_DATA_PATH / "valid").iterdir())
        metafunc.parametrize(
            "valid_image", valid_images, ids=[p.name for p in valid_images]
        )

    if "partial_image" in metafunc.fixturenames:
        partial_images: list[Path] = list((TEST_DATA_PATH / "partial").iterdir())
        metafunc.parametrize(
            "partial_image", partial_images, ids=[p.name for p in partial_images]
        )


class TestKeywordDetection:
    def test_valid_images(self, valid_image) -> None:
        img: np.ndarray = cv2.imread(str(valid_image))
        assert img is not None, f"Could not load image {valid_image}"
        assert check_all_keywords_in_image(img) is True, f"Failed on {valid_image.name}"

    def test_partial_images(self, partial_image) -> None:
        img: np.ndarray = cv2.imread(str(partial_image))
        assert img is not None, f"Could not load image {partial_image}"
        assert (
            check_all_keywords_in_image(img) is False
        ), f"Failed on {partial_image.name}"
