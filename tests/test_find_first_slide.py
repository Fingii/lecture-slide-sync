import cv2
import pytest
import numpy as np
from pathlib import Path
from slideDetection import find_first_slide


def get_test_cases_with_slide_prefix(
    slide_prefix: str,
) -> list[tuple[Path, Path]]:
    """
    Generates a list of test cases by pairing lecture videos with their
    corresponding slide images (with prefix).

    Scans `EXPECTED_SLIDES_DIR`, identifying each lecture module and extracting
    its manually captured slide images. It then constructs the path to the matching video
    from `VIDEOS_DIR`, ensuring that both files exist before adding the pair to the test cases.

    Folder structure example:
    tests/
    ├── test_find_first_slide.py
    └── test_data/
        ├── test_find_first_slide/
        │   ├── dbwt1/
        │   │   ├── first_slide_dbwt1_01.jpg
        │   │   ├── not_first_slide_dbwt1_01.jpg
        │   │   └── ...
        │   ├── ads/
        │   │   ├── first_slide_ads_01.jpg
        │   │   ├── not_first_slide_ads_01.jpg
        │   │   └── ...
        │   └── ...
        └── videos/
            ├── dbwt1/
            │   ├── dbwt1_01.mp4
            │   └── ...
            ├── ads/
            │   ├── ads_01.mp4
            │   └── ...
            └── ...

    Args:
        slide_prefix: The prefix for the slide image files (e.g., 'first_slide_', 'not_first_slide_').

    Returns:
        A list of tuples, each containing:
            - The path to a lecture video (`VIDEOS_DIR/module_name/module_name_video_num.mp4`)
            - The path to its corresponding first slide image (`EXPECTED_SLIDES_DIR/module_name/first_slide_module_name_video_num.jpg`)
    """

    TEST_DATA_ROOT = Path(__file__).parent / "test_data"
    EXPECTED_SLIDES_DIR = TEST_DATA_ROOT / "test_find_first_slide"
    VIDEOS_DIR = TEST_DATA_ROOT / "videos"

    test_cases: list[tuple[Path, Path]] = []
    for module_dir in EXPECTED_SLIDES_DIR.iterdir():

        if module_dir.is_dir():
            module_name: str = module_dir.name

            for slide_path in module_dir.glob(f"{slide_prefix}*.png"):
                video_num: str = slide_path.stem.split("_")[-1]
                video_path: Path = (
                    VIDEOS_DIR / module_name / f"{module_name}_{video_num}.mp4"
                )

                if video_path.exists():
                    test_cases.append((video_path, slide_path))
    return test_cases


@pytest.mark.parametrize(
    "video_path, expected_slide_path",
    get_test_cases_with_slide_prefix("first_slide"),
)
def test_finds_correct_first_slide_same_image_dimensions(
    video_path: Path, expected_slide_path: Path
):
    """
    Ensures that first-slides are correctly accepted by verifying that the detected frame
    is identical to the given manually captured first-slide, which was saved using cv2.imwrite
    (with no compression) at the exact first slide frame.

    The test checks if the images are the same by computing their absolute pixel differences.
    If the total sum of differences = 0, then they are identical, and the test succeeds.

    Args:
        video_path: Path to the lecture video file.
        expected_slide_path: Path to the manually captured first slide image.

    """
    result_frame: np.ndarray | None = find_first_slide(str(video_path))
    expected_frame: np.ndarray | None = cv2.imread(
        str(expected_slide_path), cv2.IMREAD_UNCHANGED
    )

    # Compute absolute difference
    difference: np.ndarray = cv2.absdiff(result_frame, expected_frame)

    assert (
        np.sum(difference) == 0
    ), f"The images are not exactly the same, difference != 0: {difference:.2f}"


@pytest.mark.parametrize(
    "video_path, not_first_slide_path",
    get_test_cases_with_slide_prefix("not_first_slide"),
)
def test_rejects_non_first_slide_same_image_dimensions(
    video_path: Path, not_first_slide_path
):
    """
    Ensures that NON first-slides are correctly rejected by verifying that the detected frame
    is NOT identical to the given manually captured first-slide, which was saved using cv2.imwrite
    (with no compression) at the exact first slide frame.

    The test checks if the images are different by computing their absolute pixel differences.
    If the total sum of differences != 0, then they are NOT identical, and the test succeeds.

    Args:
        video_path: Path to the lecture video file.
        not_first_slide_path: Path to a manually captured slide that is NOT the first one.
    """
    result_frame: np.ndarray | None = find_first_slide(str(video_path))
    expected_frame: np.ndarray | None = cv2.imread(
        str(not_first_slide_path), cv2.IMREAD_UNCHANGED
    )

    difference: np.ndarray = cv2.absdiff(result_frame, expected_frame)

    assert (
        np.sum(difference) != 0
    ), f"The images are not exactly the same, difference != 0: {difference:.2f}"
