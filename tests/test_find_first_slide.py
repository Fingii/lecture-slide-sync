import cv2
import pytest
import numpy as np
import json
from pathlib import Path
from app.core.slide_detection import detect_first_slide
from app.models.video_frame import VideoFrame

TEST_DATA_ROOT = Path(__file__).parent / "test_data"
FRAME_COUNTS_FILE = TEST_DATA_ROOT / "test_find_first_slide" / "frame_counts.json"
EXPECTED_SLIDES_DIR = TEST_DATA_ROOT / "test_find_first_slide"
VIDEOS_DIR = TEST_DATA_ROOT / "videos"


def load_frame_counts() -> dict[str, int]:
    """
    Load the first detected slide frame counts from a JSON file.

    The JSON file contains a mapping of video file paths to the frame number where the first slide was detected.
    Each key is a string representing a relative video file path, and each value is an integer representing
    the frame index where the slide first appears.

    Example JSON structure:
    {
        "dbwt1/dbwt1_01.mp4": 45,
        "ads/ads_01.mp4": 0
    }

    Returns:
        A dictionary where keys are video file paths and values are the first detected slide frame counts.
    """
    if not FRAME_COUNTS_FILE.exists():
        raise FileNotFoundError(f"Frame count file not found: {FRAME_COUNTS_FILE}")
    with open(FRAME_COUNTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def get_test_cases_with_slide_prefix(
    slide_prefix: str, include_frame_count: bool = True
) -> list[tuple[Path, Path, int]] | list[tuple[Path, Path]]:
    """
    Generates a list of test cases by pairing lecture videos with their
    corresponding slide images (with prefix) and their first detected slide frame count.

    Scans `EXPECTED_SLIDES_DIR`, identifying each lecture module and extracting
    its manually captured slide images. It then constructs the path to the matching video
    from `VIDEOS_DIR`, ensuring that both files exist before adding the pair to the test cases.
    Additionally, it retrieves the first detected slide frame count from frame_counts.json.

    **Folder structure example:**

    tests/
    ├── test_find_first_slide.py
    └── test_data/
        ├── test_find_first_slide/
        │   ├── dbwt1/
        │   │   ├── first_slide_dbwt1_01.png
        │   │   ├── not_first_slide_dbwt1_01.png
        │   │   └── ...
        │   ├── ads/
        │   │   ├── first_slide_ads_01.png
        │   │   ├── not_first_slide_ads_01.png
        │   │   └── ...
        │   ├── frame_counts.json (Manually extracted frame count, when the first slide occurs)
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
        include_frame_count: Whether to include the first detected slide frame count in the returned tuples.
        If `True`, each tuple contains `(video_path, slide_path, frame_count)`.
        If `False`, each tuple contains only `(video_path, slide_path)`.
    Returns:
        A list of tuples, each containing:
        - The path to a lecture video (`VIDEOS_DIR/module_name/module_name_video_num.mp4`)
        - The path to its corresponding first slide image (`EXPECTED_SLIDES_DIR/module_name/first_slide_module_name_video_num.png`)
        - The frame count where the first slide was detected
    """

    frame_counts: dict[str, int] = load_frame_counts()
    test_cases_with_frame: list[tuple[Path, Path, int]] = []
    test_cases_without_frame: list[tuple[Path, Path]] = []

    for module_dir in EXPECTED_SLIDES_DIR.iterdir():
        if not module_dir.is_dir():
            continue

        module_name: str = module_dir.name

        for slide_path in module_dir.glob(f"{slide_prefix}*.png"):
            video_num: str = slide_path.stem.split("_")[-1]
            video_filename = f"{module_name}_{video_num}.mp4"
            video_path: Path = VIDEOS_DIR / module_name / video_filename

            if not video_path.exists():
                continue

            if not include_frame_count:
                test_cases_without_frame.append((video_path, slide_path))
                continue

            # Get the frame count from loaded data
            video_key: str = f"{module_name}/{video_filename}"

            if video_key not in frame_counts:
                continue

            frame_count: int = frame_counts.get(video_key, -1)  # Default to -1 if missing

            if frame_count == -1:
                continue

            test_cases_with_frame.append((video_path, slide_path, frame_count))

    return test_cases_with_frame if include_frame_count else test_cases_without_frame


@pytest.mark.parametrize(
    "video_path, expected_slide_path, expected_frame_count",
    get_test_cases_with_slide_prefix("first_slide", include_frame_count=True),
)
def test_finds_correct_first_slide_same_image_dimensions(
    video_path: Path, expected_slide_path: Path, expected_frame_count: int
) -> None:
    """
    Ensures that first-slides are correctly accepted by verifying that the detected frame
    is identical to the given manually captured first-slide, which was saved using cv2.imwrite
    (with no compression) at the exact first slide frame.

    The test checks if the images are the same by computing their absolute pixel differences.
    If the total sum of differences = 0, then they are identical, and the test succeeds.

    Args:
        video_path: Path to the lecture video file.
        expected_slide_path: Path to the manually captured first slide image.
        expected_frame_count: The frame count where the first slide would be
    """
    result: VideoFrame = detect_first_slide(str(video_path))
    assert result is not None, f"No slide detected in video: {video_path}"

    detected_first_slide: np.ndarray = result.full_frame
    detected_first_slide_frame_count: int = result.frame_number

    expected_frame: np.ndarray | None = cv2.imread(str(expected_slide_path), cv2.IMREAD_UNCHANGED)

    assert expected_frame is not None, f"Could not read expected frame: {expected_slide_path}"
    assert detected_first_slide.shape == expected_frame.shape, "Image dimensions mismatch"

    # Compute absolute difference
    difference: np.ndarray = cv2.absdiff(detected_first_slide, expected_frame)

    assert np.sum(difference) == 0, f"The images are not exactly the same, difference != 0: {difference:.2f}"

    assert (
        detected_first_slide_frame_count == expected_frame_count
    ), f"Frame count mismatch. Expected {expected_frame_count}, got {detected_first_slide_frame_count}"


@pytest.mark.parametrize(
    "video_path, not_first_slide_path",
    get_test_cases_with_slide_prefix("not_first_slide", include_frame_count=False),
)
def test_rejects_non_first_slide_same_image_dimensions(video_path: Path, not_first_slide_path: Path) -> None:
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

    result: VideoFrame = detect_first_slide(str(video_path))
    assert result is not None, f"No slide detected in video: {video_path}"

    detected_first_slide: np.ndarray = result.full_frame

    non_first_slide: np.ndarray = cv2.imread(str(not_first_slide_path), cv2.IMREAD_UNCHANGED)

    assert detected_first_slide.shape == non_first_slide.shape, "Image dimensions mismatch"

    assert non_first_slide is not None, f"Failed to load expected frame: {not_first_slide_path}"

    # Compute absolute difference
    difference: np.ndarray = cv2.absdiff(detected_first_slide, non_first_slide)
    assert np.sum(difference) != 0, "Detected slide incorrectly matches a non-first slide"
