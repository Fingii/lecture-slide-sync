from dataclasses import dataclass, field

from hashing_utils import compute_hamming_distance
from lecture_slides import LectureSlides
from video_frame import VideoFrame


@dataclass
class SlideTracker:
    """
    Tracks slide transitions in a lecture video by matching video frame hashes
    against precomputed hashes of slides from a corresponding PDF.

    This class maintains internal state to ensure only forward slide transitions
    are recognized and provides utility methods to match frames and update state.

    Attributes:
        lecture_slides: A LectureSlides instance providing slide images and their hashes.
        current_slide_index: Index of the most recently detected slide (initialized to -1).
        _max_hamming_distance: Maximum allowed distance between perceptual hashes for a match.
    """

    lecture_slides: LectureSlides
    current_slide_index: int = field(init=False, default=-1)
    _max_hamming_distance: int = 8

    def find_most_similar_slide_index(self, video_frame: VideoFrame) -> int | None:
        """
        Compares the hash of the given video frame to all slide hashes and finds the most similar match.

        Args:
            video_frame: The video frame whose RoI hash is to be matched.

        Returns:
            The slide index which was most similar to the given video frame
        """
        current_hash: str = video_frame.roi_hash
        min_hamming_distance: float = float("inf")
        most_similar_slide_index: int = -1

        for i, slide_hash in enumerate(self.lecture_slides.hashes):
            hamming_distance: float = compute_hamming_distance(current_hash, slide_hash)
            if hamming_distance < min_hamming_distance:
                min_hamming_distance = hamming_distance
                most_similar_slide_index = i

        if min_hamming_distance <= self._max_hamming_distance:
            return most_similar_slide_index
        return None

    def update_slide_index(self, new_index: int) -> None:
        """
        Updates the tracker's internal current slide index to the given value.

        Intended to be called externally after confirming a valid forward transition.

        Args:
            new_index: Index of the newly confirmed slide.
        """
        self.current_slide_index = new_index
