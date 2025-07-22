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
    seen_slide_indices: set[int] = field(init=False, default_factory=set)
    _max_hamming_distance: int = 8

    def find_most_similar_slide_index(self, video_frame: VideoFrame) -> tuple[int, float] | None:
        """
        Compares the hash of the given video frame to all slide hashes and finds the most similar match.

        Args:
            video_frame: The video frame whose RoI hash is to be matched.

        Returns:
            A tuple of (most similar slide index, hamming distance) if below threshold, otherwise None.
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
            return most_similar_slide_index, min_hamming_distance
        return None

    def mark_slide_as_seen(self, index: int) -> None:
        self.seen_slide_indices.add(index)
        self.current_slide_index = index

    def has_seen_slide(self, index: int) -> bool:
        return index in self.seen_slide_indices
