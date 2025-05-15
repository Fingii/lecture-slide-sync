import cv2
from skimage.metrics import structural_similarity as ssim

VIDEO_FILE_PATH = "testVideos/01_DBWT2.mp4"

# Threshold for detecting slide changes
STRUCTURAL_SIMILARITY_THRESHOLD = 0.85


def compute_image_similarity(image1, image2):
    """
        Computes the Structural Similarity Index (SSIM) between two images.

        The function resizes both images, converts them to grayscale, and calculates SSIM
        to determine the similarity between them.

        Args:
            image1 (np.ndarray): First image.
            image2 (np.ndarray): Second image.

        Returns:
            float: Similarity score (range: -1 to 1), where 1 means identical images.

        """
    # Resize images to a smaller size for faster computation
    image1 = cv2.resize(image1, (300, 300))
    image2 = cv2.resize(image2, (300, 300))

    # Convert images to grayscale for similarity calculation
    grayscale_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayscale_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the structural similarity index (SSIM)
    similarity_score, _ = ssim(grayscale_image1, grayscale_image2, full=True)

    return similarity_score


def detect_slide_transitions():
    """
    Analyzes a video file to detect slide transitions based on structural similarity.

    The function processes the video frame by frame, comparing each frame to the previous one.
    A slide change is identified when the similarity score between consecutive frames falls
    below a predefined threshold.

    Args:
        None (uses global VIDEO_FILE_PATH to access the video).

    Returns:
        None (prints detected slide change timestamps to the console).
    """
    video_capture = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not video_capture.isOpened():
        print("Error: Unable to open the video file.")
        return

    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    print(frames_per_second)

    current_frame_index = 0
    previous_video_frame = None

    print("Starting video analysis")
    while video_capture.isOpened():
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
        frame_read_successful, current_video_frame = video_capture.read()
        if not frame_read_successful:
            break

        # For the first frame of the video
        if previous_video_frame is None:
            previous_video_frame = current_video_frame

        if previous_video_frame is not None:
            similarity_value = compute_image_similarity(previous_video_frame, current_video_frame)

            if similarity_value < STRUCTURAL_SIMILARITY_THRESHOLD:
                timestamp_seconds = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                minutes = int(timestamp_seconds // 60)
                seconds = int(timestamp_seconds % 60)
                print(f"Slide changed at {minutes:02d}:{seconds:02d}")

        previous_video_frame = current_video_frame

        current_frame_index += frames_per_second

    video_capture.release()


if __name__ == "__main__":
    detect_slide_transitions()
