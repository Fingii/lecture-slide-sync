import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Parameter
VIDEO_PATH = "testVideos/01_DBWT2.mp4"
SSIM_THRESHOLD = 0.85  # Schwellenwert für den Folienwechsel


# Ähnlichkeit zwischen zwei Bildern berechnen
def compute_similarity(img1, img2):
    # Bilder auf eine kleinere Größe reduzieren, um die Berechnungen zu beschleunigen
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)

    return score


# Video analysieren und Folienwechsel erkennen
def detectSlideChanges():
    # Öffne das Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_idx = 0
    previous_frame = None

    print("Starte Videoanalyse...")
    while cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if previous_frame is None:
            previous_frame = frame

        if previous_frame is not None:
            similarity = compute_similarity(previous_frame, frame)

            if similarity < SSIM_THRESHOLD:
                # cv2.imshow('img1', previous_frame)
                # cv2.waitKey(0)
                # cv2.imshow('img2', frame)
                # cv2.waitKey(0)
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                print(f"Folie gewechselt bei {minutes:02d}:{seconds:02d}")

        previous_frame = frame

        frame_idx += fps

    cap.release()


if __name__ == "__main__":
    detectSlideChanges()
