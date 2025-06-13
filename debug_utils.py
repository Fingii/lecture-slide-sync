import cv2
import numpy as np
import pyautogui


def show_image_resized(image: np.ndarray, window_name: str = "default", scale_factor: float = 0.8) -> None:
    """
    DEBUG ONLY: DON'T USE IN PRODUCTIVE CODE.
    Displays an image in an OpenCV window, resizes it to fit the screen while keeping the title visible.

    Args:
        image: Image to be displayed.
        window_name: Name of the OpenCV window.
        scale_factor: Scaling factor to reduce the image size (default: 0.9 for 90% of the screen).
    Returns:
        None, displays the given image centered and resized
    """
    screen_height: int = pyautogui.size().height
    screen_width: int = pyautogui.size().width
    image_height: int = image.shape[0]
    image_width: int = image.shape[1]

    scale_factor = min(
        (screen_width * scale_factor) / image_width,
        (screen_height * scale_factor) / image_height,
    )  # Calculate optimal scale factor to fit screen

    new_width: int = max(1, int(image_width * scale_factor))
    new_height: int = max(1, int(image_height * scale_factor))

    resized_image: np.ndarray = cv2.resize(image, (new_width, new_height))  # Resize image

    # Create and resize window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_width, new_height)

    # Center window on screen
    x_pos: int = (screen_width - new_width) // 2
    y_pos: int = (screen_height - new_height) // 2
    cv2.moveWindow(window_name, x_pos, y_pos)

    # Display image
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
