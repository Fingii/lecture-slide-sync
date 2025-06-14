from PIL import Image
import numpy as np
import cv2
import pymupdf  # type: ignore


def convert_pdf_slides_to_images(pdf_path: str) -> list[np.ndarray]:
    """
    Extracts each single slide from a PDF and converts them into OpenCV images.

    This function loads each slide (page) of the given PDF, gets the pixel map of it,
    converts it into a Pillow image, and transforms it into an OpenCV-compatible format (BGR `np.ndarray`).
    The extracted slides are returned as a list of OpenCV images for further processing.

    Args:
        pdf_path: Path to the PDF file containing slides.

    Returns:
        A list of OpenCV images, each representing a slide from the PDF, in order.
    """
    pdf_document: pymupdf.Document = pymupdf.open(pdf_path)
    pdf_slide_images_cv2: list[np.ndarray] = []

    for i in range(pdf_document.page_count):
        current_page_pixmap: pymupdf.Pixmap = pdf_document.get_page_pixmap(i, dpi=200)
        pillow_image: Image.Image = current_page_pixmap.pil_image()
        open_cv_image: np.ndarray = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
        pdf_slide_images_cv2.append(open_cv_image)

    pdf_document.close()
    return pdf_slide_images_cv2
