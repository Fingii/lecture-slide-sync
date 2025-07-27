from dataclasses import dataclass
from functools import cached_property
from PIL import Image
from pathlib import Path

import numpy as np
import cv2
import pymupdf  # type: ignore

from hashing_utils import compute_phashes


@dataclass
class LectureSlides:
    """
    Represents a collection of lecture slides loaded from a PDF file.

    This class loads all pages of the provided PDF, converts them into OpenCV-compatible
    images, and computes perceptual hashes for each slide. These hashes are used to
    match video frames against slide images.
    """

    pdf_file_path: Path

    @cached_property
    def _images(self) -> list[np.ndarray]:
        """
        Extracts each single slide from a PDF and converts them into OpenCV images.

        Loads each slide (page) of the given PDF, gets the pixel map of it,
        converts it into a Pillow image, and transforms it into an OpenCV-compatible format (BGR `np.ndarray`).
        """
        pdf_document: pymupdf.Document = pymupdf.open(self.pdf_file_path)
        images: list[np.ndarray] = []

        for i in range(pdf_document.page_count):
            pixmap: pymupdf.Pixmap = pdf_document.get_page_pixmap(i, dpi=200)
            pil_img: Image.Image = pixmap.pil_image()
            cv2_img: np.ndarray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            images.append(cv2_img)

        pdf_document.close()
        return images

    @cached_property
    def hashes(self) -> list[str]:
        return compute_phashes(self._images)

    @cached_property
    def plain_texts(self) -> list[str]:
        """
        Extracts raw text from each slide in the PDF.
        """
        pdf_document = pymupdf.open(self.pdf_file_path)
        all_text = [page.get_text() for page in pdf_document]
        pdf_document.close()
        return all_text
