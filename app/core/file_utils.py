import shutil
import zipfile
import starlette.datastructures

from typing import Any
from fastapi import UploadFile
from io import BytesIO
from pathlib import Path
from logs.logging_config import logger


def save_file(source: UploadFile | Path, target_dir: Path) -> Path:
    """
    Save either an UploadFile or existing Path to target directory with detailed logging.
    Returns path to saved file.

    Args:
        source: Either an UploadFile or existing Path object
        target_dir: Target directory to save the file

    Returns:
        Path: Path to the saved file

    Raises:
        ValueError: For invalid UploadFile with no filename
        Exception: For any file operation errors
    """

    logger.debug(f"Saving file. Type: {type(source)}, Target dir: {target_dir}")

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(source, starlette.datastructures.UploadFile):
            if not source.filename:
                logger.error("UploadFile provided with no filename")
                raise ValueError("UploadFile must have a filename")

            target_path = target_dir / source.filename
            with target_path.open("wb") as buffer:
                shutil.copyfileobj(source.file, buffer)
                logger.debug(f"Successfully saved {source.filename} to {target_dir}")
                return target_path

        elif isinstance(source, Path):
            target_path = target_dir / source.name
            shutil.copy2(source, target_path)
            logger.debug(f"Successfully saved {source.name} to {target_dir}")
            return target_path

    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}", exc_info=True)
        raise RuntimeError(f"File save operation failed: {str(e)}") from e


def find_matching_video_pdf_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """
    Find video-PDF pairs where filenames match EXACTLY before extension.
    Example: "lecture1.mp4" will ONLY match with "lecture1.pdf"
    """
    video_extensions: set[str] = {".mp4", ".mov", ".avi"}
    video_pdf_pairs: list[tuple[Path, Path]] = []

    pdf_stems: set[str] = {f.stem for f in directory.glob("*.pdf")}

    for ext in video_extensions:
        for video_path in directory.glob(f"*{ext}"):
            if video_path.stem in pdf_stems:
                pdf_path: Path = directory / f"{video_path.stem}.pdf"
                video_pdf_pairs.append((video_path, pdf_path))

    logger.info(f"Found {len(video_pdf_pairs)} exact-name video-PDF pairs")
    return video_pdf_pairs


def extract_zip(zip_path: Path, extract_to: Path | None = None) -> Path:
    """
    Extract a ZIP file to target directory. If no target directory is specified,
    extracts to a folder named 'extracted' in the same directory as the ZIP file.

    Args:
        zip_path: Path to the ZIP file
        extract_to: Optional target directory for extraction. Defaults to {zip_path.parent}/extracted

    Returns:
        Path: Path to the extracted directory

    Raises:
        RuntimeError: If extraction fails
        FileNotFoundError: If zip_path doesn't exist
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    resolved_target = extract_to if extract_to is not None else zip_path.parent / "extracted"

    try:
        resolved_target.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(resolved_target)
            logger.debug(f"Extracted {zip_path.name} to {resolved_target}")

        return resolved_target

    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file: {zip_path}")
        raise RuntimeError(f"Invalid ZIP file: {zip_path}") from e
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {str(e)}", exc_info=True)
        raise RuntimeError(f"ZIP extraction failed: {str(e)}") from e


def create_zip_from_content(content: dict[str, Any]) -> BytesIO:
    """
    Create a ZIP file in memory from dictionary content.

    Args:
        content: Dictionary of {filename: content} where content can be str or bytes

    Returns:
        BytesIO buffer containing the ZIP file

    Raises:
        ValueError: If content is empty
    """
    if not content:
        raise ValueError("No content provided to create ZIP")

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename, file_content in content.items():
            if isinstance(file_content, str):
                file_content = file_content.encode("utf-8")
            zipf.writestr(filename, file_content)

    zip_buffer.seek(0)
    return zip_buffer
