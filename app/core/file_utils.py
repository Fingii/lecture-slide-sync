import shutil
import zipfile
import starlette.datastructures

from fastapi import UploadFile
from pathlib import Path
from logs.logging_config import logger


def save_upload_to_file(upload_file: UploadFile, target_dir: Path) -> Path:
    """
    Save an uploaded file to target directory.

    Args:
        upload_file: FastAPI UploadFile object
        target_dir: Target directory to save the file

    Returns:
        Path to the saved file

    Raises:
        ValueError: If upload_file has no filename
        RuntimeError: If save operation fails
    """
    if not upload_file.filename:
        logger.error("UploadFile provided with no filename")
        raise ValueError("UploadFile must have a filename")

    logger.debug(f"Saving uploaded file: {upload_file.filename}, Target dir: {target_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path: Path = target_dir / upload_file.filename

    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logger.debug(f"Successfully saved upload {upload_file.filename} to {target_dir}")
        return target_path
    except Exception as e:
        logger.error(f"Failed to save upload: {str(e)}", exc_info=True)
        raise RuntimeError(f"Upload save operation failed: {str(e)}") from e


def save_str_to_file(content: str, target_path: Path, encoding: str = "utf-8") -> Path:
    """
    Save text content to a file.

    Args:
        content: String content to save
        target_path: Full path including filename with extension
        encoding: Text encoding to use

    Returns:
        Path to saved file

    Raises:
        RuntimeError: If save operation fails
    """
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with target_path.open("w", encoding=encoding) as f:
            f.write(content)

        logger.debug(f"Saved text content to {target_path}")
        return target_path

    except Exception as e:
        logger.error(f"Failed to save text file: {str(e)}", exc_info=True)
        raise RuntimeError(f"Text save failed: {str(e)}") from e


def copy_file(source_path: Path, target_dir: Path) -> Path:
    """
    Copy an existing file to target directory.

    Args:
        source_path: Path to existing file
        target_dir: Target directory to copy the file

    Returns:
        Path to the copied file

    Raises:
        FileNotFoundError: If source file doesn't exist
        RuntimeError: If copy operation fails
    """
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path: Path = target_dir / source_path.name

    try:
        shutil.copy2(source_path, target_path)
        logger.debug(f"Successfully copied {source_path.name} to {target_dir}")
        return target_path
    except Exception as e:
        logger.error(f"Failed to copy file: {str(e)}", exc_info=True)
        raise RuntimeError(f"File copy operation failed: {str(e)}") from e


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


def zipping_directory(source_dir: Path, output_zip_path: Path | None = None) -> Path:
    """
    Create a ZIP file from all files in a directory.

    Args:
        source_dir: Directory containing files to zip
        output_zip_path: Where to save the ZIP file. Defaults to source_dir's parent

    Returns:
        Path to the created ZIP file

    Raises:
        FileNotFoundError: If source directory doesn't exist
        ValueError: If source directory is empty
        RuntimeError: If ZIP creation fails
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files_to_zip: list[Path] = list(source_dir.rglob("*"))
    files_to_zip = [f for f in files_to_zip if f.is_file()]

    if not files_to_zip:
        raise ValueError(f"No files found in directory: {source_dir}")

    if output_zip_path is None:
        output_zip_path = source_dir.parent / f"{source_dir.name}.zip"

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(output_zip_path, "w") as zipf:
            for file_path in files_to_zip:
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)
                logger.debug(f"Added {file_path.name} to ZIP")

        logger.info(f"Created ZIP with {len(files_to_zip)} files: {output_zip_path}")
        return output_zip_path

    except Exception as e:
        logger.error(f"Failed to create ZIP from directory: {str(e)}", exc_info=True)
        raise RuntimeError(f"ZIP creation failed: {str(e)}") from e
