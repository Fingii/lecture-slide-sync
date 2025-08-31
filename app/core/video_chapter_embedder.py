import subprocess
from pathlib import Path
from logs.logging_config import logger


def generate_ffmpeg_metadata(slide_changes: dict[int, float]) -> str:
    """
    Generate FFmpeg metadata text for chapter embedding.

    Args:
        slide_changes: A dictionary mapping slide numbers to timestamps in seconds.

    Returns:
        A string containing FFMETADATA1-compliant chapter data.
    """
    logger.debug(f"Generating FFmpeg metadata for {len(slide_changes)} slide changes")
    chapters = sorted((timestamp, idx) for idx, timestamp in slide_changes.items())

    metadata = ";FFMETADATA1\n"
    for (timestamp, slide_index), (next_timestamp, _) in zip(chapters, chapters[1:] + [(None, None)]):
        metadata += f"""
[CHAPTER]
TIMEBASE=1/1000
START={int(timestamp * 1000)}
END={int(next_timestamp * 1000) if next_timestamp else int(timestamp * 1000) + 1000}
title=Slide {slide_index}
"""
    logger.debug("Successfully generated FFmpeg metadata")
    return metadata.strip()


def generate_video_with_chapters(
    slide_changes: dict[int, float],
    input_video_path: Path,
    output_dir: Path,
) -> Path:
    """
    Generate a video with embedded chapters at the specified output path.

    Args:
        slide_changes: dictionary mapping slide numbers to timestamps (seconds)
        input_video_path: Path to the input video file
        output_dir: Directory where output video will be saved

    Returns:
        Path to the generated video file with chapters

    Raises:
        RuntimeError: If processing fails
        FileNotFoundError: If input video doesn't exist
    """
    logger.info(f"Starting video chapter generation for {input_video_path.name}")

    if not input_video_path.exists():
        error_msg = f"Input video not found: {input_video_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_video_path.stem}_chapters{input_video_path.suffix}"
    metadata_content = generate_ffmpeg_metadata(slide_changes)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(input_video_path),  # Input video
                "-i",
                "-",  # Metadata from stdin
                "-map_metadata",
                "1",  # Use metadata from 2nd input
                "-codec",
                "copy",  # Copy streams without re-encoding
                "-y",  # Overwrite output
                str(output_path),  # Output file path
            ],
            input=metadata_content,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Successfully generated video with chapters at {output_path}")
        return output_path

    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed with exit code {e.returncode}: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
