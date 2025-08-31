import re
import os

from datetime import timedelta
from typing import TypedDict
from faster_whisper import WhisperModel, BatchedInferencePipeline  # type: ignore
from pathlib import Path

from logs.logging_config import logger


class SRTEntry(TypedDict):
    index: int
    start: float
    end: float
    text: str


class SlideBlock(TypedDict):
    index: int
    start: float
    end: float
    text: str


ROOT_DIR = Path(__file__).resolve().parents[2]


def transcribe_video_to_srt(
    video_file_path: Path,
) -> str:
    """
    Transcribes a video file to SRT format using faster-whisper with batched inference.

    Notes:
    - Models are not downloaded at runtime. We load a local CTranslate2 model
    (https://huggingface.co/collections/Systran/faster-whisper-6867ecec0e757ee14896e2d3)
    from: ``ROOT_DIR / "faster-whisper-models" / <model_size>``.
    Example: "base" -> ROOT_DIR/faster-whisper-models/base

    Args:
        video_file_path: Path to the input video file.
    """
    import logging

    env_model_size = os.getenv("WHISPER_MODEL", "base").strip()

    logging.getLogger("faster_whisper.transcribe").setLevel(logging.INFO)
    logger.info("Transcribing: %s with model: %s", video_file_path.name, env_model_size)
    model_dir = ROOT_DIR / "faster-whisper-models" / f"{env_model_size}"
    if not model_dir.exists():
        logger.error(f"Local faster-whisper model not found: {model_dir}")
        raise FileNotFoundError(f"Local faster-whisper model not found: {model_dir}")
    whisper_model = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
    pipeline = BatchedInferencePipeline(whisper_model)

    segments, _ = pipeline.transcribe(audio=str(video_file_path), batch_size=8, log_progress=True)

    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        srt_lines.append(f"{i}")
        srt_lines.append(f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}")
        srt_lines.append(f"{segment.text.strip()}\n")

    logger.info("Transcription complete: %d segments", len(srt_lines))
    return "\n".join(srt_lines)


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert a float timestamp in seconds to an SRT-compatible time string.

    Args:
        seconds: The time offset in seconds (can include fractions for milliseconds).

    Returns:
        A string formatted as "HH:MM:SS,mmm", suitable for use in SRT subtitle files.
    """
    td: timedelta = timedelta(seconds=seconds)
    total_seconds: int = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds: int = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


def srt_time_to_seconds(srt_timestamp: str) -> float:
    """
    Convert an SRT-formatted timestamp to seconds as a float.

    Args:
        srt_timestamp: Timestamp in the format "HH:MM:SS,mmm" (e.g., "00:02:15,450").

    Returns:
        Time in seconds (e.g., 135.45 for "00:02:15,450").
    """
    hours_str, minutes_str, seconds_milliseconds = srt_timestamp.split(":")
    seconds_str, milliseconds_str = seconds_milliseconds.split(",")

    hours = int(hours_str)
    minutes = int(minutes_str)
    seconds = int(seconds_str)
    milliseconds = int(milliseconds_str)

    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_srt_string(srt_content: str) -> list[SRTEntry]:
    """
    Parses raw SRT content into a list of structured subtitle entries.

    Args:
        srt_content: The full content of the .srt file as a string.

    Returns:
        A list of subtitle entries, each with index, start time, end time, and text.
    """
    entries_raw: list[str] = re.split(r"\n\n+", srt_content.strip())
    entries: list[SRTEntry] = []

    for entry_raw in entries_raw:
        lines: list[str] = entry_raw.strip().splitlines()
        if len(lines) < 3:
            continue  # Skip incomplete or malformed entries

        index: int = int(lines[0])
        time_range: str = lines[1]
        text: str = " ".join(lines[2:]).strip()

        start_str, end_str = time_range.split(" --> ")
        start_sec: float = srt_time_to_seconds(start_str)
        end_sec: float = srt_time_to_seconds(end_str)

        entry: SRTEntry = {
            "index": index,
            "start": start_sec,
            "end": end_sec,
            "text": text,
        }
        entries.append(entry)

    entries.sort(key=lambda x: x["start"])
    return entries


def merge_srt_by_slide_ranges(
    srt_content: str,
    slide_timestamps: dict[int, float],
) -> str:
    """
    Merge SRT entries into slide-aligned blocks based on slide start timestamps.

    Args:
        srt_content: The original SRT content as a string.
        slide_timestamps: Mapping of slide index -> start time (in seconds).

    Returns:
        A string in SRT format where entries are grouped by slide time ranges.
    """
    logger.info("Merging SRT file with %d slide timestamps", len(slide_timestamps))
    srt_entries: list[SRTEntry] = parse_srt_string(srt_content)

    sorted_slide_entries: list[tuple[int, float]] = sorted(slide_timestamps.items(), key=lambda item: item[1])
    video_end_time: float = max(entry["end"] for entry in srt_entries)

    merged_blocks: list[SlideBlock] = []

    for i, (slide_index, start_time) in enumerate(sorted_slide_entries):
        end_time = sorted_slide_entries[i + 1][1] if i + 1 < len(sorted_slide_entries) else video_end_time
        if end_time <= start_time:
            continue

        block_texts: list[str] = [e["text"] for e in srt_entries if start_time <= e["start"] < end_time]

        merged_blocks.append(
            {
                "index": i + 1,
                "start": start_time,
                "end": end_time,
                "text": "\n".join(block_texts),
            }
        )

    merged_srt_lines: list[str] = []
    for block in merged_blocks:
        merged_srt_lines.extend(
            [
                str(block["index"]),
                f"{seconds_to_srt_time(block['start'])} --> {seconds_to_srt_time(block['end'])}",
                block["text"],
                "",
            ]
        )

    logger.info("Created %d merged slide blocks from %d SRT entries", len(merged_blocks), len(srt_entries))
    return "\n".join(merged_srt_lines)
