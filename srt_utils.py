import re
from datetime import timedelta
from typing import TypedDict
from faster_whisper import WhisperModel  # type: ignore
import os


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


def transcribe_video_to_srt(
    video_path: str,
    srt_save_path: str,
    model_size: str = "base",
    language: str = "de",
) -> None:
    """
    Transcribes a video file to SRT format using faster-whisper and saves it .

    Args:
        video_path: Path to the input video file.
        srt_save_path: Path where the output .srt file should be saved.
        model_size: Model variant to use (tiny, base, small, medium, large).
        language: Language code (e.g., "de", "en" etc.).
    """
    whisper_model: WhisperModel = WhisperModel(model_size, compute_type="auto", cpu_threads=os.cpu_count())

    segments, _ = whisper_model.transcribe(audio=video_path, language=language, log_progress=True)

    with open(srt_save_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(segments, start=1):
            srt_file.write(f"{i}\n")
            srt_file.write(f"{seconds_to_srt_time(segment.start)} --> {seconds_to_srt_time(segment.end)}\n")
            srt_file.write(f"{segment.text.strip()}\n\n")


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


def parse_srt_file(srt_path: str) -> list[SRTEntry]:
    """
    Parses an SRT subtitle file into a list of structured entries.

    Args:
        srt_path: Path to the .srt file to parse.

    Returns:
        A list of subtitle entries, each with index, start time, end time, and text.
    """
    with open(srt_path, "r", encoding="utf-8") as f:
        content: str = f.read()

    entries_raw: list[str] = re.split(r"\n\n+", content.strip())
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
    srt_path: str,
    slide_timestamps: dict[int, float],
) -> str:
    """
    Merge SRT entries into slide-aligned blocks based on slide start timestamps.

    Args:
        srt_path: Path to original .srt file.
        slide_timestamps: Mapping of slide index -> start time (in seconds).

    Returns:
        A string in SRT format where entries are grouped by slide time ranges.
    """
    srt_entries: list[SRTEntry] = parse_srt_file(srt_path)

    sorted_slide_entries: list[tuple[int, float]] = sorted(slide_timestamps.items(), key=lambda item: item[1])
    video_end_time: float = max(entry["end"] for entry in srt_entries)

    merged_blocks: list[SlideBlock] = []

    for i, (slide_index, start_time) in enumerate(sorted_slide_entries):
        if i + 1 < len(sorted_slide_entries):
            _, end_time = sorted_slide_entries[i + 1]
        else:
            end_time = video_end_time

        # Sanity check to avoid inverted timestamps
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
        start_str: str = seconds_to_srt_time(block["start"])
        end_str: str = seconds_to_srt_time(block["end"])
        merged_srt_lines.extend([str(block["index"]), f"{start_str} --> {end_str}", block["text"], ""])

    return "\n".join(merged_srt_lines)
