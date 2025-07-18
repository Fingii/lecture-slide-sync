import re
from datetime import timedelta
from typing import Dict, List


def seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp format."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


def parse_srt_file(srt_path: str) -> List[dict]:
    """Parses an SRT file into a list of entries with start, end, and text."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    entries_raw = re.split(r"\n\n+", content.strip())
    entries = []

    for entry in entries_raw:
        lines = entry.strip().splitlines()
        if len(lines) < 3:
            continue

        index = int(lines[0])
        times = lines[1]
        text = " ".join(lines[2:]).strip()

        start_str, end_str = times.split(" --> ")
        h, m, s_ms = start_str.split(":")
        s, ms = s_ms.split(",")
        start_sec = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        h, m, s_ms = end_str.split(":")
        s, ms = s_ms.split(",")
        end_sec = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        entries.append({"index": index, "start": start_sec, "end": end_sec, "text": text})

    entries.sort(key=lambda x: x["start"])
    return entries


def merge_srt_by_slide_ranges(
    srt_path: str,
    slide_timestamps: Dict[int, float],
    video_end_time: float | None = None,
) -> str:
    """
    Merge SRT entries into slide-aligned blocks based on slide start timestamps.

    Args:
        srt_path: Path to original .srt file.
        slide_timestamps: Mapping of slide index -> start time (seconds).
        video_end_time: Optional total duration of video (in seconds).
    """
    srt_entries = parse_srt_file(srt_path)

    # Sort slide timestamps by their actual start time, not slide number
    sorted_slide_entries = sorted(slide_timestamps.items(), key=lambda item: item[1])

    merged_blocks = []

    for i, (slide_index, start_time) in enumerate(sorted_slide_entries):
        if i + 1 < len(sorted_slide_entries):
            _, end_time = sorted_slide_entries[i + 1]
        else:
            end_time = video_end_time or max(e["end"] for e in srt_entries)

        # Sanity check to avoid inverted timestamps
        if end_time <= start_time:
            print(f"⚠ Skipping slide {slide_index} due to invalid time range: {start_time} → {end_time}")
            continue

        block_texts = [e["text"] for e in srt_entries if start_time <= e["start"] < end_time]

        merged_blocks.append(
            {
                "index": i + 1,
                "start": start_time,
                "end": end_time,
                "text": "\n".join(block_texts),
            }
        )

    # Generate final string
    merged_srt_lines = []
    for block in merged_blocks:
        merged_srt_lines.append(f"{block['index']}")
        merged_srt_lines.append(
            f"{seconds_to_srt_time(block['start'])} --> {seconds_to_srt_time(block['end'])}"
        )
        merged_srt_lines.append(block["text"])
        merged_srt_lines.append("")  # Empty line between blocks

    return "\n".join(merged_srt_lines)
