import shutil
import tempfile
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.core.slide_detection import detect_slide_transitions
from app.core.srt_utils import merge_srt_by_slide_ranges, transcribe_video_to_srt
from logs.logging_config import logger

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/detect")
async def detect(
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
    video: UploadFile = File(...),
    pdf: UploadFile = File(...),
):
    keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
    temp_dir: Path = Path(tempfile.mkdtemp())

    try:
        assert video.filename is not None, "Video file must have a filename"
        assert pdf.filename is not None, "PDF file must have a filename"

        video_path: Path = temp_dir / f"video_{uuid.uuid4()}{Path(video.filename).suffix}"
        pdf_path: Path = temp_dir / f"pdf_{uuid.uuid4()}{Path(pdf.filename).suffix}"

        logger.debug(f"Saving files to temp location: {temp_dir}")
        with video_path.open("wb") as video_file, pdf_path.open("wb") as pdf_file:
            shutil.copyfileobj(video.file, video_file)
            shutil.copyfileobj(pdf.file, pdf_file)

        srt_content = transcribe_video_to_srt(video_path)
        slide_changes = detect_slide_transitions(
            video_file_path=video_path,
            pdf_file_path=pdf_path,
            keywords_to_be_matched=keywords_set,
            sampling_interval_seconds=sampling_interval,
        )
        merged_srt: str = merge_srt_by_slide_ranges(srt_content, slide_changes)
        logger.info("Algorithm finished: Streaming merged SRT file: merged.srt")
        return StreamingResponse(
            BytesIO(merged_srt.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=merged.srt"},
        )

    except Exception as e:
        logger.exception("Error during /detect processing")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/batch-detect")
async def batch_detect(
    zipfile_input: UploadFile = File(...),
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
):
    keywords_set = {k.strip().upper() for k in keywords.split()}
    temp_dir = Path(tempfile.mkdtemp())

    logger.info(f"Starting batch detection for ZIP file: {zipfile_input.filename}")
    try:
        zip_path = temp_dir / f"input_{uuid.uuid4()}.zip"
        with zip_path.open("wb") as f:
            shutil.copyfileobj(zipfile_input.file, f)
        logger.debug(f"Saved ZIP to {zip_path}")
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.debug(f"Extracted ZIP to {extract_dir}")
        output_dir = temp_dir / "merged"
        output_dir.mkdir()
        durations: list[tuple[str, str | float]] = []

        for video_path in extract_dir.glob("*.mp4"):
            pdf_path = extract_dir / f"{video_path.stem}.pdf"
            if not pdf_path.exists():
                logger.warning(f"No matching PDF found for {video_path.name}")
                continue

            start_time = time.time()
            try:
                slide_changes = detect_slide_transitions(
                    video_file_path=video_path,
                    pdf_file_path=pdf_path,
                    keywords_to_be_matched=keywords_set,
                    sampling_interval_seconds=sampling_interval,
                )
                srt_content = transcribe_video_to_srt(video_path)
                merged_srt = merge_srt_by_slide_ranges(srt_content, slide_changes)

                # Save results
                output_path = output_dir / f"{video_path.stem}_merged.srt"
                output_path.write_text(merged_srt, encoding="utf-8")
                durations.append((video_path.name, round(time.time() - start_time, 2)))

            except Exception as e:
                logger.error("Error processing %s: %s", video_path.name, str(e))
                durations.append((video_path.name, f"Error: {str(e)}"))

        # Create results zip
        zip_out_path = temp_dir / "results.zip"
        with zipfile.ZipFile(zip_out_path, "w") as zipf:
            for srt_file in output_dir.glob("*.srt"):
                zipf.write(srt_file, arcname=srt_file.name)

            timings_content = "\n".join(f"{name}: {time}" for name, time in durations)
            zipf.writestr("timings.txt", timings_content)

        logger.info(f"Batch processing complete. Streaming result zip")
        return StreamingResponse(
            zip_out_path.open("rb"),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=results.zip"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
