import tempfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.core.file_utils import (
    find_matching_video_pdf_pairs,
    extract_zip,
    save_upload_to_file,
    copy_file,
    save_str_to_file,
    zipping_directory,
)
from app.core.processing import generate_merged_srt
from logs.logging_config import logger

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_FOLDER = PROJECT_ROOT / "media"


@router.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/detect")
async def detect(
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
    uploaded_video: UploadFile = File(...),
    uploaded_pdf: UploadFile = File(...),
):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logger.info("Starting detection for single video pdf pair")
        tmp_dir: Path = Path(tmp_dir_str)

        try:
            keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
            uploaded_video_path: Path = save_upload_to_file(uploaded_video, MEDIA_FOLDER)  # permanent
            uploaded_pdf_path: Path = save_upload_to_file(uploaded_pdf, tmp_dir)  # temporary

            merged_srt: str = generate_merged_srt(
                video_file_path=uploaded_video_path,
                pdf_file_path=uploaded_pdf_path,
                keywords_to_be_matched=keywords_set,
                sampling_interval_seconds=sampling_interval,
            )

            logger.info(f"Detection done: Streaming merged SRT file: {uploaded_video_path.stem}_merged.srt")
            return StreamingResponse(
                BytesIO(merged_srt.encode("utf-8")),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename={uploaded_video_path.stem}_merged.srt"
                },
            )

        except Exception as e:
            logger.exception("Error during /detect processing")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-detect")
async def batch_detect(
    uploaded_zip: UploadFile = File(...),
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
):
    logger.info(f"Starting batch detection for ZIP file: {uploaded_zip.filename}")
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir: Path = Path(tmp_dir_str)

        try:
            keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
            zip_path: Path = save_upload_to_file(uploaded_zip, tmp_dir)
            extracted_zip_dir: Path = extract_zip(zip_path)

            video_pdf_pairs: list[tuple[Path, Path]] = find_matching_video_pdf_pairs(extracted_zip_dir)
            results_dir_path: Path = tmp_dir / "results"
            results_dir_path.mkdir()

            for video_path, pdf_path in video_pdf_pairs:
                try:
                    merged_srt: str = generate_merged_srt(
                        video_file_path=video_path,
                        pdf_file_path=pdf_path,
                        keywords_to_be_matched=keywords_set,
                        sampling_interval_seconds=sampling_interval,
                    )

                    copy_file(video_path, MEDIA_FOLDER)
                    srt_filename: str = f"{video_path.stem}_merged.srt"
                    save_str_to_file(merged_srt, results_dir_path / srt_filename)
                except Exception as e:
                    logger.error("Error processing %s: %s", video_path.name, str(e))

            result_zip_path: Path = zipping_directory(results_dir_path)
            logger.info(f"Batch processing complete. Streaming result zip")
            return StreamingResponse(
                BytesIO(result_zip_path.read_bytes()),
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=batch_results.zip"},
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
