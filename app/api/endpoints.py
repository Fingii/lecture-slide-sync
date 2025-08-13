import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from starlette.background import BackgroundTask

from app.core.file_utils import (
    find_matching_video_pdf_pairs,
    extract_zip,
    save_upload_to_file,
    copy_file,
    save_str_to_file,
    zipping_directory,
)
from app.core.processing import generate_merged_srt
from app.core.video_chapter_embedder import generate_video_with_chapters
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
    generate_chapters: bool = Form(False),
):
    logger.info("Starting detection for single video pdf pair")
    # NOTE:
    # - We use a TemporaryDirectory for intermediate work; it auto-deletes at 'with' exit.
    # - We DO NOT stream directly from the tmp dir, because FileResponse reads the file AFTER
    #   this function returns, and the tmp dir would be gone.
    # - Instead, we write the final artifact (SRT or ZIP) into MEDIA_FOLDER for a stable path,
    #   return it via FileResponse, and delete that final artifact with a BackgroundTask.
    # - The uploaded video saved in MEDIA_FOLDER is PERMANENT and must NOT be deleted.
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir: Path = Path(tmp_dir_str)

        results_dir_path: Path = tmp_dir / "results"
        results_dir_path.mkdir()

        try:
            keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
            uploaded_video_path: Path = save_upload_to_file(uploaded_video, MEDIA_FOLDER)  # permanent
            uploaded_pdf_path: Path = save_upload_to_file(uploaded_pdf, tmp_dir)  # temporary

            merged_srt, slide_changes = generate_merged_srt(
                video_file_path=uploaded_video_path,
                pdf_file_path=uploaded_pdf_path,
                keywords_to_be_matched=keywords_set,
                sampling_interval_seconds=sampling_interval,
            )
            srt_filename: str = f"{uploaded_video_path.stem}_merged.srt"
            tmp_merged_srt_path: Path = save_str_to_file(merged_srt, results_dir_path / srt_filename)

            if generate_chapters:
                generate_video_with_chapters(
                    slide_changes=slide_changes,
                    input_video_path=uploaded_video_path,
                    output_dir=results_dir_path,
                )
                # Save zip in MEDIA_FOLDER for convenience, will be deleted after sending
                final_path = zipping_directory(results_dir_path, MEDIA_FOLDER / "results.zip")
            else:
                # Save SRT in MEDIA_FOLDER for convenience, will be deleted after sending
                final_path = copy_file(tmp_merged_srt_path, MEDIA_FOLDER)

            filename: str = "result.zip" if generate_chapters else f"{uploaded_video_path.stem}_merged.srt"
            logger.info(f"Detection done: Streaming {filename}")

            return FileResponse(
                path=final_path,
                media_type="application/zip" if generate_chapters else "text/plain",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
                background=BackgroundTask(final_path.unlink, missing_ok=True),
            )

        except Exception as e:
            logger.exception("Error during /detect processing")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-detect")
async def batch_detect(
    uploaded_zip: UploadFile = File(...),
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
    generate_chapters: bool = Form(False),
):
    logger.info(f"Starting batch detection for ZIP file: {uploaded_zip.filename}")
    # See NOTE in /detect: we persist final artifact to MEDIA_FOLDR (stable path)
    # and delete it after send; tmp dir is only for intermediates.
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir: Path = Path(tmp_dir_str)
        results_dir_path: Path = tmp_dir / "results"
        results_dir_path.mkdir()

        try:
            keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
            zip_path: Path = save_upload_to_file(uploaded_zip, tmp_dir)
            extracted_zip_dir: Path = extract_zip(zip_path)

            video_pdf_pairs: list[tuple[Path, Path]] = find_matching_video_pdf_pairs(extracted_zip_dir)

            for video_path, pdf_path in video_pdf_pairs:
                try:
                    merged_srt, slide_changes = generate_merged_srt(
                        video_file_path=video_path,
                        pdf_file_path=pdf_path,
                        keywords_to_be_matched=keywords_set,
                        sampling_interval_seconds=sampling_interval,
                    )

                    copy_file(video_path, MEDIA_FOLDER)
                    srt_filename: str = f"{video_path.stem}_merged.srt"
                    save_str_to_file(merged_srt, results_dir_path / srt_filename)

                    if generate_chapters:
                        generate_video_with_chapters(
                            slide_changes=slide_changes,
                            input_video_path=video_path,
                            output_dir=results_dir_path,
                        )

                except Exception as e:
                    logger.error("Error processing %s: %s", video_path.name, str(e))

            # Save zip in MEDIA_FOLDER for convenience, will be deleted after sending
            final_path = zipping_directory(results_dir_path, MEDIA_FOLDER / "batch_results.zip")
            logger.info(f"Batch processing complete. Streaming result zip")
            return FileResponse(
                path=final_path,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=batch_results.zip"},
                background=BackgroundTask(final_path.unlink, missing_ok=True),
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
