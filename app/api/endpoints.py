import tempfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.core.file_utils import (
    save_file,
    find_matching_video_pdf_pairs,
    create_zip_from_content,
    extract_zip,
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
            uploaded_video_path: Path = save_file(uploaded_video, MEDIA_FOLDER)  # permanent
            uploaded_pdf_path: Path = save_file(uploaded_pdf, tmp_dir)  # temporary

            merged_srt: str = generate_merged_srt(
                video_file_path=uploaded_video_path,
                pdf_file_path=uploaded_pdf_path,
                keywords_to_be_matched=keywords_set,
                sampling_interval_seconds=sampling_interval,
            )

            logger.info(f"Detection done: Streaming merged SRT file: {uploaded_video_path.name}_merged.srt")
            return StreamingResponse(
                BytesIO(merged_srt.encode("utf-8")),
                media_type="text/plain",
                headers={
                    "Content-Disposition": f"attachment; filename={uploaded_video_path.name}_merged.srt"
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

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir: Path = Path(tmp_dir_str)
        logger.info(f"Starting batch detection for ZIP file: {uploaded_zip.filename}")

        try:
            keywords_set: set[str] = {k.strip().upper() for k in keywords.split()}
            zip_path: Path = save_file(uploaded_zip, tmp_dir)
            extracted_zip_dir: Path = extract_zip(zip_path)

            video_pdf_pairs: list[tuple[Path, Path]] = find_matching_video_pdf_pairs(extracted_zip_dir)
            srt_contents: dict[str, str] = {}  # {filename: content}

            for video_path, pdf_path in video_pdf_pairs:
                try:
                    merged_srt: str = generate_merged_srt(
                        video_file_path=video_path,
                        pdf_file_path=pdf_path,
                        keywords_to_be_matched=keywords_set,
                        sampling_interval_seconds=sampling_interval,
                    )

                    save_file(video_path, MEDIA_FOLDER)
                    filename: str = video_path.stem
                    srt_contents[f"{filename}_merged.srt"] = merged_srt

                except Exception as e:
                    logger.error("Error processing %s: %s", video_path.name, str(e))

            zip_result = create_zip_from_content(srt_contents)

            logger.info(f"Batch processing complete. Streaming result zip")
            return StreamingResponse(
                zip_result,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=results.zip"},
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
