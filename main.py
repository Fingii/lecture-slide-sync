import shutil
import tempfile
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

from slide_detection import detect_slide_transitions
from srt_utils import merge_srt_by_slide_ranges, transcribe_video_to_srt
from logs.logging_config import configure_loggers, logger

configure_loggers()
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slide Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f4f4;
                border-radius: 8px;
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%;
                margin-bottom: 15px;
            }
            button {
                padding: 10px 20px;
                margin-right: 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            #result, #status {
                margin-top: 20px;
                white-space: pre-wrap;
                background-color: #e2e2e2;
                padding: 10px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <h2>Slide Change Detector</h2>
        <form id="uploadForm">
            <label>Video file:</label>
            <input type="file" name="video" required><br>
            <label>PDF file:</label>
            <input type="file" name="pdf" required><br>
            <label>Keywords (space separated):</label>
            <input type="text" name="keywords" value="FH AACHEN UNIVERSITY OF APPLIED SCIENCES" required><br>
            <label>Sampling interval (seconds):</label>
            <input type="number" name="sampling_interval" value="1.0" step="0.1" min="0.1" required><br>
            <button type="submit">Detect</button>
            <button type="button" id="cancelBtn" style="background-color:#dc3545;">Cancel</button>
        </form>

        <div id="status">Waiting for upload...</div>
        <div id="result"></div>

        <script>
            const form = document.getElementById("uploadForm");
            const status = document.getElementById("status");
            const result = document.getElementById("result");
            const cancelBtn = document.getElementById("cancelBtn");

            let controller = null;

            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                status.innerText = "Uploading and processing... ⏳";
                result.innerText = "";

                const formData = new FormData(form);
                controller = new AbortController();

                try {
                    const res = await fetch("/detect", {
                        method: "POST",
                        body: formData,
                        signal: controller.signal,
                    });

                    if (!res.ok) {
                        throw new Error("Server error");
                    }

                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = "merged.srt";
                    link.click();
                    status.innerText = "Detection finished. File downloaded.";
                } catch (err) {
                    if (err.name === "AbortError") {
                        status.innerText = "Request canceled";
                    } else {
                        status.innerText = "Error: " + err.message;
                    }
                } finally {
                    controller = null;
                }
            });

            cancelBtn.addEventListener("click", () => {
                if (controller) {
                    controller.abort();
                }
            });
        </script>
    </body>
    </html>
    """


@app.get("/batch", response_class=HTMLResponse)
def batch_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Slide Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 600px;
                margin: 40px auto;
                padding: 20px;
                background: #f4f4f4;
                border-radius: 8px;
            }
            input[type="file"], input[type="text"], input[type="number"] {
                width: 100%;
                margin-bottom: 15px;
            }
            button {
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            #status {
                margin-top: 20px;
                background-color: #e2e2e2;
                padding: 10px;
                border-radius: 4px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h2>Batch Slide Detection</h2>
        <form id="batchForm">
            <label>Upload ZIP (mp4, pdf, srt triplets):</label>
            <input type="file" name="zipfile_input" accept=".zip" required><br>

            <label>Keywords (space separated):</label>
            <input type="text" name="keywords" value="FH AACHEN UNIVERSITY OF APPLIED SCIENCES" required><br>

            <label>Sampling interval (seconds):</label>
            <input type="number" name="sampling_interval" value="1.0" step="0.1" min="0.1" required><br>

            <button type="submit">Upload & Process</button>
        </form>

        <div id="status">Waiting for ZIP upload...</div>

        <script>
            const form = document.getElementById("batchForm");
            const status = document.getElementById("status");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                status.innerText = "Uploading and processing... ⏳";

                const formData = new FormData(form);
                try {
                    const res = await fetch("/batch-detect", {
                        method: "POST",
                        body: formData
                    });

                    if (!res.ok) {
                        throw new Error("Server error");
                    }

                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement("a");
                    link.href = url;
                    link.download = "results.zip";
                    link.click();
                    status.innerText = "Download ready: results.zip";
                } catch (err) {
                    status.innerText = "Error: " + err.message;
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/detect")
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


@app.post("/batch-detect")
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
