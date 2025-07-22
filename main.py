import os
import shutil
import tempfile
import time
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

from slide_detection import detect_slide_transition_and_merge_srt

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
            <label>SRT file:</label>
            <input type="file" name="srt" required><br>
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
    srt: UploadFile = File(...),
):
    keywords_set = set(k.strip().upper() for k in keywords.split())

    video_path = f"temp_{uuid.uuid4()}_{video.filename}"
    pdf_path = f"temp_{uuid.uuid4()}_{pdf.filename}"
    srt_path = f"temp_{uuid.uuid4()}_{srt.filename}"

    try:
        for uploaded, path in [(video, video_path), (pdf, pdf_path), (srt, srt_path)]:
            with open(path, "wb") as f:
                shutil.copyfileobj(uploaded.file, f)

        merged_srt_str = detect_slide_transition_and_merge_srt(
            video_file_path=video_path,
            pdf_file_path=pdf_path,
            srt_file_path=srt_path,
            keywords_to_be_matched=keywords_set,
            sampling_interval_seconds=sampling_interval,
        )

        return StreamingResponse(
            BytesIO(merged_srt_str.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=merged.srt"},
        )

    finally:
        for path in [video_path, pdf_path, srt_path]:
            if os.path.exists(path):
                os.remove(path)


@app.post("/batch-detect")
async def batch_detect(
    zipfile_input: UploadFile = File(...),
    keywords: Annotated[str, Form()] = "FH AACHEN UNIVERSITY OF APPLIED SCIENCES",
    sampling_interval: Annotated[float, Form()] = 1.0,
):
    keywords_set = set(k.strip().upper() for k in keywords.split())

    temp_dir = Path(tempfile.mkdtemp())
    zip_path = temp_dir / f"{uuid.uuid4()}.zip"

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(zipfile_input.file, f)

    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    output_dir = temp_dir / "merged"
    output_dir.mkdir()

    durations = []
    for video in extract_dir.glob("*.mp4"):
        name_stem = video.stem
        pdf = extract_dir / f"{name_stem}.pdf"
        srt = extract_dir / f"{name_stem}.srt"

        if not (pdf.exists() and srt.exists()):
            continue

        start_time = time.time()
        merged_srt = detect_slide_transition_and_merge_srt(
            str(video),
            str(pdf),
            str(srt),
            keywords_to_be_matched=keywords_set,
            sampling_interval_seconds=sampling_interval,
        )
        duration = time.time() - start_time
        durations.append((video.name, round(duration, 2)))

        merged_path = output_dir / f"{name_stem}_merged.srt"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_srt)

    # Write timing log
    timing_path = output_dir / "timings.txt"
    with open(timing_path, "w") as f:
        for name, sec in durations:
            f.write(f"{name}: {sec} seconds\n")

    # Zip output
    zip_out_path = temp_dir / "results.zip"
    with zipfile.ZipFile(zip_out_path, "w") as zipf:
        for srt_file in output_dir.glob("*.srt"):
            zipf.write(srt_file, arcname=srt_file.name)
        zipf.write(timing_path, arcname="timings.txt")

    return StreamingResponse(
        open(zip_out_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=results.zip"},
    )
