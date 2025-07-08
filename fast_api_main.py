from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse

from typing import Annotated
import shutil
import os
import uuid


from slide_detection import detect_slide_transitions

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
            input[type="file"], input[type="text"] {
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
            <input type="text" name="keywords" placeholder="e.g. FH AACHEN SCIENCES" required><br>
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
                controller = new AbortController(); // Create new controller for this request

                try {
                    const res = await fetch("/detect", {
                        method: "POST",
                        body: formData,
                        signal: controller.signal,
                    });

                    if (!res.ok) {
                        throw new Error("Server error");
                    }

                    const json = await res.json();
                    status.innerText = "Detection finished";
                    result.innerText = JSON.stringify(json, null, 2);
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


@app.post("/detect")
async def detect(
    keywords: Annotated[str, Form(...)],
    video: UploadFile = File(...),
    pdf: UploadFile = File(...),
):
    # Convert string to set of uppercase words
    keywords_set: set[str] = set(k.strip().upper() for k in keywords.split())

    video_path: str = f"temp_{uuid.uuid4()}_{video.filename}"
    pdf_path: str = f"temp_{uuid.uuid4()}_{pdf.filename}"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

        slide_changes: dict[int, int] = detect_slide_transitions(
            video_file_path=video_path,
            pdf_file_path=pdf_path,
            keywords_to_be_matched=keywords_set,
        )

        return JSONResponse(content={"slide_changes": slide_changes})

    finally:
        for path in [video_path, pdf_path]:
            if os.path.exists(path):
                os.remove(path)
