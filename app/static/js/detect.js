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