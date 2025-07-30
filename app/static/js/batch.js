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