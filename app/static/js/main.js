document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Processing mode and files
    let processingMode = null; // 'single' or 'batch'
    let videoFile = null;
    let pdfFile = null;
    let zipFile = null;
    let currentStep = 1;
    const totalSteps = 3;
    let processingInterval = null;

    // DOM elements
    const progressLine = document.getElementById('progressLine');
    const step1 = document.getElementById('step1');
    const step2 = document.getElementById('step2');
    const step3 = document.getElementById('step3');
    const stepContents = document.querySelectorAll('.step-content');
    const nextToStep2 = document.getElementById('nextToStep2');
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    const dropZone = document.getElementById('dropZone');
    const intervalRange = document.getElementById('samplingIntervalRange');
    const intervalInput = document.getElementById('samplingInterval');
    const intervalValue = document.getElementById('intervalValue');

    intervalRange.addEventListener('input', () => {
        intervalInput.value = intervalRange.value;
        intervalValue.textContent = intervalRange.value;
    });

    intervalInput.addEventListener('input', () => {
        intervalRange.value = intervalInput.value;
        intervalValue.textContent = intervalInput.value;
    });

    // Keywords Tag System
    const keywordsInput = document.getElementById('keywordInput');
    const addKeywordBtn = document.getElementById('addKeywordBtn');
    const keywordsTags = document.getElementById('keywordsTags');
    const hiddenKeywords = document.getElementById('keywords');

    function updateKeywords() {
        const keywords = Array.from(keywordsTags.querySelectorAll('.keyword-tag'))
            .map(tag => tag.dataset.value);
        hiddenKeywords.value = keywords.join(' ');
    }

    function addKeyword(keyword) {
        if (!keyword.trim()) return;

        const tag = document.createElement('div');
        tag.className = 'keyword-tag badge bg-primary me-1 mb-1 d-flex align-items-center';
        tag.innerHTML = `
            ${keyword}
            <button type="button" class="btn-close btn-close-white ms-2" aria-label="Remove"></button>
        `;
        tag.dataset.value = keyword;

        tag.querySelector('button').addEventListener('click', () => {
            tag.remove();
            updateKeywords();
        });

        keywordsTags.appendChild(tag);
        updateKeywords();
    }

    // Add keyword on button click or Enter
    addKeywordBtn.addEventListener('click', () => {
        addKeyword(keywordsInput.value.trim());
        keywordsInput.value = '';
    });

    keywordsInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addKeyword(keywordsInput.value.trim());
            keywordsInput.value = '';
            e.preventDefault();
        }
    });

    // Initialize with default keywords if needed
    const defaultKeywords = ["FH", "AACHEN", "UNIVERSITY", "OF", "APPLIED", "SCIENCES"];
    defaultKeywords.forEach(kw => addKeyword(kw));

    // Initialize the progress line
    updateProgressLine();

    // File handling function
    function handleFiles(files) {
        const fileArray = Array.from(files);

        // Check each file and assign to appropriate variable
        fileArray.forEach(file => {
            if (file.type.startsWith('video/')) {
                videoFile = file;
            } else if (file.name.toLowerCase().endsWith('.pdf')) {
                pdfFile = file;
            } else if (file.name.toLowerCase().endsWith('.zip')) {
                zipFile = file;
            }
        });

        // Validate the file combination
        if (zipFile && fileArray.length === 1) {
            processingMode = 'batch';
        } else if (videoFile && pdfFile) {
            processingMode = 'single';
        } else {
            alert('Please upload either:\n- A video file\n- A PDF file\n- Or both together\n- Or a single ZIP file');
            fileInput.value = '';
            return;
        }

        updateFileDisplay();
    }

    function updateFileDisplay() {
        fileInfo.innerHTML = '';
        fileInfo.classList.remove('d-none');

        if (processingMode === 'batch') {
            fileInfo.innerHTML = `
                <div class="alert alert-info d-flex justify-content-between align-items-center">
                    <span><i class="bi bi-file-earmark-zip-fill me-2"></i>${zipFile.name}</span>
                    <button type="button" class="btn btn-sm btn-outline-danger clear-file" data-type="zip">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            `;
        } else {
            if (videoFile) {
                fileInfo.innerHTML += `
                    <div class="alert alert-info d-flex justify-content-between align-items-center mb-2">
                        <span><i class="bi bi-file-earmark-play-fill me-2"></i>${videoFile.name}</span>
                        <button type="button" class="btn btn-sm btn-outline-danger clear-file" data-type="video">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                `;
            }
            if (pdfFile) {
                fileInfo.innerHTML += `
                    <div class="alert alert-info d-flex justify-content-between align-items-center">
                        <span><i class="bi bi-file-earmark-pdf-fill me-2"></i>${pdfFile.name}</span>
                        <button type="button" class="btn btn-sm btn-outline-danger clear-file" data-type="pdf">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                `;
            }
        }

        // Add event listeners to clear buttons
        document.querySelectorAll('.clear-file').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                clearFile(btn.dataset.type);
            });
        });

        // Enable Next button only if we have valid files
        checkFilesReady();
    }

    function clearFile(type) {
        if (type === 'video') videoFile = null;
        if (type === 'pdf') pdfFile = null;
        if (type === 'zip') zipFile = null;
        fileInput.value = '';
        updateFileDisplay();
    }

    function resetFileState() {
        processingMode = null;
        videoFile = pdfFile = zipFile = null;
        fileInfo.classList.add('d-none');
        nextToStep2.disabled = true;
    }

    function checkFilesReady() {
        nextToStep2.disabled = !(
            (processingMode === 'batch' && zipFile) ||
            (processingMode === 'single' && videoFile && pdfFile)
        );
    }

    // File input change handler
    fileInput.addEventListener('change', function(e) {
        handleFiles(e.target.files);
    });

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, function(e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, function() {
            dropZone.classList.add('active');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, function() {
            dropZone.classList.remove('active');
        });
    });

    dropZone.addEventListener('drop', function(e) {
        handleFiles(e.dataTransfer.files);
    });

    // Step navigation functions
    function goToStep(step) {
        stepContents.forEach(content => content.classList.remove('active'));
        currentStep = step;

        [step1, step2, step3].forEach((stepEl, index) => {
            stepEl.classList.remove('active', 'completed');
            if (index + 1 < currentStep) {
                stepEl.classList.add('completed');
            } else if (index + 1 === currentStep) {
                stepEl.classList.add('active');
            }
        });

        document.getElementById(`step${step}-content`).classList.add('active');
        updateProgressLine();
    }

    function updateProgressLine() {
        const progressPercentage = ((currentStep - 1) / (totalSteps - 1)) * 100;
        progressLine.style.width = `${progressPercentage}%`;
    }

    // Step navigation event listeners
    nextToStep2.addEventListener('click', () => goToStep(2));

    document.getElementById('nextToStep3').addEventListener('click', () => {
        document.getElementById('reviewFileNames').textContent =
            processingMode === 'single'
                ? `Video: ${videoFile.name}, PDF: ${pdfFile.name}`
                : `ZIP: ${zipFile.name}`;
        document.getElementById('reviewKeywords').textContent = document.getElementById('keywords').value;
        document.getElementById('reviewInterval').textContent =
            `${document.getElementById('samplingInterval').value} seconds`;
          const chaptersOn = document.getElementById('generateChapters').checked;
          document.getElementById('reviewChapters').innerHTML = chaptersOn
            ? '<span class="badge bg-success">On</span> — MP4 with chapter markers'
            : '<span class="badge bg-secondary">Off</span> - SRT only';
        goToStep(3);
    });

    document.getElementById('backToStep1').addEventListener('click', () => goToStep(1));
    document.getElementById('backToStep2').addEventListener('click', () => goToStep(2));

    // Process button handler
    document.getElementById('processBtn').addEventListener('click', async function(e) {
        e.preventDefault();

        if (!processingMode) {
            alert('Please select files first');
            return;
        }

        // Show processing screen
        document.getElementById('step3-content').classList.remove('active');
        document.getElementById('processing-content').classList.add('active');

        // Start progress animation
        let progress = 0;
        document.getElementById('processingBar').style.width = '0%';
        processingInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 90) progress = 90;
            document.getElementById('processingBar').style.width = `${progress}%`;
            document.getElementById('processingStatus').textContent = getProcessingMessage(progress);
        }, 800);

        try {
            const formData = new FormData();
            formData.append('keywords', document.getElementById('keywords').value);
            formData.append('sampling_interval', document.getElementById('samplingInterval').value);
            formData.append('generate_chapters', document.getElementById('generateChapters').checked ? 'true' : 'false');

            const endpoint = processingMode === 'single' ? '/detect' : '/batch-detect';

            if (processingMode === 'single') {
                formData.append('uploaded_video', videoFile);
                formData.append('uploaded_pdf', pdfFile);
            } else {
                formData.append('uploaded_zip', zipFile);
            }

            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Server responded with ${response.status}`);

            clearInterval(processingInterval);
            document.getElementById('processingBar').style.width = '100%';
            document.getElementById('processingStatus').textContent = 'Processing complete! Streaming result, this may take a while';

            const contentDisposition = response.headers.get('Content-Disposition');
            const filename = contentDisposition
                ? contentDisposition.split('filename=')[1].replace(/"/g, '')
                : (processingMode === 'single' ? 'merged.srt' : 'results.zip');

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);

            document.getElementById('downloadLink').href = url;
            document.getElementById('downloadLink').setAttribute('download', filename);

            setTimeout(() => {
                document.getElementById('processing-content').classList.remove('active');
                document.getElementById('results-content').classList.add('active');
            }, 1000);

        } catch (error) {
            console.error('Error:', error);
            clearInterval(processingInterval);
            document.getElementById('processingStatus').innerHTML =
                `<div class="alert alert-danger">Error processing files: ${error.message}</div>`;
        }
    });

    function getProcessingMessage(progress) {
        const messages = processingMode === 'single' ? [
            "Analyzing video content...",
            "Extracting slide information...",
            "Matching slides to video frames...",
            "Generating synchronized subtitles...",
            "Finalizing output..."
        ] : [
            "Extracting ZIP contents...",
            "Processing first video...",
            "Matching slides across lectures...",
            "Generating synchronized subtitles...",
            "Compressing results..."
        ];

        const index = Math.min(Math.floor(progress / 20), messages.length - 1);
        return messages[index];
    }

    // Cancel processing
    document.getElementById('cancelProcessing').addEventListener('click', () => {
        clearInterval(processingInterval);
        document.getElementById('processing-content').classList.remove('active');
        document.getElementById('step3-content').classList.add('active');
    });

    // Start new process
    document.getElementById('newProcess').addEventListener('click', () => {
        resetFileState();
        document.getElementById('keywords').value = 'FH AACHEN UNIVERSITY OF APPLIED SCIENCES';
        document.getElementById('samplingInterval').value = '1.0';
        document.getElementById('results-content').classList.remove('active');
        goToStep(1);
    });
});