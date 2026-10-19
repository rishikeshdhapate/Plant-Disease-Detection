/**
 * FloraGuard AI - Plant Disease Detection
 * Client-Side Upload Interaction, Drag-and-Drop, Preview, & Loading Management
 */

document.addEventListener('DOMContentLoaded', function () {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('leafFileInput');
    const uploadPrompt = document.getElementById('uploadPrompt');
    const uploadPreview = document.getElementById('uploadPreview');
    const previewImg = document.getElementById('previewImg');
    const previewFileName = document.getElementById('previewFileName');
    const previewFileSize = document.getElementById('previewFileSize');
    const removeImgBtn = document.getElementById('removeImgBtn');
    const resetBtn = document.getElementById('resetBtn');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('leafUploadForm');
    const processingFeedback = document.getElementById('processingFeedback');
    const formActions = document.getElementById('formActions');

    if (!dropZone || !fileInput) {
        return; // Not on the detection page
    }

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'];

    // Format bytes to readable string (KB / MB)
    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Process and validate selected file
    function handleSelectedFile(file) {
        if (!file) return;

        // Check file extension
        const extension = file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(extension)) {
            alert('Invalid file format. Please upload a JPG, JPEG, PNG, or WEBP image.');
            resetUploadState();
            return;
        }

        // Check file size
        if (file.size > MAX_FILE_SIZE) {
            alert('File is too large! Maximum image size allowed is 16 MB.');
            resetUploadState();
            return;
        }

        // Render preview via FileReader
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImg.src = e.target.result;
            previewFileName.textContent = file.name;
            previewFileSize.textContent = formatBytes(file.size);

            uploadPrompt.classList.add('d-none');
            uploadPreview.classList.remove('d-none');
            if (resetBtn) resetBtn.classList.remove('d-none');
            if (submitBtn) {
                submitBtn.removeAttribute('disabled');
                submitBtn.focus();
            }
        };
        reader.readAsDataURL(file);
    }

    // Reset upload component back to initial state
    function resetUploadState() {
        fileInput.value = '';
        previewImg.src = '#';
        uploadPreview.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        if (resetBtn) resetBtn.classList.add('d-none');
        if (submitBtn) submitBtn.setAttribute('disabled', 'disabled');
    }

    // Listen to standard file input changes
    fileInput.addEventListener('change', function (e) {
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag-and-Drop Event Listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-active');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-active');
        }, false);
    });

    dropZone.addEventListener('drop', function (e) {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            fileInput.files = dt.files;
            handleSelectedFile(dt.files[0]);
        }
    }, false);

    // Remove selected image
    if (removeImgBtn) {
        removeImgBtn.addEventListener('click', function (e) {
            e.preventDefault();
            e.stopPropagation();
            resetUploadState();
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', function (e) {
            e.preventDefault();
            resetUploadState();
        });
    }

    // Form Submission: Show Loading Spinner State
    if (uploadForm) {
        uploadForm.addEventListener('submit', function (e) {
            if (!fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an image before submitting.');
                return;
            }

            // Show loading animation and disable double submit
            dropZone.classList.add('d-none');
            formActions.classList.add('d-none');
            processingFeedback.classList.remove('d-none');
        });
    }
});
