document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.querySelector('.browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const previewVideo = document.getElementById('preview-video');
    const fileName = document.getElementById('file-name');
    const changeFileBtn = document.getElementById('change-file-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loader');
    const resultCard = document.getElementById('result-card');
    const resetBtn = document.getElementById('reset-btn');

    let currentFile = null;

    // Trigger file input when clicking browse text
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent bubbling if dropZone also has click listener
        fileInput.click();
    });

    // Trigger file input when clicking the drop zone (optional, but good UX)
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle file selection via input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and Drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        fileName.textContent = file.name;
        
        // Hide upload area, show preview
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        resultCard.classList.add('hidden'); // Hide previous results if any

        // Show preview based on type
        if (file.type.startsWith('image/')) {
            previewImage.src = URL.createObjectURL(file);
            previewImage.classList.remove('hidden');
            previewVideo.classList.add('hidden');
        } else if (file.type.startsWith('video/')) {
            previewVideo.src = URL.createObjectURL(file);
            previewVideo.classList.remove('hidden');
            previewImage.classList.add('hidden');
        }
    }

    // Change File
    changeFileBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = ''; // Reset input
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultCard.classList.add('hidden');
        previewImage.src = '';
        previewVideo.src = '';
    });

    // Analyze Button
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        loader.classList.remove('hidden');
        resultCard.classList.add('hidden');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        const endpoint = currentFile.type.startsWith('image/') ? '/detect/image' : '/detect/video';

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const result = await response.json();
            showResult(result);

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        } finally {
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    // Reset Button (Analyze Another)
    resetBtn.addEventListener('click', () => {
        changeFileBtn.click();
    });

    function showResult(data) {
        loader.classList.add('hidden');
        resultCard.classList.remove('hidden');
    
        const score = Math.round(data.score * 100);
        const label = data.label;
    
        // Update Badge
        const badge = document.getElementById('result-badge');
        badge.textContent = label;
        badge.className = `badge ${label.toLowerCase()}`; // Ensure lowercase for CSS class
    
        // Update Circle
        document.getElementById('score-value').textContent = score;
        const circle = document.getElementById('score-circle-path');
        // Stroke color based on score
        const color = score > 50 ? '#ef4444' : '#22c55e';
        circle.style.stroke = color;
        circle.style.strokeDasharray = `${score}, 100`;
    
        // Update details if available
        if (data.details) {
            document.getElementById('ela-bar').style.width = `${data.details.ela_score * 100}%`;
            document.getElementById('model-bar').style.width = `${data.details.model_confidence * 100}%`;
        } else if (data.frame_scores) {
            // For video, maybe show average in bars for now
            document.getElementById('ela-bar').style.width = `${score}%`; // Simplified for video
            document.getElementById('model-bar').style.width = `${score}%`;
        }
    }
});

